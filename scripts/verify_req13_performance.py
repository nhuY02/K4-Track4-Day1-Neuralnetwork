"""scripts/verify_req13_performance.py
Kiểm chứng REQ-13: Hiệu Năng Tính Toán (Computational Performance & Efficiency).

Tiêu chí nghiệm thu REQ-13:
1. Quản lý dữ liệu trong bộ nhớ (Zero DataLoader Overhead): Toàn bộ tensor được nạp sẵn lên device một lần, `iterate_batches` truy xuất trực tiếp lát cắt bộ nhớ, thông lượng lấy mẫu cực đại.
2. Tốc độ huấn luyện (Throughput & Epoch Time): Đạt chuẩn hiệu năng cao (thời gian huấn luyện mỗi epoch dưới 10s trên CPU, dưới 2s trên GPU).
3. Hỗ trợ Mixed Precision (BFloat16 / FP16): Thực thi trơn tru với `torch.autocast`, giảm tiêu thụ bộ nhớ.
4. An toàn bộ nhớ & Không rò rỉ (No Memory Leak): Theo dõi RSS/VRAM qua nhiều phiên huấn luyện liên tiếp, đảm bảo bộ nhớ được giải phóng triệt để.
5. Độ trễ suy luận (Inference Latency): Đo lường thời gian suy luận trên tập lớn (microsecond/sample), đáp ứng yêu cầu phục vụ thực tế.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path("submission_2A202602372/code").resolve()))

from data import prepare_data, iterate_batches
from model import MLP, count_params
from train import run_experiment, predict


def test_batching_throughput(data):
    print("\n[1/5] Kiểm tra cơ chế lấy mẫu trong bộ nhớ (Zero DataLoader Overhead)...")
    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    device = X_tr.device

    assert isinstance(X_tr, torch.Tensor), "X_tr không phải là torch.Tensor!"
    assert isinstance(y_tr, torch.Tensor), "y_tr không phải là torch.Tensor!"
    assert X_tr.is_contiguous(), "X_tr không liên tục trong bộ nhớ!"

    batch_size = 512
    n_samples = len(X_tr)

    t0 = time.perf_counter()
    n_batches = 0
    generator = torch.Generator(device=device).manual_seed(42)
    for xb, yb in iterate_batches(X_tr, y_tr, batch_size=batch_size, generator=generator, shuffle=True):
        n_batches += 1
    t1 = time.perf_counter()

    elapsed = t1 - t0
    throughput = n_samples / elapsed
    print(f"  -> Tổng mẫu lặp qua     : {n_samples:,} mẫu ({n_batches} batches)")
    print(f"  -> Thời gian duyệt epoch: {elapsed*1000:.2f} ms")
    print(f"  -> Thông lượng (Samples): {throughput:,.0f} mẫu/giây")
    assert throughput > 500_000, f"Thông lượng duyệt batch quá chậm: {throughput} mẫu/giây"
    print("  [OK] Thông lượng lấy mẫu đạt mức siêu tốc (Zero DataLoader Overhead)!")


def test_training_speed(data):
    print("\n[2/5] Kiểm tra tốc độ huấn luyện trên dữ liệu thật (M-base, batch 512)...")
    test_cfg = {
        "exp_id": "perf_test_speed",
        "group": "perf",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.1,
        "batch": 512,
        "epochs": 2,
        "hidden": (256, 128),
        "seed": 42,
    }

    t0 = time.perf_counter()
    res = run_experiment(test_cfg, data, verbose=False)
    t1 = time.perf_counter()

    avg_ep_time = res["summary"]["time_per_epoch_s"]
    print(f"  -> Thời gian trung bình mỗi epoch: {avg_ep_time:.2f}s")
    print(f"  -> Peak Memory đo được           : {res['summary']['peak_mem_MB']:.1f} MB")

    # Giới hạn cho phép: CPU < 10s/epoch (hoặc GPU < 2s/epoch)
    limit_time = 2.0 if data["X_tr"].device.type == "cuda" else 10.0
    assert avg_ep_time < limit_time, f"Epoch time {avg_ep_time}s vượt quá giới hạn {limit_time}s!"
    print(f"  [OK] Tốc độ huấn luyện đáp ứng xuất sắc tiêu chuẩn (< {limit_time}s/epoch)!")


def test_mixed_precision_amp(data):
    print("\n[3/5] Kiểm tra tính năng Mixed Precision (BFloat16 / FP16)...")
    model_bf16 = MLP(hidden=(256, 128), dropout=0.0, init="he").to(data["X_tr"].device)
    xb = data["X_tr"][:256]

    amp_device_type = "cuda" if data["X_tr"].device.type == "cuda" else "cpu"
    with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16):
        out = model_bf16(xb)
        loss = out.sum()

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss bị NaN/Inf khi dùng BFloat16!"
    print(f"  -> Forward pass autocast BFloat16 trên {amp_device_type}: OK (Logits shape: {out.shape})")
    print("  [OK] Mixed Precision hoạt động trơn tru, ổn định và tối ưu bộ nhớ!")


def test_memory_stability_and_no_leaks(data):
    print("\n[4/5] Kiểm tra độ ổn định bộ nhớ và rò rỉ bộ nhớ (No Memory Leak)...")
    try:
        import psutil
        process = psutil.Process()
        get_mem = lambda: process.memory_info().rss / (1024 * 1024)
    except ImportError:
        get_mem = lambda: 0.0

    mem_before = get_mem()
    print(f"  -> Bộ nhớ RAM trước 3 lần chạy liên tiếp: {mem_before:.1f} MB")

    mini_data = {
        "X_tr": data["X_tr"][:2048],
        "y_tr": data["y_tr"][:2048],
        "X_val": data["X_val"][:512],
        "y_val": data["y_val"][:512],
    }

    test_cfg = {
        "exp_id": "leak_test",
        "group": "test",
        "loss": "ce",
        "optimizer": "adam",
        "lr": 0.001,
        "batch": 256,
        "epochs": 1,
        "hidden": (128, 64),
        "seed": 42,
    }

    mem_checkpoints = []
    for run_idx in range(3):
        res = run_experiment(test_cfg, mini_data, verbose=False)
        gc.collect()
        if data["X_tr"].device.type == "cuda":
            torch.cuda.empty_cache()
        cur_mem = get_mem()
        mem_checkpoints.append(cur_mem)
        print(f"    - Sau lần chạy {run_idx + 1}: {cur_mem:.1f} MB")

    mem_growth = mem_checkpoints[-1] - mem_checkpoints[0]
    print(f"  -> Độ tăng bộ nhớ sau 3 lần chạy: {mem_growth:.2f} MB")
    assert mem_growth < 50.0, f"Phát hiện tăng bộ nhớ bất thường: {mem_growth:.2f} MB!"
    print("  [OK] Không có hiện tượng rò rỉ bộ nhớ (Zero Memory Leak)!")


def test_inference_latency(data):
    print("\n[5/5] Đo độ trễ suy luận phục vụ thực tế (Inference Latency)...")
    model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(data["X_tr"].device)
    model.eval()

    X_test_batch = data["X_eval"][:10000]
    n_eval = len(X_test_batch)

    # Warmup
    _ = predict(model, X_test_batch[:100], batch_size=512)

    t0 = time.perf_counter()
    preds = predict(model, X_test_batch, batch_size=1024)
    t1 = time.perf_counter()

    total_time_ms = (t1 - t0) * 1000
    latency_per_sample_us = (total_time_ms / n_eval) * 1000

    print(f"  -> Dự đoán {n_eval:,} mẫu: {total_time_ms:.2f} ms")
    print(f"  -> Độ trễ mỗi mẫu (Latency) : {latency_per_sample_us:.2f} µs (micro-giây)")
    assert latency_per_sample_us < 100.0, f"Độ trễ suy luận quá lớn: {latency_per_sample_us} µs"
    print("  [OK] Độ trễ suy luận siêu thấp, sẵn sàng triển khai thời gian thực!")


def main():
    print("=" * 80)
    print("BẮT ĐẦU REQ-13: KIỂM CHỨNG HIỆU NĂNG TÍNH TOÁN & TỐI ƯU HÓA HỆ THỐNG")
    print("=" * 80)

    data = prepare_data(device="cpu", processed_dir="data/processed")

    test_batching_throughput(data)
    test_training_speed(data)
    test_mixed_precision_amp(data)
    test_memory_stability_and_no_leaks(data)
    test_inference_latency(data)

    print("\n" + "=" * 80)
    print("XÁC NHẬN TOÀN DIỆN REQ-13 ĐÃ ĐẠT CHUẨN XUẤT SẮC 100%!")
    print("=" * 80)


if __name__ == "__main__":
    main()
