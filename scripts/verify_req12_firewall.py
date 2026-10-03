"""scripts/verify_req12_firewall.py
Kiểm chứng REQ-12: Tính Toàn Vẹn Dữ Liệu & Quy Tắc Tường Lửa (Data Integrity / Firewall Rule).

Tiêu chí nghiệm thu REQ-12:
1. Quy tắc Tường lửa (Firewall Rule): Tập eval CHỈ dùng ở bước cuối cùng để chấm điểm. Tuyệt đối không xuất hiện trong huấn luyện, chọn cấu hình, chuẩn hóa hay early stopping.
2. Không rò rỉ dữ liệu (No Data Leakage): Standardizer (mean, std) CHỈ ĐƯỢC fit trên tập huấn luyện con (train_sub), không tính trên train_full, val hay eval.
3. Bảo toàn thuộc tính nhị phân: 10 cột đầu (liên tục) được chuẩn hóa z-score, 44 cột sau (nhị phân 0/1) giữ nguyên vẹn 100%.
4. Phân tầng chuẩn xác (Stratified Split): Tỷ lệ 7 lớp giữa train_sub và val khớp nhau và tương đồng với train_full.
5. Số lượng mẫu và kích thước tensor chính xác tuyệt đối:
   - train_sub: 371.847 x 54
   - val: 92.962 x 54
   - eval: 116.203 x 54
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path("submission_2A202602372/code").resolve()))

from data import load_split, make_val_split, fit_standardizer, apply_standardizer, prepare_data
from train import run_experiment


def test_sample_counts_and_shapes(data, raw_train_full, raw_eval):
    print("\n[1/5] Kiểm tra số lượng mẫu, kích thước và kiểu dữ liệu...")
    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_eval = data["X_eval"]
    y_eval = data["y_eval"]

    assert X_tr.shape == (371847, 54), f"Sai shape X_tr: {X_tr.shape}"
    assert y_tr.shape == (371847,), f"Sai shape y_tr: {y_tr.shape}"
    assert X_val.shape == (92962, 54), f"Sai shape X_val: {X_val.shape}"
    assert y_val.shape == (92962,), f"Sai shape y_val: {y_val.shape}"
    assert X_eval.shape == (116203, 54), f"Sai shape X_eval: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Sai shape y_eval: {y_eval.shape}"

    assert len(X_tr) + len(X_val) == 464809, "Tổng train_sub + val không bằng 464809!"
    assert len(X_eval) == 116203, "Eval set không đúng 116203 mẫu!"

    assert X_tr.dtype == torch.float32, f"Sai dtype X_tr: {X_tr.dtype}"
    assert y_tr.dtype == torch.int64, f"Sai dtype y_tr: {y_tr.dtype}"
    assert X_val.dtype == torch.float32, f"Sai dtype X_val: {X_val.dtype}"
    assert y_val.dtype == torch.int64, f"Sai dtype y_val: {y_val.dtype}"
    assert X_eval.dtype == torch.float32, f"Sai dtype X_eval: {X_eval.dtype}"
    assert y_eval.dtype == torch.int64, f"Sai dtype y_eval: {y_eval.dtype}"

    print("  -> Train Sub : 371,847 mẫu (80.00% của train)")
    print("  -> Val Set   :  92,962 mẫu (20.00% của train)")
    print("  -> Eval Set  : 116,203 mẫu (100% giữ kín cho final evaluation)")
    print("  [OK] Số lượng mẫu và kích thước tensors chuẩn xác 100%!")


def test_stratification(y_full, y_tr, y_val):
    print("\n[2/5] Kiểm tra tính phân tầng theo nhãn (Stratified Split)...")
    full_counts = np.bincount(y_full, minlength=7)
    tr_counts = np.bincount(y_tr.cpu().numpy(), minlength=7)
    val_counts = np.bincount(y_val.cpu().numpy(), minlength=7)

    full_dist = full_counts / len(y_full)
    tr_dist = tr_counts / len(y_tr)
    val_dist = val_counts / len(y_val)

    print(f"  {'Lớp':5s} | {'Train Full':12s} | {'Train Sub':12s} | {'Val':12s} | {'Độ lệch lớn nhất':18s}")
    print("  " + "-" * 68)
    for c in range(7):
        max_diff = max(abs(tr_dist[c] - full_dist[c]), abs(val_dist[c] - full_dist[c]))
        print(f"  Lớp {c} | {full_dist[c]*100:10.4f}% | {tr_dist[c]*100:10.4f}% | {val_dist[c]*100:10.4f}% | {max_diff*100:16.6f}%")
        assert max_diff < 0.0005, f"Lớp {c} bị lệch phân tầng quá 0.05%!"
    print("  [OK] Tính phân tầng giữa các tập đạt chuẩn xác tuyệt đối!")


def test_standardization_and_binary_preservation(data):
    print("\n[3/5] Kiểm tra chuẩn hóa 10 cột liên tục và bảo toàn 44 cột nhị phân...")
    X_tr = data["X_tr"].cpu().numpy()
    X_val = data["X_val"].cpu().numpy()
    X_eval = data["X_eval"].cpu().numpy()

    # 1. Kiểm tra 10 cột liên tục đầu tiên
    tr_num_mean = X_tr[:, :10].mean(axis=0)
    tr_num_std = X_tr[:, :10].std(axis=0)

    print(f"  -> Mean 10 cột số train_sub: min={tr_num_mean.min():.5f}, max={tr_num_mean.max():.5f}")
    print(f"  -> Std  10 cột số train_sub: min={tr_num_std.min():.5f}, max={tr_num_std.max():.5f}")
    assert np.allclose(tr_num_mean, 0.0, atol=1e-2), "Mean 10 cột trên train_sub chưa đạt chuẩn 0!"
    assert np.allclose(tr_num_std, 1.0, atol=1e-2), "Std 10 cột trên train_sub chưa đạt chuẩn 1!"

    # 2. Kiểm tra 44 cột nhị phân sau (cột 10..53)
    tr_bin = X_tr[:, 10:]
    val_bin = X_val[:, 10:]
    eval_bin = X_eval[:, 10:]

    assert np.all(np.isin(tr_bin, [0.0, 1.0])), "Cột nhị phân train_sub bị biến đổi sai!"
    assert np.all(np.isin(val_bin, [0.0, 1.0])), "Cột nhị phân val bị biến đổi sai!"
    assert np.all(np.isin(eval_bin, [0.0, 1.0])), "Cột nhị phân eval bị biến đổi sai!"

    print(f"  -> 44 cột nhị phân: Hoàn toàn giữ nguyên giá trị nhị phân thuần túy {{0.0, 1.0}} trên cả 3 tập!")
    print("  [OK] Chuẩn hóa và bảo toàn đặc trưng nhị phân đạt chuẩn 100%!")


def test_no_data_leakage(data):
    print("\n[4/5] Kiểm tra bằng chứng toán học chống rò rỉ dữ liệu (No Data Leakage)...")
    # Tải lại dữ liệu thô chưa chuẩn hóa
    X_train_full, y_train_full, X_eval, y_eval, _ = load_split("data/processed")
    X_tr_raw, _, X_val_raw, _ = make_val_split(X_train_full, y_train_full, val_fraction=0.2, seed=42)

    # 1. Thống kê fit trên train_sub
    mean_tr, std_tr = fit_standardizer(X_tr_raw)

    # 2. Thống kê fit sai lầm nếu tính trên train_full
    mean_full, std_full = fit_standardizer(X_train_full)

    # 3. Thống kê fit sai lầm nếu tính trên val
    mean_val, std_val = fit_standardizer(X_val_raw)

    diff_mean_vs_full = np.abs(mean_tr - mean_full).max()
    diff_mean_vs_val = np.abs(mean_tr - mean_val).max()

    print(f"  -> Sai khác lớn nhất giữa mean(train_sub) và mean(train_full): {diff_mean_vs_full:.6f}")
    print(f"  -> Sai khác lớn nhất giữa mean(train_sub) và mean(val)       : {diff_mean_vs_val:.6f}")
    assert diff_mean_vs_full > 0.0, "Mean train_sub không được trùng khít tuyệt đối với train_full!"
    assert diff_mean_vs_val > 0.0, "Mean train_sub không được trùng khít tuyệt đối với val!"

    # Xác nhận data['mean'] và data['std'] chính là mean_tr và std_tr
    assert np.allclose(data["mean"], mean_tr), "Pipeline không sử dụng mean của train_sub!"
    assert np.allclose(data["std"], std_tr), "Pipeline không sử dụng std của train_sub!"
    print("  [OK] Bằng chứng toán học khẳng định KHÔNG CÓ RÒ RỈ DỮ LIỆU (Zero Data Leakage)!")


def test_firewall_rule(data):
    print("\n[5/5] Kiểm tra Quy tắc Tường lửa trong quá trình huấn luyện và chọn mô hình...")
    # Thử chạy một cấu hình huấn luyện nhỏ
    mini_data = {
        "X_tr": data["X_tr"][:1024],
        "y_tr": data["y_tr"][:1024],
        "X_val": data["X_val"][:512],
        "y_val": data["y_val"][:512],
        # Đặt bẫy vào eval để phát hiện nếu train.py vô tình chạm vào eval
        "X_eval": None,
        "y_eval": None,
        "eval_row_id": None,
    }

    test_cfg = {
        "exp_id": "test_firewall",
        "group": "test",
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.05,
        "batch": 256,
        "epochs": 1,
        "hidden": (128, 64),
        "seed": 42,
    }

    try:
        # run_experiment KHÔNG được chạm vào X_eval hay y_eval
        res = run_experiment(test_cfg, mini_data, verbose=False)
        print("  -> Huấn luyện và validation hoàn tất mà KHÔNG chạm vào tập eval!")
        assert "val_macro_f1" in res["summary"]
    except TypeError as e:
        assert False, f"Vi phạm tường lửa: run_experiment đã cố truy cập vào eval_data: {e}"

    print("  [OK] Quy tắc tường lửa bảo vệ tập Eval được thực thi triệt để 100%!")


def main():
    print("=" * 80)
    print("BẮT ĐẦU REQ-12: KIỂM CHỨNG TÍNH TOÀN VẸN DỮ LIỆU & QUY TẮC TƯỜNG LỬA (FIREWALL RULE)")
    print("=" * 80)

    data = prepare_data(device="cpu", processed_dir="data/processed")
    X_train_full, y_train_full, X_eval, y_eval, _ = load_split("data/processed")

    test_sample_counts_and_shapes(data, X_train_full, X_eval)
    test_stratification(y_train_full, data["y_tr"], data["y_val"])
    test_standardization_and_binary_preservation(data)
    test_no_data_leakage(data)
    test_firewall_rule(data)

    print("\n" + "=" * 80)
    print("XÁC NHẬN TOÀN DIỆN REQ-12 ĐÃ ĐẠT CHUẨN XUẤT SẮC 100%!")
    print("=" * 80)


if __name__ == "__main__":
    main()
