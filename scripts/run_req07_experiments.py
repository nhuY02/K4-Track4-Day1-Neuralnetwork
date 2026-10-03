"""scripts/run_req07_experiments.py
Triển khai toàn bộ 7 Chủ đề Thí nghiệm Khoa học (Part 3 trong GUIDE.md).
Bao gồm:
  1. Loss: CE vs MSE
  2. Optimizer: SGD, SGD+Momentum, Adam (nhiều lr), AdamW
  3. Hyperparameters: Batch sizes (128, 512, 2048), Kiến trúc (M-base, M-wide, M-deep)
  4. Regularization / Dropout: q in {0.0, 0.1, 0.2, 0.3}
  5. Gradient Clipping: c=0.5, stress test lr=1.5 (no-clip vs clip=1.0)
  6. Mixed Precision: FP32 vs BF16 (đo time, memory, giải thích FP16/BF16)
  7. Weight Initialization: He, Xavier, Normal, Zeros
  8. Final Candidates: Kết hợp các yếu tố tối ưu để chọn Final Model tốt nhất.

Tất cả kết quả:
  - Lưu JSON vào results/ và submission_2A202602372/results/
  - Lưu ảnh từng thí nghiệm vào figures/ và submission_2A202602372/figures/
  - Lưu ảnh so sánh từng nhóm (compare_*.png)
  - Cập nhật tự động vào experiments.xlsx
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np

# Thêm code vào sys.path
sys.path.insert(0, str(Path("submission_2A202602372/code").resolve()))

from data import prepare_data
from train import run_experiment, final_eval
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx

def main():
    print("=" * 80)
    print("BẮT ĐẦU REQ-07: TRIỂN KHAI 7 CHỦ ĐỀ THÍ NGHIỆM KHOA HỌC")
    print("=" * 80)

    # 1. Nạp dữ liệu
    print("\n[1/3] Nạp dữ liệu chuẩn bị...")
    data = prepare_data(device="cpu", processed_dir="data/processed")

    # Đọc kết quả baseline đã chạy ở REQ-06
    baseline_s1_file = Path("submission_2A202602372/results/base-s1.json")
    assert baseline_s1_file.exists(), "Cần có base-s1.json từ REQ-06"
    with open(baseline_s1_file, "r", encoding="utf-8") as f:
        res_base_s1 = json.load(f)

    # Đọc eval_result của baseline
    eval_result_file = Path("submission_2A202602372/eval_result.json")
    eval_scores_base = None
    if eval_result_file.exists():
        with open(eval_result_file, "r", encoding="utf-8") as f:
            ev_data = json.load(f)
            eval_scores_base = {"acc": ev_data["accuracy"], "macro_f1": ev_data["macro_f1"]}

    # Cấu hình chuẩn của Baseline để kế thừa
    BASE = copy.deepcopy(res_base_s1["cfg"])

    # Danh sách cấu hình thí nghiệm
    experiments = [
        # --- CHỦ ĐỀ 1: HÀM MẤT MÁT (LOSS) ---
        {
            "cfg": {**BASE, "exp_id": "loss-mse", "group": "loss", "loss": "mse",
                    "description": "MSE Loss trên one-hot targets"},
            "notes": "MSE Loss (phạt bậc hai); gradient nhỏ hơn CE khi dự đoán sai lệch lớn.",
        },

        # --- CHỦ ĐỀ 2: BỘ TỐI ƯU HOÁ (OPTIMIZER) ---
        {
            "cfg": {**BASE, "exp_id": "opt-sgd-lr0.1", "group": "optimizer", "optimizer": "sgd", "lr": 0.1, "momentum": 0.0,
                    "description": "Thuần SGD không momentum (lr=0.1)"},
            "notes": "SGD thuần không có quán tính (momentum=0.0); dao động zic-zac trong rãnh hẹp.",
        },
        {
            "cfg": {**BASE, "exp_id": "opt-adam-lr1e-3", "group": "optimizer", "optimizer": "adam", "lr": 0.001,
                    "description": "Adam với lr=0.001 (chuẩn phổ biến)"},
            "notes": "Adam với lr=1e-3; thích ứng bước học theo căn bậc hai mô-men cấp 2.",
        },
        {
            "cfg": {**BASE, "exp_id": "opt-adam-lr3e-3", "group": "optimizer", "optimizer": "adam", "lr": 0.003,
                    "description": "Adam với lr=0.003 (tốc độ học lớn hơn)"},
            "notes": "Adam với lr=3e-3; tốc độ học cao hơn giúp hội tụ sớm hơn.",
        },
        {
            "cfg": {**BASE, "exp_id": "opt-adamw-lr1e-3", "group": "optimizer", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01,
                    "description": "AdamW với lr=0.001, weight_decay=0.01"},
            "notes": "AdamW tách riêng suy giảm trọng số (decoupled weight decay 0.01).",
        },

        # --- CHỦ ĐỀ 3: HYPERPARAMETERS (BATCH SIZE & ARCHITECTURE) ---
        {
            "cfg": {**BASE, "exp_id": "batch-128", "group": "hparam", "batch": 128,
                    "description": "Batch size nhỏ (128) - 2905 bước/epoch"},
            "notes": "Batch 128: 2.905 bước/epoch (gấp 4 lần baseline); nhiễu gradient cao hỗ trợ thoát cực tiểu.",
        },
        {
            "cfg": {**BASE, "exp_id": "batch-2048", "group": "hparam", "batch": 2048,
                    "description": "Batch size lớn (2048) - 181 bước/epoch"},
            "notes": "Batch 2048: chỉ 181 bước/epoch; gradient mượt hơn nhưng tiến ít bước hơn trong 20 epochs.",
        },
        {
            "cfg": {**BASE, "exp_id": "arch-mwide", "group": "hparam", "hidden": (512, 256),
                    "description": "Kiến trúc M-wide (512->256, 161.287 tham số)"},
            "notes": "M-wide (161k params); tăng gấp 3.3 lần dung lượng biểu diễn so với M-base.",
        },
        {
            "cfg": {**BASE, "exp_id": "arch-mdeep", "group": "hparam", "hidden": (256, 128, 64),
                    "description": "Kiến trúc M-deep (256->128->64, 55.687 tham số)"},
            "notes": "M-deep: 3 lớp ẩn; tăng chiều sâu biểu diễn đặc trưng phân cấp.",
        },

        # --- CHỦ ĐỀ 4: DROPOUT REGULARIZATION ---
        {
            "cfg": {**BASE, "exp_id": "drop-0.1", "group": "dropout", "dropout": 0.1,
                    "description": "Dropout q=0.1 sau ReLU các lớp ẩn"},
            "notes": "Dropout q=0.1; ngẫu nhiên vô hiệu hoá 10% nơ-ron ẩn sau ReLU.",
        },
        {
            "cfg": {**BASE, "exp_id": "drop-0.2", "group": "dropout", "dropout": 0.2,
                    "description": "Dropout q=0.2 sau ReLU các lớp ẩn"},
            "notes": "Dropout q=0.2; kiểm tra khả năng chống quá khớp khi tăng độ ngẫu nhiên.",
        },
        {
            "cfg": {**BASE, "exp_id": "drop-0.3", "group": "dropout", "dropout": 0.3,
                    "description": "Dropout q=0.3 sau ReLU các lớp ẩn"},
            "notes": "Dropout q=0.3; mức dropout cao đối với dữ liệu 371k mẫu có thể gây underfitting nhẹ.",
        },

        # --- CHỦ ĐỀ 5: GRADIENT CLIPPING ---
        {
            "cfg": {**BASE, "exp_id": "clip-0.5", "group": "clipping", "clip_norm": 0.5,
                    "description": "Gradient clipping c=0.5 (nhỏ hơn mức trung bình 0.56)"},
            "notes": "Clipping c=0.5 chủ động cắt tỉa gradient khi vượt ngưỡng 0.5; kiểm chứng độ ổn định.",
        },
        {
            "cfg": {**BASE, "exp_id": "clip-stress-noclip", "group": "clipping", "lr": 1.5, "clip_norm": None,
                    "description": "Phép thử stress: lr cực cao (1.5) không dùng clip"},
            "notes": "Phép thử phản chứng: lr=1.5 không clip; quan sát bùng nổ gradient và mất ổn định.",
        },
        {
            "cfg": {**BASE, "exp_id": "clip-stress-clip1.0", "group": "clipping", "lr": 1.5, "clip_norm": 1.0,
                    "description": "Phép thử stress: lr cực cao (1.5) có clip c=1.0"},
            "notes": "Phép thử phản chứng: lr=1.5 có clip c=1.0; chứng minh gradient clipping cứu mô hình khỏi phân kỳ.",
        },

        # --- CHỦ ĐỀ 6: MIXED PRECISION ---
        {
            "cfg": {**BASE, "exp_id": "amp-bf16", "group": "amp", "precision": "bf16",
                    "description": "Mixed Precision BFloat16 (autocast)"},
            "notes": "BFloat16 autocast; giữ dải động 8-bit số mũ như FP32, không cần GradScaler.",
        },

        # --- CHỦ ĐỀ 7: WEIGHT INITIALIZATION ---
        {
            "cfg": {**BASE, "exp_id": "init-xavier", "group": "init", "init": "xavier",
                    "description": "Khởi tạo Xavier Normal (Glorot)"},
            "notes": "Xavier Normal (Var=2/(nin+nout)); tối ưu cho kích hoạt tuyến tính/tanh, hơi hụt phương sai với ReLU.",
        },
        {
            "cfg": {**BASE, "exp_id": "init-normal", "group": "init", "init": "normal",
                    "description": "Khởi tạo Normal N(0, 0.01^2) phương sai nhỏ"},
            "notes": "Normal std=0.01; phương sai nhỏ dẫn đến tín hiệu kích hoạt teo tóp sau ReLU (vanishing).",
        },
        {
            "cfg": {**BASE, "exp_id": "init-zeros", "group": "init", "init": "zeros",
                    "description": "Khởi tạo tất cả trọng số bằng 0 (zeros)"},
            "notes": "Zeros: mọi nơ-ron nhận tín hiệu 0, ReLU(0)=0; mất tính phá vỡ đối xứng, mạng không thể học.",
        },

        # --- CHỦ ĐỀ 8: CẤU HÌNH TỐI ƯU CUỐI CÙNG (FINAL MODEL CANDIDATES) ---
        {
            "cfg": {**BASE, "exp_id": "cand-mwide-adam", "group": "final",
                    "hidden": (512, 256), "optimizer": "adam", "lr": 0.002, "batch": 512,
                    "description": "Ứng viên Tối ưu: M-wide + Adam (lr=0.002)"},
            "notes": "Kết hợp kiến trúc M-wide (161k params) và bộ tối ưu Adam lr=0.002 để tối đa hoá Macro-F1.",
        },
        {
            "cfg": {**BASE, "exp_id": "cand-mbase-adam", "group": "final",
                    "hidden": (256, 128), "optimizer": "adam", "lr": 0.002, "batch": 256,
                    "description": "Ứng viên Tối ưu: M-base + Adam (lr=0.002) + Batch 256"},
            "notes": "M-base chuẩn + Adam lr=0.002 + Batch 256; tăng số bước cập nhật để tối ưu hóa hội tụ.",
        },
    ]

    print(f"[2/3] Tổng số thí nghiệm cần chạy: {len(experiments)} thí nghiệm.")
    all_results = [res_base_s1]
    
    # Nạp seed 2 & seed 3 nếu có
    for s_id in ["base-s2", "base-s3"]:
        s_file = Path(f"submission_2A202602372/results/{s_id}.json")
        if s_file.exists():
            with open(s_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))

    # Chạy từng thí nghiệm
    for idx, item in enumerate(experiments, start=1):
        cfg = item["cfg"]
        notes = item["notes"]
        exp_id = cfg["exp_id"]

        print(f"\n[{idx}/{len(experiments)}] Đang chạy: {exp_id} ({cfg['description']})...")
        res = run_experiment(cfg, data, verbose=True)
        res["notes"] = notes
        all_results.append(res)

        # Lưu JSON
        save_result(res, results_dir="results")
        save_result(res, results_dir="submission_2A202602372/results")

        # Vẽ ảnh từng thí nghiệm
        fig_root = f"figures/{exp_id}.png"
        fig_sub = f"submission_2A202602372/figures/{exp_id}.png"
        plot_run(res, fig_root)
        plot_run(res, fig_sub)
        print(f"  -> Đã lưu biểu đồ: {fig_root} & {fig_sub}")

    # Vẽ các biểu đồ so sánh theo nhóm
    print("\n[3/3] Tạo các biểu đồ so sánh theo nhóm (compare_*.png)...")
    
    groups_to_compare = {
        "loss": ("loss", "val_macro_f1", "So sánh Val Macro-F1: Cross-Entropy vs MSE Loss"),
        "optimizer": ("optimizer", "val_macro_f1", "So sánh Val Macro-F1 giữa các Bộ tối ưu (SGD, Adam, AdamW)"),
        "hparam_batch": ("hparam", "val_macro_f1", "So sánh Val Macro-F1 theo Kích thước Batch (128, 512, 2048)"),
        "hparam_arch": ("hparam", "val_macro_f1", "So sánh Val Macro-F1 theo Dung lượng Kiến trúc (M-base, M-wide, M-deep)"),
        "dropout": ("dropout", "val_macro_f1", "So sánh Val Macro-F1 theo Tỉ lệ Dropout (0.0, 0.1, 0.2, 0.3)"),
        "clipping": ("clipping", "val_loss", "So sánh Val Loss: Ổn định Gradient và Phép thử Stress (Clip vs No-Clip)"),
        "init": ("init", "val_macro_f1", "So sánh Val Macro-F1 giữa các Phương pháp Khởi tạo Trọng số"),
        "final": ("final", "val_macro_f1", "So sánh Val Macro-F1 giữa Baseline và các Ứng viên Tối ưu Cuối cùng"),
    }

    # Bổ sung baseline vào từng nhóm để so sánh trực quan
    for grp_key, (grp_name, metric, title) in groups_to_compare.items():
        if grp_key == "hparam_batch":
            grp_res = [res_base_s1] + [r for r in all_results if r["cfg"]["exp_id"] in ["batch-128", "batch-2048"]]
        elif grp_key == "hparam_arch":
            grp_res = [res_base_s1] + [r for r in all_results if r["cfg"]["exp_id"] in ["arch-mwide", "arch-mdeep"]]
        elif grp_key == "clipping":
            grp_res = [res_base_s1] + [r for r in all_results if r["cfg"]["group"] == "clipping"]
        else:
            grp_res = [res_base_s1] + [r for r in all_results if r["cfg"]["group"] == grp_name]

        out_fig_root = f"figures/compare_{grp_key}.png"
        out_fig_sub = f"submission_2A202602372/figures/compare_{grp_key}.png"
        plot_compare(grp_res, metric=metric, path=out_fig_root, title=title)
        plot_compare(grp_res, metric=metric, path=out_fig_sub, title=title)
        print(f"  - Đã lưu biểu đồ so sánh nhóm: {out_fig_root}")

    # Ghi toàn bộ các dòng vào bảng Excel experiments.xlsx
    print("\nCập nhật toàn bộ các thí nghiệm vào experiments.xlsx...")
    rows = []
    
    # 1. Baseline s1, s2, s3
    rows.append(to_row(res_base_s1, eval_scores=eval_scores_base, notes="Baseline chính thức (Seed 1)"))
    for r in all_results:
        if r["cfg"]["exp_id"] in ["base-s2", "base-s3"]:
            rows.append(to_row(r, notes=f"Baseline seed run ({r['cfg']['exp_id']})"))

    # 2. Toàn bộ các thí nghiệm còn lại
    for r in all_results:
        exp_id = r["cfg"]["exp_id"]
        if exp_id.startswith("base-s"):
            continue
        notes = r.get("notes", "")
        rows.append(to_row(r, notes=notes))

    template_path = "templates/experiment_table_template.xlsx"
    out_xlsx_root = "experiments.xlsx"
    out_xlsx_sub = "submission_2A202602372/experiments.xlsx"

    write_xlsx(rows, template_path=template_path, out_path=out_xlsx_root)
    write_xlsx(rows, template_path=template_path, out_path=out_xlsx_sub)

    print("\n" + "=" * 80)
    print("HOÀN THÀNH TẤT CẢ CÁC THÍ NGHIỆM CHO REQ-07!")
    print("=" * 80)

if __name__ == "__main__":
    main()
