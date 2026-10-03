"""scripts/run_req06_baseline.py
Thực thi REQ-06: Huấn luyện Baseline M-base trên 3 seeds (1, 2, 3).
Tính toán:
  - Mean, std, và ngưỡng nhiễu 2*sigma của Val Macro-F1.
  - Lưu plots (figures/base-s1.png, base-s2.png, base-s3.png, so sánh seeds).
  - Lưu JSON (results/base-s*.json).
  - Dự đoán eval cho baseline (predictions_eval.csv).
  - Cập nhật experiments.xlsx từ templates/experiment_table_template.xlsx.
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
from results_table import save_result, to_row, write_xlsx

def main():
    print("=" * 80)
    print("BẮT ĐẦU REQ-06: HUẤN LUYỆN BASELINE VÀ ĐO NGƯỠNG NHIỄU SEED (SEEDS 1, 2, 3)")
    print("=" * 80)

    # 1. Nạp dữ liệu
    print("\n[Bước 1/6] Nạp dữ liệu chuẩn bị...")
    data = prepare_data(device="cpu", processed_dir="data/processed")

    # 2. Định nghĩa cấu hình Baseline
    base_cfg = {
        "loss": "ce",
        "optimizer": "sgd_momentum",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "batch": 512,
        "epochs": 20,
        "hidden": (256, 128),
        "dropout": 0.0,
        "init": "he",
        "clip_norm": None,
        "precision": "fp32",
    }

    seeds = [1, 2, 3]
    results = []

    # 3. Huấn luyện 3 seeds
    print("\n[Bước 2/6] Huấn luyện Baseline trên 3 seeds...")
    for s in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["exp_id"] = f"base-s{s}"
        cfg["group"] = "baseline"
        cfg["description"] = f"Baseline M-base (Seed {s})"
        cfg["seed"] = s

        print(f"\n>>> Chạy Baseline Seed {s} ({cfg['exp_id']})...")
        res = run_experiment(cfg, data, verbose=True)
        results.append(res)

    # 4. Dự đoán trên tập eval cho Baseline (base-s1)
    print("\n[Bước 3/6] Đánh giá Baseline trên tập Eval (116,203 mẫu)...")
    res_s1 = results[0]
    pred_path_sub = "submission_2A202602372/predictions_eval.csv"
    pred_path_root = "predictions_eval.csv"

    eval_scores = final_eval(res_s1["cfg"], res_s1, data, pred_path=pred_path_sub)
    shutil.copy2(pred_path_sub, pred_path_root)

    # 5. Lưu biểu đồ và kết quả JSON
    print("\n[Bước 4/6] Lưu biểu đồ và file JSON kết quả...")
    for res in results:
        exp_id = res["cfg"]["exp_id"]
        # Lưu JSON
        save_result(res, results_dir="results")
        save_result(res, results_dir="submission_2A202602372/results")

        # Lưu ảnh từng thí nghiệm
        fig_root = f"figures/{exp_id}.png"
        fig_sub = f"submission_2A202602372/figures/{exp_id}.png"
        plot_run(res, fig_root)
        plot_run(res, fig_sub)
        print(f"  - Đã lưu biểu đồ: {fig_root} & {fig_sub}")

    # Vẽ biểu đồ so sánh 3 seeds
    plot_compare(
        results,
        metric="val_macro_f1",
        path="figures/baseline_seeds_val_macro_f1.png",
        title="So sánh Val Macro-F1 giữa 3 Seeds của Baseline M-base",
    )
    plot_compare(
        results,
        metric="val_macro_f1",
        path="submission_2A202602372/figures/baseline_seeds_val_macro_f1.png",
        title="So sánh Val Macro-F1 giữa 3 Seeds của Baseline M-base",
    )
    plot_compare(
        results,
        metric="val_loss",
        path="figures/baseline_seeds_val_loss.png",
        title="So sánh Val Loss giữa 3 Seeds của Baseline M-base",
    )
    plot_compare(
        results,
        metric="val_loss",
        path="submission_2A202602372/figures/baseline_seeds_val_loss.png",
        title="So sánh Val Loss giữa 3 Seeds của Baseline M-base",
    )
    print("  - Đã lưu biểu đồ so sánh các seeds.")

    # 6. Tính toán thống kê và cập nhật experiments.xlsx
    print("\n[Bước 5/6] Tính toán thống kê độ nhiễu giữa các seeds...")
    val_f1s = [r["summary"]["val_macro_f1"] for r in results]
    val_accs = [r["summary"]["val_acc"] for r in results]
    best_losses = [r["summary"]["best_val_loss"] for r in results]

    mean_f1 = float(np.mean(val_f1s))
    std_f1 = float(np.std(val_f1s, ddof=1))  # Sample std (ddof=1)
    noise_f1 = 2.0 * std_f1

    mean_acc = float(np.mean(val_accs))
    std_acc = float(np.std(val_accs, ddof=1))

    mean_loss = float(np.mean(best_losses))
    std_loss = float(np.std(best_losses, ddof=1))

    print("-" * 60)
    print(f"BẢNG TỔNG HỢP BASELINE (3 SEEDS):")
    for s_idx, r in enumerate(results, start=1):
        s_res = r["summary"]
        print(f"  Seed {s_idx} ({r['cfg']['exp_id']}): Val Loss = {s_res['best_val_loss']:.4f} (ep {s_res['best_epoch']}) | Val Acc = {s_res['val_acc']:.4f} | Val Macro-F1 = {s_res['val_macro_f1']:.4f}")
    print("-" * 60)
    print(f"Val Macro-F1 : Trung bình = {mean_f1:.4f} | Độ lệch chuẩn (σ) = {std_f1:.4f} | Ngưỡng nhiễu (2σ) = {noise_f1:.4f}")
    print(f"Val Accuracy : Trung bình = {mean_acc:.4f} | Độ lệch chuẩn (σ) = {std_acc:.4f}")
    print(f"Best Val Loss: Trung bình = {mean_loss:.4f} | Độ lệch chuẩn (σ) = {std_loss:.4f}")
    print("-" * 60)
    print(f"==> KẾT LUẬN NGƯỠNG NHIỄU: Bất kỳ cải tiến nào có |Δ Val F1| <= {noise_f1:.4f} (2σ)")
    print(f"    đều KHÔNG được coi là có ý nghĩa thống kê vượt trội (chỉ là nhiễu ngẫu nhiên).")
    print("-" * 60)

    # 7. Ghi vào experiments.xlsx
    print("\n[Bước 6/6] Ghi dữ liệu vào experiments.xlsx...")
    rows = []
    # Seed 1 (có eval_scores)
    row_s1 = to_row(results[0], eval_scores=eval_scores, notes="Baseline chính thức (Seed 1)")
    rows.append(row_s1)
    # Seed 2 & 3
    row_s2 = to_row(results[1], notes="Baseline seed run (Seed 2)")
    rows.append(row_s2)
    row_s3 = to_row(results[2], notes="Baseline seed run (Seed 3)")
    rows.append(row_s3)

    template_path = "templates/experiment_table_template.xlsx"
    out_xlsx_root = "experiments.xlsx"
    out_xlsx_sub = "submission_2A202602372/experiments.xlsx"

    write_xlsx(rows, template_path=template_path, out_path=out_xlsx_root)
    write_xlsx(rows, template_path=template_path, out_path=out_xlsx_sub)

    # Lưu lại stats vào JSON để tiện tra cứu
    stats = {
        "seeds": seeds,
        "val_macro_f1": {"individual": val_f1s, "mean": mean_f1, "std": std_f1, "noise_2sigma": noise_f1},
        "val_acc": {"individual": val_accs, "mean": mean_acc, "std": std_acc},
        "best_val_loss": {"individual": best_losses, "mean": mean_loss, "std": std_loss},
        "eval_scores_base_s1": {
            "acc": float(eval_scores["acc"]),
            "macro_f1": float(eval_scores["macro_f1"]),
        },
    }
    with open("results/baseline_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open("submission_2A202602372/results/baseline_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== HOÀN THÀNH TOÀN DIỆN REQ-06! ===")

if __name__ == "__main__":
    main()
