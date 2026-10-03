"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl


def save_result(result: dict, results_dir: str = "results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]

    save_dict = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"],
    }

    def convert(o):
        if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        if isinstance(o, (np.float32, np.float64, np.float16)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    file_path = out_dir / f"{exp_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, default=convert)
    return str(file_path)


def load_results(results_dir: str = "results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    p = Path(results_dir)
    if not p.exists():
        return []
    results = []
    for f in sorted(p.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg = result["cfg"]
    summary = result["summary"]
    row: dict = {}

    # 1. Cấu hình
    row["exp_id"] = cfg.get("exp_id", "")
    row["group"] = cfg.get("group", "")
    row["description"] = cfg.get("description", "")
    row["loss"] = str(cfg.get("loss", "ce")).upper()
    row["optimizer"] = cfg.get("optimizer", "")
    row["lr"] = cfg.get("lr")
    row["weight_decay"] = cfg.get("weight_decay", 0.0)
    row["batch"] = cfg.get("batch", 512)
    row["epochs"] = cfg.get("epochs", 20)

    h = cfg.get("hidden", (256, 128))
    row["hidden"] = str(h) if isinstance(h, (tuple, list)) else str(h)

    row["dropout"] = cfg.get("dropout", 0.0)
    row["clip_norm"] = cfg.get("clip_norm", "") if cfg.get("clip_norm") is not None else ""
    row["precision"] = cfg.get("precision", "fp32")
    row["init"] = cfg.get("init", "he")
    row["seed"] = cfg.get("seed", 1)

    # 2. Tóm tắt kết quả
    row["step0_loss"] = summary.get("step0_loss")
    row["best_val_loss"] = summary.get("best_val_loss")
    row["best_epoch"] = summary.get("best_epoch")
    row["final_train_loss"] = summary.get("final_train_loss")
    row["final_val_loss"] = summary.get("final_val_loss")
    row["val_acc"] = summary.get("val_acc")
    row["val_macro_f1"] = summary.get("val_macro_f1")
    row["time_per_epoch_s"] = summary.get("time_per_epoch_s")
    row["peak_mem_MB"] = summary.get("peak_mem_MB")
    row["diverged"] = "Có" if summary.get("diverged") else "Không"

    # 3. Đánh giá eval (chỉ có ở baseline và cấu hình cuối)
    if eval_scores is not None:
        row["eval_acc"] = eval_scores.get("acc", "")
        row["eval_macro_f1"] = eval_scores.get("macro_f1", "")
    else:
        row["eval_acc"] = ""
        row["eval_macro_f1"] = ""

    row["figure_file"] = f"figures/{row['exp_id']}.png"
    row["notes"] = notes

    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]

    # Đọc tiêu đề dòng 1 để ánh xạ tên cột -> số cột
    col_mapping = {}
    for col_idx in range(1, ws.max_column + 1):
        val = ws.cell(row=1, column=col_idx).value
        if val:
            col_mapping[val] = col_idx

    formula_cols = {
        "step0_gap_vs_lnC",
        "gap_val_minus_train",
        "delta_val_f1_vs_base",
        "beyond_noise",
    }

    # Ghi dữ liệu bắt đầu từ dòng 2
    for r_idx, row_data in enumerate(rows, start=2):
        for key, val in row_data.items():
            if key in col_mapping and key not in formula_cols:
                ws.cell(row=r_idx, column=col_mapping[key], value=val)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"Đã lưu bảng kết quả thực nghiệm ({len(rows)} dòng) -> {out_path}")

