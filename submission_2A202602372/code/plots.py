"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    """
    history = result.get("history", {})
    cfg = result.get("cfg", {})
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    exp_id = cfg.get("exp_id", "exp")

    # Tạo thư mục cha nếu chưa có
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    fig.suptitle(
        f"[{exp_id}] {cfg.get('description', '')} | opt={cfg.get('optimizer')} | lr={cfg.get('lr')} | bs={cfg.get('batch')}",
        fontsize=12,
        fontweight="bold",
    )

    # 1. Train loss & Val loss
    ax1 = axes[0]
    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="#d62728", lw=2, linestyle="--")
    best_ep = result.get("best_epoch")
    if best_ep:
        ax1.axvline(best_ep, color="green", linestyle=":", alpha=0.7, label=f"Best Ep ({best_ep})")
    ax1.set_title("Hàm mất mát (Loss)", fontsize=11)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Val Accuracy & Val Macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, history.get("val_acc", []), label="Val Acc", color="#2ca02c", lw=2)
    if "val_macro_f1" in history:
        ax2.plot(epochs, history.get("val_macro_f1", []), label="Val Macro-F1", color="#ff7f0e", lw=2)
    # Đường chuẩn đoán đa số
    ax2.axhline(0.4876, color="gray", linestyle=":", alpha=0.6, label="Majority Class (0.488)")
    ax2.set_title("Độ chính xác & Macro-F1 trên Val", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Gradient Norm trước khi clip
    ax3 = axes[2]
    ax3.plot(epochs, history.get("grad_norm", []), label="Grad Norm (L2)", color="#9467bd", lw=2)
    clip_c = cfg.get("clip_norm")
    if clip_c:
        ax3.axhline(clip_c, color="red", linestyle=":", alpha=0.7, label=f"Clip threshold c={clip_c}")
    ax3.set_title("Gradient Norm (trước Clip)", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục để so sánh nhóm."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for res in results:
        cfg = res.get("cfg", {})
        exp_id = cfg.get("exp_id", "exp")
        hist = res.get("history", {})
        vals = hist.get(metric, [])
        epochs = range(1, len(vals) + 1)
        ax.plot(epochs, vals, lw=1.8, label=f"{exp_id} ({cfg.get('optimizer')}, lr={cfg.get('lr')})")

    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_overfit_20(losses: list[float], path: str, title: str = "Kiểm tra Sức khỏe: Quá khớp 20 mẫu") -> None:
    """Vẽ đường cong loss khi huấn luyện quá khớp 20 mẫu."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    steps = range(1, len(losses) + 1)
    ax.plot(steps, losses, color="#1f77b4", lw=2, label="Cross-Entropy Loss")
    ax.axhline(0.01, color="red", linestyle="--", alpha=0.6, label="Ngưỡng mục tiêu (< 0.01)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Số bước cập nhật (Steps)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

