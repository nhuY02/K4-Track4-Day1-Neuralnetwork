"""error_analysis.py — Phân tích lỗi và trực quan hoá ma trận nhầm lẫn (Confusion Matrix).

Hỗ trợ REQ-09:
- Vẽ heatmap ma trận nhầm lẫn nguyên bản và chuẩn hoá (%).
- So sánh F1-score từng lớp giữa baseline và mô hình cuối.
- Phân tích tương quan sinh thái và địa hình rừng cho các cặp lớp thường nhầm lẫn.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "0: Spruce/Fir",
    "1: Lodgepole Pine",
    "2: Ponderosa Pine",
    "3: Cottonwood/Willow",
    "4: Aspen",
    "5: Douglas-fir",
    "6: Krummholz",
]

CLASS_SHORT_NAMES = [
    "Spruce/Fir",
    "Lodgepole",
    "Ponderosa",
    "Cottonwood",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]


def plot_confusion_matrix(cm: np.ndarray, path: str, title: str = "Confusion Matrix", normalize: bool = False):
    """Vẽ heatmap ma trận nhầm lẫn chuyên nghiệp."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=180)
    cmap = plt.cm.Blues

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize:
        cbar.ax.set_ylabel("Tỉ lệ phân loại (%)", rotation=-90, va="bottom", fontsize=10)
    else:
        cbar.ax.set_ylabel("Số lượng mẫu (samples)", rotation=-90, va="bottom", fontsize=10)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_SHORT_NAMES,
        yticklabels=CLASS_SHORT_NAMES,
        title=title,
        ylabel="Nhãn thực tế (True Label)",
        xlabel="Nhãn dự đoán (Predicted Label)",
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_display[i, j]
            if normalize:
                text_str = f"{val:.1%}" if val >= 0.005 else ("<0.5%" if val > 0 else "0%")
            else:
                text_str = f"{int(val):,}" if val > 0 else "0"

            color = "white" if val > thresh else "black"
            ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=8.5, fontweight="bold" if i == j else "normal")

    ax.grid(False)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_f1_comparison(base_metrics: list[dict], final_metrics: list[dict], path: str):
    """Vẽ biểu đồ cột so sánh F1 từng lớp giữa Baseline và Final Model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    classes = [m["cls"] for m in base_metrics]
    names = [CLASS_SHORT_NAMES[c] for c in classes]
    base_f1 = [m["f1"] for m in base_metrics]
    final_f1 = [m["f1"] for m in final_metrics]
    deltas = [f - b for b, f in zip(base_f1, final_f1)]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.bar(x - width/2, base_f1, width, label="Baseline M-base (Macro-F1: 0.8473)", color="#4682B4", alpha=0.9)
    ax.bar(x + width/2, final_f1, width, label="Mô hình Vô địch M-wide+Adam (Macro-F1: 0.8948)", color="#2E8B57", alpha=0.9)

    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("So sánh Hiệu năng F1-Score Từng Lớp: Baseline vs Mô hình Vô địch", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lớp {c}\n{n}" for c, n in zip(classes, names)], fontsize=9.5)
    ax.set_ylim(0.65, 1.0)
    ax.axhline(0.86, color="red", linestyle="--", alpha=0.7, label="Mục tiêu Rubric (0.86)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    for i in range(len(classes)):
        f_val = final_f1[i]
        d_val = deltas[i]
        sign = "+" if d_val >= 0 else ""
        ax.annotate(
            f"{sign}{d_val*100:.1f}%\n({f_val:.3f})",
            xy=(x[i] + width/2, f_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color="#006400",
        )

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
