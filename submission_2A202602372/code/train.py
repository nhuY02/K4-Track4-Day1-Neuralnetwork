"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.1,                    # Đã chọn bằng val (quét [0.01, 0.03, 0.05, 0.1, 0.2] trên val): lr=0.1 đạt val macro-F1 ~ 0.858
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Khớp 100% logic với scripts/evaluate.py của giảng viên.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    was_training = model.training
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=-1)
        preds.append(pred)
    model.train(was_training)
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    preds = []
    n = len(X)
    num_classes = 7

    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_one_hot = F.one_hot(yb, num_classes=num_classes).float()
            loss = F.mse_loss(logits, y_one_hot, reduction="sum")
        else:
            raise ValueError(f"Hàm mất mát '{loss_name}' không hỗ trợ.")

        total_loss += float(loss.item())
        preds.append(logits.argmax(dim=-1))

    all_preds = torch.cat(preds, dim=0)
    avg_loss = total_loss / n
    acc = float((all_preds == y).float().mean().item())

    # Dựng confusion matrix (7, 7)
    y_true_np = y.cpu().numpy()
    y_pred_np = all_preds.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true_np, y_pred_np), 1)

    macro_f1 = macro_f1_from_confusion(cm)

    model.train(was_training)
    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "cm": cm,
    }


def compute_loss(logits, y, loss_name: str = "ce"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát '{loss_name}' không hỗ trợ. Chọn: 'ce', 'mse'.")



import math
from pathlib import Path
import pandas as pd


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)
        verbose: in log từng epoch nếu True

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    """
    # 0. Trích xuất siêu tham số
    exp_id = cfg.get("exp_id", "exp")
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")
    loss_name = cfg.get("loss", "ce")
    opt_name = cfg.get("optimizer", "sgd_momentum")
    lr = float(cfg.get("lr", 0.01))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    batch_size = int(cfg.get("batch", 512))
    epochs = int(cfg.get("epochs", 20))
    clip_norm = cfg.get("clip_norm", None)
    if clip_norm is not None:
        clip_norm = float(clip_norm)
    precision = cfg.get("precision", "fp32")
    seed = int(cfg.get("seed", 42))

    # 1. Đặt seed để tái lập hoàn toàn
    set_seed(seed)
    device = data["X_tr"].device
    generator = torch.Generator(device=device).manual_seed(seed)

    # 2. Khởi tạo mô hình
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], (
            f"Lỗi số lượng tham số cho kiến trúc {hidden}: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
        )

    # 3. Khởi tạo optimizer
    optimizer = build_optimizer(
        opt_name, model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    # 4. Cấu hình Mixed Precision
    use_amp = (precision == "fp16" and device.type == "cuda") or (precision == "bf16")
    scaler = torch.cuda.amp.GradScaler() if (precision == "fp16" and device.type == "cuda") else None
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = (
        torch.float16 if precision == "fp16"
        else (torch.bfloat16 if precision == "bf16" else torch.float32)
    )

    # 5. Đo Loss bước 0 trên tập Validation (chế độ eval, trước bất kỳ bước huấn luyện nào)
    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = float(step0_eval["loss"])

    if verbose:
        print(f"\n{'='*70}")
        print(f"[{exp_id}] Bắt đầu huấn luyện {epochs} epoch | opt={opt_name} | lr={lr} | batch={batch_size} | init={init}")
        print(f"[{exp_id}] Step 0 Loss on Val: {step0_loss:.4f} (Lý thuyết ln 7 ≈ 1.9459)")
        print(f"{'='*70}")

    # Tập con train cố định để tính train_loss ở chế độ eval()
    # Lấy 50.000 mẫu đầu tiên của train_sub để vừa chính xác vừa tăng tốc độ chạy epoch
    n_train_eval = min(50000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:n_train_eval]
    y_tr_eval = data["y_tr"][:n_train_eval]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_state = None
    diverged = False

    # 6. Vòng lặp huấn luyện chính
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = clip_gradients(model.parameters(), clip_norm)
                    optimizer.step()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            if math.isnan(gn) or math.isinf(gn):
                diverged = True
                break

            batch_grad_norms.append(gn)

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0

        if diverged:
            if verbose:
                print(f"[{exp_id}] CẢNH BÁO: Mô hình phân kỳ (Loss hoặc Gradient NaN/Inf) ở Epoch {epoch}!")
            break

        mean_gn = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        # Đánh giá cuối epoch ở chế độ eval()
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)

        ep_tr_loss = float(tr_eval["loss"])
        ep_val_loss = float(val_eval["loss"])
        ep_val_acc = float(val_eval["acc"])
        ep_val_f1 = float(val_eval["macro_f1"])

        history["epoch"].append(epoch)
        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)
        history["val_macro_f1"].append(ep_val_f1)
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(ep_time)

        # Lưu checkpoint tốt nhất theo val_loss
        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            best_val_acc = ep_val_acc
            best_val_macro_f1 = ep_val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"[{exp_id}] Ep {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {ep_tr_loss:.4f} | "
                f"Val Loss: {ep_val_loss:.4f} | "
                f"Val Acc: {ep_val_acc:.4f} | "
                f"Val F1: {ep_val_f1:.4f} | "
                f"GradNorm: {mean_gn:.3f} | "
                f"Time: {ep_time:.1f}s"
            )

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Đo bộ nhớ
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        try:
            import psutil
            peak_mem_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))
        except Exception:
            peak_mem_mb = 0.0

    mean_time_s = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": round(mean_time_s, 2),
        "peak_mem_MB": round(peak_mem_mb, 1),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "model": model,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Đã ghi file dự đoán ({len(df):,} dòng) -> {p}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str = "predictions_eval.csv") -> dict:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    """
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])

    # Dự đoán trên toàn bộ tập eval ở chế độ eval
    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()

    # Ghi file CSV
    write_predictions(data["eval_row_id"], preds_np, pred_path)

    # Đánh giá nội bộ để đối chiếu
    cm = np.zeros((7, 7), dtype=np.int64)
    y_eval_np = data["y_eval"].cpu().numpy()
    np.add.at(cm, (y_eval_np, preds_np), 1)

    acc = float((preds_np == y_eval_np).mean())
    macro_f1 = macro_f1_from_confusion(cm)

    print(f"\n[Final Eval] Kết quả trên tập eval.npz (116,203 mẫu):")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Macro-F1: {macro_f1:.4f}")

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "cm": cm,
        "pred_path": pred_path,
    }

