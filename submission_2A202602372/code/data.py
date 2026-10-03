"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"

    assert train_path.exists(), f"Không tìm thấy file {train_path}. Vui lòng chạy split_data.py trước."
    assert eval_path.exists(), f"Không tìm thấy file {eval_path}. Vui lòng chạy split_data.py trước."

    with np.load(train_path) as tr:
        X_train_full = tr["X"]
        y_train_full = tr["y"]

    with np.load(eval_path) as ev:
        X_eval = ev["X"]
        y_eval = ev["y"]
        eval_row_id = ev["row_id"]

    # Kiểm tra kích thước và kiểu dữ liệu theo đúng quy định của rubric/GUIDE
    assert X_train_full.shape == (464809, 54), f"Shape X_train_full không đúng: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"Shape y_train_full không đúng: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"Shape X_eval không đúng: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Shape y_eval không đúng: {y_eval.shape}"
    assert eval_row_id.shape == (116203,), f"Shape eval_row_id không đúng: {eval_row_id.shape}"

    assert X_train_full.dtype == np.float32, f"Dtype X_train_full phải là float32, hiện tại: {X_train_full.dtype}"
    assert y_train_full.dtype == np.int64, f"Dtype y_train_full phải là int64, hiện tại: {y_train_full.dtype}"
    assert X_eval.dtype == np.float32, f"Dtype X_eval phải là float32, hiện tại: {X_eval.dtype}"
    assert y_eval.dtype == np.int64, f"Dtype y_eval phải là int64, hiện tại: {y_eval.dtype}"
    assert eval_row_id.dtype == np.int64, f"Dtype eval_row_id phải là int64, hiện tại: {eval_row_id.dtype}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    Trả lời: Để tránh rò rỉ thông tin từ tập val/eval vào quá trình huấn luyện (Data Leakage).
    """
    # Chỉ tính trên 10 cột liên tục đầu tiên
    numeric_data = X_tr[:, :N_NUMERIC]
    mean = np.mean(numeric_data, axis=0)
    std = np.std(numeric_data, axis=0)

    # Đảm bảo không chia cho 0 nếu đặc trưng có phương sai bằng 0
    std = np.where(std == 0.0, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_scaled = X.copy()
    safe_std = np.where(std == 0.0, 1.0, std)
    # Áp dụng công thức z-score: (x - mean) / std cho 10 cột liên tục đầu tiên
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / safe_std
    return X_scaled.astype(np.float32)


def prepare_data(device: str = "cpu", val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # 1. Nạp tập train và eval
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    # 2. Tách validation từ train (phân tầng theo nhãn)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # 3. Tính toán thống kê chuẩn hoá CHỈ trên tập train_sub
    mean, std = fit_standardizer(X_tr)

    # 4. Chuẩn hoá các tập bằng cùng bộ mean, std
    X_tr_std = apply_standardizer(X_tr, mean, std)
    X_val_std = apply_standardizer(X_val, mean, std)
    X_eval_std = apply_standardizer(X_eval, mean, std)

    # 5. Chuyển sang PyTorch Tensor và nạp lên thiết bị (device)
    target_device = torch.device(device)
    X_tr_t = torch.as_tensor(X_tr_std, dtype=torch.float32, device=target_device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64, device=target_device)
    X_val_t = torch.as_tensor(X_val_std, dtype=torch.float32, device=target_device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.int64, device=target_device)
    X_eval_t = torch.as_tensor(X_eval_std, dtype=torch.float32, device=target_device)
    y_eval_t = torch.as_tensor(y_eval, dtype=torch.int64, device=target_device)

    # 6. Tính baseline đoán lớp đa số trên tập val
    counts = np.bincount(y_val, minlength=7)
    majority_class = int(counts.argmax())
    majority_acc = float(counts[majority_class] / len(y_val))

    print(f"--- THỐNG KÊ DỮ LIỆU ĐÃ CHUẨN BỊ (Device: {device}) ---")
    print(f"Train sub-set : X {X_tr_t.shape}, y {y_tr_t.shape}")
    print(f"Val set       : X {X_val_t.shape}, y {y_val_t.shape}")
    print(f"Eval set      : X {X_eval_t.shape}, y {y_eval_t.shape}")
    print(f"Chiến lược đoán đa số (lớp {majority_class}) trên Val cho Accuracy: {majority_acc:.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; ta giữ lại trọn vẹn để không bỏ sót mẫu.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
