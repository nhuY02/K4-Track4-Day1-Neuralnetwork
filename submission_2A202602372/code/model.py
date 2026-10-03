"""model.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm/class có `raise NotImplementedError`.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
        in_features: số đặc trưng đầu vào (54)
        num_classes: số lớp đầu ra (7)
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        init: str = "he",
        in_features: int = 54,
        num_classes: int = 7,
    ):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout_rate = float(dropout)
        self.init_mode = init
        self.in_features = in_features
        self.num_classes = num_classes

        # Dựng danh sách các lớp tuần tự theo đúng quy chuẩn kiến trúc
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h in self.hidden:
            # 1. Lớp tuyến tính có bias
            layers.append(nn.Linear(prev_dim, h, bias=True))
            # 2. Hàm kích hoạt phi tuyến ReLU
            layers.append(nn.ReLU())
            # 3. Dropout chỉ đặt sau ReLU của lớp ẩn (khi dropout > 0)
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = h

        # Lớp tuyến tính cuối cùng: đầu ra là logits thô (B, 7), KHÔNG có ReLU hay Softmax
        layers.append(nn.Linear(prev_dim, num_classes, bias=True))

        # Đóng gói kiến trúc thành nn.Sequential
        self.net = nn.Sequential(*layers)

        # Khởi tạo trọng số theo chế độ được chỉ định
        init_weights(self, init)

        # Kiểm tra tính toàn vẹn số lượng tham số quy định (assert)
        if self.hidden in EXPECTED_PARAMS:
            expected = EXPECTED_PARAMS[self.hidden]
            actual = count_params(self)
            assert actual == expected, (
                f"Lỗi tham số: Kiến trúc {self.hidden} có {actual} tham số, khác chuẩn {expected}!"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear: Kaiming Uniform a=sqrt(5))
    """
    if init == "default":
        # Giữ nguyên mặc định PyTorch
        return

    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            else:
                raise ValueError(
                    f"Khởi tạo '{init}' không hợp lệ. Chọn: 'zeros', 'normal', 'xavier', 'he', 'default'."
                )

            # Bias luôn khởi tạo bằng 0 theo quy định
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor, track_layer: str = "relu") -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp (ở bước 0, một lô val) — dùng cho thí nghiệm khởi tạo.

    Args:
        model: Mô hình MLP
        x: Batch tensor dữ liệu đầu vào (B, in_features)
        track_layer: Đo sau 'linear' hay sau 'relu' (mặc định 'relu')

    Trả về:
        Danh sách std của activation qua từng tầng
    """
    was_training = model.training
    model.eval()
    stds: list[float] = []
    h = x

    for layer in model.net:
        h = layer(h)
        if track_layer == "linear" and isinstance(layer, nn.Linear):
            stds.append(float(h.std().item()))
        elif track_layer == "relu" and isinstance(layer, nn.ReLU):
            stds.append(float(h.std().item()))

    model.train(was_training)
    return stds


def check_grad_flow(model: nn.Module) -> dict[str, float]:
    """Kiểm tra gradient chảy tới mọi tham số (tránh vanishing/exploding gradient).

    Trả về:
        Dict {tên_tham_số: grad_norm_l2}
    """
    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                grad_norms[name] = 0.0
            else:
                grad_norms[name] = float(param.grad.detach().norm(2).item())
    return grad_norms

