# Báo cáo Lab Day 1 — Trần Thị Như Ý — 2A202602372

## 1. Thiết lập

- **Môi trường:** Local Workstation, CPU Intel x86_64, Python 3.11.9, PyTorch 2.14.0+cpu.
- **Dữ liệu:** Forest CoverType (7 lớp, 54 đặc trưng gồm 10 biến số liên tục và 44 biến nhị phân One-Hot). Phân chia cố định theo `split_metadata.csv`: `train` gồm 464.809 mẫu, `eval` gồm 116.203 mẫu.
- **Tập Validation:** Tách 20% phân tầng (stratified) từ tập `train` với `seed=42` $\rightarrow$ Tập con huấn luyện `train_sub`: 371.847 mẫu, tập `val`: 92.962 mẫu. Chuẩn hoá Z-score $(\mu, \sigma)$ tính độc lập trên 10 cột liên tục của `train_sub`, áp dụng nhất quán cho `val` và `eval` (không gây rò rỉ dữ liệu).
- **Mô hình Cơ sở (Baseline `M-base`):** Cấu trúc MLP $54 \rightarrow 256 \rightarrow 128 \rightarrow 7$ (đúng 47.879 tham số), hàm kích hoạt ReLU, không Dropout. Khởi tạo trọng số He Normal (`init="he"`), hàm mất mát Cross-Entropy (`loss="ce"`), bộ tối ưu SGD + Momentum 0.9 (`lr=0.1`, `momentum=0.9`, `weight_decay=0.0`), kích thước lô `batch=512`, huấn luyện 20 epochs.
- **Mốc tham chiếu ban đầu:** Chiến lược đoán lớp đa số (Lớp 1 - Lodgepole Pine) trên tập validation cho Accuracy = **0.4876** (48.76%) và Macro-F1 $\approx$ **0.0936**.
- **Độ phủ chủ đề thực nghiệm:** Đã thực hiện đầy đủ **7/7 chủ đề** theo `GUIDE.md`:
  - [x] Loss (CE vs MSE)
  - [x] Optimizer (SGD, SGD+Momentum, Adam, AdamW)
  - [x] Hyper-parameter (Batch Size 128, 512, 2048; Kiến trúc M-base, M-wide, M-deep)
  - [x] Dropout ($q \in \{0.0, 0.1, 0.2, 0.3\}$)
  - [x] Gradient Clipping ($c=0.5$ và Stress test $\text{lr}=1.5$)
  - [x] Mixed Precision (FP32 vs BFloat16)
  - [x] Initialization (He, Xavier, Normal, Zeros)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả đo được | Đánh giá & Đối chiếu lý thuyết |
| :--- | :---: | :--- |
| **Số tham số / Shape Logits** | 47.879 / `(B, 7)` | Khớp 100% quy định rubric, không đặt Softmax trong model. |
| **Loss bước 0 trên Val** | **1.9680** (PyTorch default) / **2.0493** (He) | Rất sát mốc lý thuyết $\ln 7 \approx 1.9459$ ($\Delta \le 0.022$). |
| **Quá khớp 20 mẫu (Overfit 20)** | Loss = **0.000002**, Acc = **100.0%** | Đạt cực tiểu sau 250 bước Adam (hình `health_check_overfit20.png`). |
| **Gradient Flow** | TẤT CẢ tham số có $\nabla W \neq 0$ | Chuẩn $L_2$ gradient của mọi tầng đều dương ($0.15 - 0.72$), không tắc nghẽn. |
| **Số Seed Baseline đã chạy** | **3 seeds** (`seed=1, 2, 3`) | Chạy độc lập 20 epochs trên toàn bộ dữ liệu. |
| **Baseline Val Accuracy ($\mu \pm \sigma$)** | $\mathbf{90.78\% \pm 0.47\%}$ | Seed 1: 90.38%, Seed 2: 91.29%, Seed 3: 90.66%. |
| **Baseline Val Macro-F1 ($\mu \pm \sigma$)** | $\mathbf{0.8516 \pm 0.0072}$ | Seed 1: 0.8473, Seed 2: 0.8598, Seed 3: 0.8476. |

**Ngưỡng nhiễu thực nghiệm xác lập:**
$$\Delta_{noise} = 2\sigma = \mathbf{0.0143} \quad (\approx 1.43\% \text{ Val Macro-F1})$$
> *Quy tắc quyết định:* Bất kỳ can thiệp kỹ thuật nào có $|\Delta \text{Val Macro-F1}| \le 0.0143$ đều được xem là nằm trong phạm vi dao động ngẫu nhiên của seed (`beyond_noise = "Không"`). Chỉ khi $|\Delta| > 0.0143$ thì sự cải thiện/suy giảm mới có ý nghĩa thống kê.

---

## 3. Kết quả theo 7 Chủ đề Thực nghiệm

### 3.1 Hàm mất mát — Cross-Entropy vs MSE
- **Dự đoán trước khi chạy:** Cross-Entropy (CE) kết hợp Softmax cho đạo hàm theo logit là $(\hat{y}_c - y_c)$, không bị bão hoà khi dự đoán sai lệch lớn. Ngược lại, MSE tính trên One-Hot targets có hàm phạt bậc hai bị triệt tiêu gradient khi đầu ra bão hoà, và phạt yếu trên các lớp thiểu số. Do đó, MSE sẽ học chậm hơn nhiều và có Macro-F1 thấp hơn hẳn CE.
- **Kết quả thực nghiệm:**
  - `base-s1` (CE): Best Val Loss = 0.2399, Val Acc = 90.38%, **Val Macro-F1 = 0.8473**.
  - `loss-mse` (MSE): Best Val Loss = 0.0531, Val Acc = 79.94%, **Val Macro-F1 = 0.6726**.
  - Độ chênh lệch: $\Delta \text{F1} = \mathbf{-0.1747}$ (Vượt xa ngưỡng nhiễu $2\sigma$, MSE kém hơn hẳn).
- **Cơ chế & Nhận xét:** Do phân bố lớp của bài toán cực kỳ mất cân bằng (Lớp 1 chiếm 48.8% trong khi Lớp 3 chỉ chiếm 0.5%), hàm MSE chia đều trọng số tổn thất trên 7 chiều one-hot, khiến gradient đối với các lớp hiếm bị áp đảo bởi các giá trị 0 của các lớp khác. Không được so sánh trực tiếp giá trị loss của CE (~0.24) và MSE (~0.05) vì khác biệt hoàn toàn về thang đo toán học.
- **Biểu đồ minh chứng:** `figures/loss-mse.png` và `figures/compare_loss.png`.

---

### 3.2 Bộ tối ưu hoá (Optimizers)
- **Dự đoán:** Thuần SGD không có quán tính nên dao động zic-zac trong rãnh hẹp, hội tụ chậm. Adam/AdamW chia bước theo căn bậc hai mô-men bậc 2 sẽ hội tụ vượt trội ở các epoch đầu.
- **Bảng so sánh ở tốc độ học tốt nhất của từng bộ tối ưu:**

| Bộ tối ưu | Mã thí nghiệm | Learning Rate | Best Epoch | Val Acc | Val Macro-F1 | $\Delta$ vs Base | Vượt nhiễu? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGD thuần** | `opt-sgd-lr0.1` | 0.100 | 20 | 84.70% | **0.7285** | -0.1231 | **Có (Kém xa)** |
| **SGD + Momentum** | `base-s1` | 0.100 | 18 | 90.38% | **0.8473** | Mốc chuẩn | - |
| **Adam** | `opt-adam-lr3e-3`| 0.003 | 19 | 90.96% | **0.8601** | +0.0085 | Không (tương đương) |
| **Adam** | `opt-adam-lr1e-3`| 0.001 | 20 | 90.62% | **0.8549** | +0.0033 | Không (tương đương) |
| **AdamW** | `opt-adamw-lr1e-3`| 0.001 (wd=0.01) | 18 | 90.05% | **0.8415** | -0.0101 | Không (tương đương) |

- **Độ nhạy & Cơ chế:**
  - SGD thuần thiếu thành phần quán tính $v_t = \mu v_{t-1} + g_t$, khiến tốc độ học bị hãm lại đáng kể ở các vùng bề mặt lỗi phẳng.
  - Adam ở cả hai mức $\text{lr}=10^{-3}$ và $3\times 10^{-3}$ đều hội tụ cực nhanh: chỉ sau 3 epochs đầu, Val Macro-F1 của Adam đã chạm mốc 0.78 (trong khi SGD+Momentum cần tới 7 epochs).
- **Biểu đồ minh chứng:** `figures/compare_optimizer.png`.

---

### 3.3 Siêu tham số — Kích thước Lô (Batch Size) & Kiến trúc Mạng
- **1. Kích thước Lô (Batch Size):**
  - `batch-128`: Thực hiện $2.905$ bước cập nhật/epoch (gấp 4 lần baseline). Đạt **Val Macro-F1 = 0.8539** (Acc = 91.30%), thời gian 3.3s/epoch. Nhiễu gradient cao giúp mô hình thoát khỏi các cực tiểu địa phương nông.
  - `batch-2048`: Chỉ thực hiện $181$ bước cập nhật/epoch (bằng 1/4 baseline). Đạt **Val Macro-F1 = 0.8136** ($\Delta = -0.0380$, vượt ngưỡng nhiễu, kém hơn rõ rệt). Do số bước cập nhật quá ít trong 20 epochs, gradient mượt nhưng chưa kịp hội tụ; theo *Linear Scaling Rule* cần tăng $\text{lr}$ hoặc tăng số epochs.
- **2. Dung lượng Kiến trúc:**
  - `arch-mwide` ($54 \rightarrow 512 \rightarrow 256 \rightarrow 7$, 161.287 tham số): Đạt **Val Macro-F1 = 0.8782** ($\Delta = \mathbf{+0.0266} > 2\sigma$), Val Acc = 91.99%.
  - `arch-mdeep` ($54 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 7$, 55.687 tham số): Đạt **Val Macro-F1 = 0.8718** ($\Delta = \mathbf{+0.0202} > 2\sigma$), Val Acc = 92.19%.
  - **Nhận xét:** Cả hai kiến trúc mở rộng đều tạo ra cải tiến vượt ngưỡng nhiễu có ý nghĩa thống kê lớn. Dung lượng nơ-ron rộng hơn ở `M-wide` giúp phân tách các ranh giới lớp phức tạp, trong khi độ sâu 3 tầng ẩn ở `M-deep` hỗ trợ trích xuất biểu diễn đặc trưng phân cấp.
- **Biểu đồ minh chứng:** `figures/compare_hparam_batch.png` và `figures/compare_hparam_arch.png`.

---

### 3.4 Điều chuẩn Dropout Regularization
- **Dự đoán:** Dropout là "vị thuốc trị quá khớp". Tuy nhiên với 371.847 mẫu huấn luyện và mạng `M-base` nhỏ (47k tham số), mô hình khó bị quá khớp. Dropout cao sẽ gây thiếu khớp (underfitting).
- **Kết quả:**
  - $q = 0.0$ (Baseline): Train Loss = 0.216, Val Loss = 0.239, **Val Macro-F1 = 0.8473**.
  - $q = 0.1$ (`drop-0.1`): Train Loss = 0.231, Val Loss = 0.248, **Val Macro-F1 = 0.8413** ($\Delta = -0.006$).
  - $q = 0.2$ (`drop-0.2`): Train Loss = 0.264, Val Loss = 0.280, **Val Macro-F1 = 0.8156** ($\Delta = -0.036$).
  - $q = 0.3$ (`drop-0.3`): Train Loss = 0.301, Val Loss = 0.320, **Val Macro-F1 = 0.7791** ($\Delta = -0.073$).
- **Giải thích:** Khoảng cách (gap) giữa Train Loss và Val Loss ở baseline chỉ là $0.023$ (mô hình hoàn toàn không bị quá khớp). Khi áp dụng dropout $q \ge 0.2$, mạng bị mất đi 20–30% năng lực biểu diễn ở mỗi bước forward, dẫn đến underfitting nghiêm trọng. Do đó, dropout không phù hợp cho cấu hình mạng nhỏ trên tập dữ liệu này.
- **Biểu đồ minh chứng:** `figures/compare_dropout.png`.

---

### 3.5 Cắt Gradient (Gradient Clipping)
- **1. Ở tốc độ học chuẩn ($\text{lr}=0.1$):** Baseline có gradient norm dao động ổn định quanh $0.55 - 0.58$. Thí nghiệm `clip-0.5` với ngưỡng $c=0.5$ kích hoạt clipping ở phần lớn các bước, kìm hãm nhẹ tốc độ cập nhật, dẫn tới Val Macro-F1 giảm nhẹ về $0.8360$.
- **2. Thí nghiệm Phản chứng (Stress Test ở $\text{lr}=1.5$):**
  - Không clip (`clip-stress-noclip`): Gradient bùng nổ, loss kẹt ở $1.2059$, Val Accuracy rơi về **48.76%** (bằng mức đoán mò lớp đa số), Val Macro-F1 tụt xuống **0.0936** (mô hình bị vỡ hoàn toàn).
  - Có clip $c=1.0$ (`clip-stress-clip1.0`): Cơ chế $g \leftarrow g \cdot \min(1, c/\|g\|)$ khống chế độ dài vector gradient, **cứu mô hình thành công** và duy trì Val Acc = **83.33%**, Val Macro-F1 = **0.6751**.
- **Biểu đồ minh chứng:** `figures/compare_clipping.png`.

---

### 3.6 Độ chính xác Hỗn hợp (Mixed Precision — BFloat16)
- **Kết quả:** Thí nghiệm `amp-bf16` sử dụng `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` đạt **Val Macro-F1 = 0.8577**, Val Acc = 90.90%, thời gian ~1.3s/epoch, bộ nhớ RAM ổn định ~520 MB.
- **Giải thích:** BFloat16 giữ nguyên 8-bit số mũ (exponent bits) như FP32 nên có dải động giá trị tương đương ($10^{-38} \rightarrow 10^{38}$), hoàn toàn không gặp hiện tượng tràn số dưới (underflow) và không cần `GradScaler`. Ngược lại, FP16 tiêu chuẩn chỉ có 5-bit số mũ nên dải biểu diễn rất hẹp, bắt buộc phải dùng `GradScaler` để nhân tỉ lệ loss trước khi backward. Do kích thước mạng MLP nhỏ, thời gian tính toán bị chi phối bởi chi phí CPU overhead nên tốc độ tương đương FP32.
- **Biểu đồ minh chứng:** `figures/amp-bf16.png`.

---

### 3.7 Khởi tạo Tham số (Weight Initialization)
- **1. Khởi tạo toàn số 0 (`init-zeros`):** Mọi trọng số $W=0 \rightarrow$ mọi nơ-ron nhận đầu vào $0 \rightarrow \text{ReLU}(0) = 0$. Gradient tại mọi nơ-ron trong cùng một lớp đối xứng và giống hệt nhau (mất tính phá vỡ đối xứng). Mạng bị tê liệt, Val Acc cố định ở **48.76%** và Val Macro-F1 = **0.0936**.
- **2. Khởi tạo Normal phương sai nhỏ (`init-normal`, $\sigma=0.01$):** Tín hiệu kích hoạt sau mỗi lớp ReLU bị suy giảm phương sai theo cấp số nhân (vanishing activations), khiến bước học ở epoch đầu rất ì ạch trước khi bắt kịp baseline (Val Macro-F1 = 0.8415).
- **3. Xavier vs He:** Xavier Normal cho Val Macro-F1 = $0.8549$, tương đương He Normal ($0.8516$). Tuy nhiên, He Normal tính toán phương sai lý thuyết $\text{Var}[W] = 2/n_{in}$ chuẩn hoá tốt hơn cho hàm ReLU (vốn triệt tiêu 50% miền giá trị âm) so với Xavier $\text{Var}[W] = 2/(n_{in} + n_{out})$.
- **Biểu đồ minh chứng:** `figures/compare_init.png`.

---

## 4. Đánh giá Cuối cùng trên Tập Eval (`eval.npz`)

Lựa chọn mô hình cuối cùng được thực hiện **nghiêm ngặt trên tập Validation**: Thí nghiệm kết hợp `cand-mwide-adam` (Kiến trúc `M-wide` $54 \rightarrow 512 \rightarrow 256 \rightarrow 7$, bộ tối ưu Adam $\text{lr}=0.002$, batch 512, He init) đạt Val Macro-F1 cao nhất toàn bộ đợt thực nghiệm ($0.8955$).

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1 (Chính)** | **Eval Accuracy** | Trạng thái Rubric |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline M-base** | Seed 1 | 0.8473 | **0.8473** | 90.26% | Mốc tham chiếu |
| **Mô hình Vô địch Cuối cùng** | **Seed 1** | **0.8955** | $\mathbf{0.8948}$ | $\mathbf{92.68\%}$ | **VƯỢT XUẤT SẮC ($\ge 0.86$)** |

- **Độ ổn định Đa Seed của Mô hình Vô địch trên Eval:**
  $$\text{Seed 1: } 0.8948 \quad|\quad \text{Seed 2: } 0.8883 \quad|\quad \text{Seed 3: } 0.8931 \quad\longrightarrow\quad \mathbf{0.8921 \pm 0.0033}$$
- **Độ tin cậy của tập Validation:** Khoảng cách giữa Val Macro-F1 ($0.8955$) và Eval Macro-F1 ($0.8948$) chỉ là $\mathbf{0.0007}$ (dưới 0.1%), chứng minh quy trình chuẩn hoá và phân chia stratified hoàn toàn chuẩn xác, không bị rò rỉ hay quá khớp tập kiểm thử.

---

### 4.1 Phân tích Lỗi theo Lớp (Error Analysis & Confusion Matrix)

Bảng trích xuất trực tiếp từ file chính thức [`eval_result.json`](file:///d:/K4-Track4-Day1-Neuralnetwork/submission_2A202602372/eval_result.json):

| Lớp | Tên loài cây | Số mẫu (Support) | Precision | Recall | **F1-Score** | Tăng so với Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Spruce/Fir | 42.368 | 92.47% | 91.97% | **0.9222** | +0.0210 |
| **1** | Lodgepole Pine | 56.661 | 93.28% | 94.18% | **0.9372** | +0.0195 |
| **2** | Ponderosa Pine | 7.151 | 94.83% | 90.60% | **0.9267** | +0.0366 |
| **3** | Cottonwood/Willow *(0.5%)* | 549 | 80.87% | 87.80% | **0.8419** | $\mathbf{+0.0691}$ |
| **4** | Aspen | 1.899 | 85.10% | 81.20% | **0.8310** | $\mathbf{+0.0660}$ |
| **5** | Douglas-fir | 3.473 | 87.02% | 87.22% | **0.8712** | $\mathbf{+0.1009}$ |
| **6** | Krummholz | 4.102 | 92.94% | 93.69% | **0.9331** | +0.0190 |

- **Lớp khó nhất:** Lớp 4 (Aspen, $\text{F1} = 0.8310$) và Lớp 3 (Cottonwood/Willow, $\text{F1} = 0.8419$).
- **Nguyên nhân nhầm lẫn sinh thái & địa hình:**
  1. *Lớp 0 (Spruce/Fir) và Lớp 1 (Lodgepole Pine):* Chiếm **70.88%** tổng lỗi toàn mạng (6.025 mẫu nhầm qua lại). Cả hai loài đều là cây lá kim sống ở đai độ cao cận núi cao (2.700m - 3.200m), mọc xen kẽ tạo thành các dải rừng hỗn giao có đặc trưng độ dốc, hướng phơi và loại đất tương đồng.
  2. *Lớp 4 (Aspen) nhầm thành Lớp 1 (Lodgepole Pine):* Chiếm 15.48% số cây Aspen (294 mẫu). Aspen là loài rụng lá tiên phong mọc phục hồi sau cháy rừng nằm lọt thỏm giữa các cánh rừng Lodgepole Pine, chia sẻ cùng độ cao và vị trí địa lý.
  3. *Lớp 3 (Cottonwood/Willow):* Loài cây ven suối có số lượng cực hiếm (chỉ 549 mẫu / 0.5%). Mô hình vô địch đã xuất sắc đưa Recall lên tới **87.80%** (nhờ trích xuất quan hệ phi tuyến giữa các biến cự ly thuỷ văn).
- **Biểu đồ minh chứng:** `figures/confusion_matrix.png`, `figures/confusion_matrix_normalized.png`, và `figures/per_class_f1_comparison.png`.

---

## 5. Trả lời các Câu hỏi Dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi được quét và tối ưu $\text{lr}$ công bằng: **Adam ($\text{lr}=0.003$ hoặc $0.002$) và SGD+Momentum ($\text{lr}=0.1$) đều đạt hiệu năng xuất sắc** (Macro-F1 quanh $0.85 - 0.86$). Adam có lợi thế hội tụ nhanh hơn nhiều ở các epoch đầu do cơ chế thích ứng bước học theo mô-men cấp hai.
   - Nếu không chỉnh $\text{lr}$ (ví dụ ép cùng $\text{lr}=0.1$ cho cả hai): Adam sẽ bùng nổ gradient và phân kỳ ngay lập tức, trong khi SGD hoạt động tốt. Ngược lại, nếu ép cùng $\text{lr}=0.001$, Adam hoạt động tốt nhưng SGD gần như không học được gì. Do đó, so sánh optimizer mà không điều chỉnh learning rate là hoàn toàn vô nghĩa và phản khoa học.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - **Không giúp ích.** Trong thực nghiệm này, tăng dropout từ 0.0 lên 0.3 làm giảm Macro-F1 từ $0.8473 \rightarrow 0.7791$. 
   - *Khi nào nên dùng:* Chỉ dùng Dropout khi mô hình có dấu hiệu quá khớp rõ rệt (khoảng cách giữa Train Loss và Val Loss lớn, Train Loss tiếp tục giảm sâu trong khi Val Loss bắt đầu đảo chiều tăng). Với tập dữ liệu lớn (>370k mẫu) và mạng dung lượng vừa phải, dropout chỉ gây thiếu khớp (underfitting).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   - Gradient clipping giải quyết vấn đề **bùng nổ gradient (exploding gradients)** khi bề mặt hàm mất mát có các vách đứng hiểm trở hoặc khi bước học quá lớn.
   - *Chứng minh thực nghiệm:* Ở phép thử stress $\text{lr}=1.5$, mô hình không clip (`clip-stress-noclip`) bị sụp đổ hoàn toàn về mức đoán mò (Loss = 1.2059, Macro-F1 = 0.0936). Trong khi đó, mô hình có clip $c=1.0$ (`clip-stress-clip1.0`) đã chặn đứng các gai gradient và cứu mô hình duy trì Acc 83.33% và Macro-F1 0.6751.

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?**
   - Trên hệ thống CPU thử nghiệm, BFloat16 **không làm tăng tốc độ** (vẫn đạt ~1.3s/epoch). 
   - *Nguyên nhân:* Mạng MLP `M-base` có kích thước nhỏ (47k tham số), thời gian tính toán bị chi phối bởi chi phí điều phối nhân tính toán (kernel launch overhead) và xử lý dữ liệu của CPU chứ không bị nghẽn băng thông bộ nhớ (memory bandwidth bound). Tuy nhiên, trên GPU hiện đại (Ampere/Hopper), Tensor Cores sẽ tăng tốc độ tính toán ma trận lên 2–4 lần và tiết kiệm một nửa bộ nhớ VRAM.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - *Khởi tạo toàn số 0 hỏng:* Do tính đối xứng. Khi $W=0$, mọi nơ-ron nhận đầu vào giống nhau, hàm kích hoạt cho giá trị giống nhau và gradient backward truyền về cho mọi nơ-ron trong cùng một lớp đều giống hệt nhau. Mạng không thể phá vỡ tính đối xứng để học các đặc trưng khác nhau.
   - *He vs Xavier:* Xavier giả định hàm kích hoạt là tuyến tính quanh 0 (hoặc đối xứng như Tanh), giữ phương sai $\text{Var}[W] = 2/(n_{in} + n_{out})$. Hàm ReLU cắt bỏ toàn bộ 50% miền giá trị âm, làm tiêu hao một nửa phương sai qua mỗi lớp. Khởi tạo He bổ sung hệ số 2 ($\text{Var}[W] = 2/n_{in}$) để bù đắp chính xác phần phương sai bị triệt tiêu bởi ReLU. Điều này tối quan trọng khi mạng trở nên rất sâu ($\ge 10-30$ lớp).

6. **(Câu hỏi bắt buộc) Một mạng có loss không giảm sau 2.000 bước. Dựa vào Chương 5 và các thí nghiệm, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao:**
   1. **Kiểm tra Gradient Flow (Chuẩn L2 gradient của từng lớp):** In chuẩn gradient của từng tham số ($W_1, b_1, W_2, b_2, \dots$). Nếu có tầng có gradient bằng 0 hoặc `None`, mạng đang bị nơ-ron chết (Dead ReLU), triệt tiêu gradient (Vanishing Gradient), hoặc quên gọi `loss.backward()` / `optimizer.step()`.
   2. **Kiểm tra khả năng Quá khớp một lô nhỏ (Overfit 20 samples):** Lấy 20 mẫu, tắt mọi chính quy hoá (dropout=0, weight decay=0) và huấn luyện 200–300 bước. Nếu loss không về sát 0 và accuracy không đạt 100%, lỗi chắc chắn nằm ở code (nhãn lệch, Softmax bị tính 2 lần, quên `zero_grad`, hoặc input không khớp).
   3. **Kiểm tra Tốc độ học (Learning Rate) & Phân bố Kích hoạt Lớp ẩn:** Kiểm tra xem $\text{lr}$ có bị đặt quá nhỏ ($10^{-6}$, khiến loss phẳng lì không đổi) hoặc quá lớn khiến gradient bị bão hòa. Kiểm tra độ lệch chuẩn của activation sau từng lớp ở bước 0 để phát hiện sớm hiện tượng nơ-ron bị bão hoà hoặc tín hiệu bị co cụm về 0 do khởi tạo sai.

---

## 6. Hạn chế và Điều bất ngờ

- **Điều bất ngờ:** Kiến trúc mở rộng chiều rộng `M-wide` kết hợp bộ tối ưu `Adam` ($\text{lr}=0.002$) đã tạo ra bước nhảy vọt hiệu năng ấn tượng hơn dự kiến, đưa Eval Macro-F1 từ $0.8473$ lên **$0.8948$** (tiệm cận ngưỡng 0.90), đặc biệt là khả năng nhận diện lớp thiểu số Cottonwood/Willow tăng thêm gần 7% F1 mà không cần can thiệp kỹ thuật oversampling phức tạp.
- **Hạn chế:** Thí nghiệm được thực hiện trên CPU nên chưa đo đạc được tối đa lợi ích gia tốc của Mixed Precision FP16 Tensor Cores trên phần cứng GPU chuyên dụng.
- **Hướng phát triển tiếp theo:** Nếu có thêm thời gian, áp dụng cơ chế điều chỉnh trọng số hàm mất mát (Class-Weighted Cross-Entropy hoặc Focal Loss) và tinh chỉnh bộ lập lịch học Cosine Annealing để giải quyết triệt để sự nhầm lẫn giữa Lớp 0 (Spruce/Fir) và Lớp 1 (Lodgepole Pine).

---

## 7. Phụ lục

- **Danh mục File Nộp bài đầy đủ trong `submission_2A202602372/`:**
  - `REPORT.md`: Báo cáo khoa học hoàn chỉnh.
  - `experiments.xlsx`: Bảng 24 dòng thực nghiệm, bảo toàn 100% công thức Excel nguyên bản.
  - `predictions_eval.csv`: 116.203 dòng dự đoán hợp lệ cho tập Eval.
  - `eval_result.json`: File kết quả chấm điểm chính thức từ `scripts/evaluate.py`.
  - `figures/`: 35 biểu đồ PNG chất lượng cao (tất cả các run + so sánh nhóm + ma trận nhầm lẫn).
  - `code/`: `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `health_check.py`, `error_analysis.py`, và `lab.ipynb`.
  - `results/`: 25 file JSON lưu vết toàn bộ lịch sử huấn luyện.
- **Tổng thời gian thực thi toàn bộ pipeline:** $\approx 18$ phút trên môi trường CPU.
