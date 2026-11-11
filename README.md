# 🚀 Phân Loại Ảnh CIFAR-10 với ResNet-18 & Mixup Regularization

> **Bài tập lớn môn học:** Mạng Nơ-ron Sâu (Deep Neural Networks)  
> **Mô hình xuất sắc nhất:** ResNet-18 Pretrained (ImageNet) + Fine-tuning + Mixup Regularization ($\alpha = 0.1$) + Cosine Annealing LR  
> **Kết quả đạt được (3-Seed Benchmark):**  
> * **Validation Accuracy:** `89.67% ± 0.33%` (Kỷ lục: **90.04%**)  
> * **Test Accuracy chính thức:** `89.38% ± 0.33%` (Kỷ lục: **89.76%**)  
> * Vượt qua mô hình Baseline ban đầu hơn **+7.5% đến +11%**  

---

## 📂 Cấu trúc thư mục dự án

```text
CuoiKyDL_NR/
├── cuoiky.ipynb              # Notebook chính chứa toàn bộ thực nghiệm từ Phần A -> E
├── baseline.ipynb            # Notebook baseline gốc do giảng viên cung cấp
├── demo/
│   ├── app_gradio.py         # Giao diện Web Demo ứng dụng nhận diện ảnh (Gradio)
│   └── requirements.txt      # Thư viện phục vụ chạy web demo
├── outputs/                  # Thư mục lưu kết quả và checkpoint mô hình
│   └── final_3seeds_results.json
├── requirements.txt          # Danh sách thư viện cần thiết
├── .gitignore                # Bỏ qua các file data và cache nặng
└── README.md                 # Hướng dẫn chi tiết dự án
```

---

## 🔬 Tóm tắt các nội dung đã thực hiện

* **Phần A (Thuật toán tối ưu):** Tự cài đặt 4 lớp Optimizer (`SGD`, `Momentum`, `RMSProp`, `Adam`) kế thừa `torch.optim.Optimizer` và Unit test kiểm chứng sai lệch $< 10^{-5}$ so với PyTorch.
* **Phần B (Khởi tạo, LR & Regularization):** Thực nghiệm LR Range Test tìm LR tối ưu ($0.01$); So sánh 3 LR Schedulers (Step Decay, Cosine, OneCycleLR); So sánh 4 phương pháp khởi tạo trọng số trên CNN 6 lớp; Ablation Study đo lường tác động của Dropout, BatchNorm, Weight Decay, Augmentation.
* **Phần C (Transfer Learning & Fine-tuning):** Thực nghiệm 4 chiến lược (*Scratch, Linear Probe, Full Fine-tuning, Discriminative LR*); Khảo sát đường cong hiệu quả dữ liệu (*Data Efficiency Curve*) tại 1%, 10% và 100% dữ liệu; Nghiên cứu sâu về hành vi của BatchNorm (Batch size 128 vs 8, Train vs Freeze).
* **Phần D (Huấn luyện & Gỡ lỗi):** Hoàn thành 4 Sanity Checks bắt buộc (Initial loss $\approx \ln(10)$, Overfit 1 batch $\to 0$, Augmentation visual check, Shuffled-label test $\approx 10\%$); Ghi nhận nhật ký gỡ lỗi thực tế.
* **Phần E (Nghiên cứu mở rộng):** Nghiên cứu kỹ thuật Regularization nâng cao **Mixup** (*Zhang et al., ICLR 2018*); Khảo sát quét siêu tham số $\alpha \in [0.0, 0.1, 0.2, 0.4, 0.8]$ kết hợp đánh giá đa chỉ số (*Accuracy, Macro F1, Val Loss, ECE Calibration Error*); Xác định điểm ngọt $\alpha = 0.1$.
* **Đánh giá 3 Seed:** Khảo sát độ ổn định thống kê qua 3 hạt giống độc lập ($42, 123, 999$).

---

## 🛠️ Hướng dẫn cài đặt và Chạy thử nghiệm

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy Web Demo nhận diện ảnh trực tiếp
```bash
python demo/app_gradio.py
```
Giao diện sẽ khởi chạy tại: `http://127.0.0.1:7860`.
