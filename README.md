# 🚀 Phân Loại Ảnh CIFAR-10 với ResNet-18 & Mixup Regularization

> **Đồ án môn học:** Mạng Nơ-ron Sâu (Deep Neural Networks)  
> **Sinh viên thực hiện:** BLong1501, Lê Đức Vũ, Nguyễn Văn Trường  
> **Mô hình xuất sắc nhất:** ResNet-18 Pretrained (ImageNet) + Full Fine-tuning + Mixup Regularization ($\alpha = 0.1$) + Cosine Annealing LR  
> **Kết quả đạt được (3-Seed Benchmark):**  
> * **Validation Accuracy:** `89.67% ± 0.33%` (Kỷ lục: **90.04%**)  
> * **Test Accuracy chính thức:** `89.38% ± 0.33%` (Kỷ lục: **89.76%**)  
> * Vượt qua mô hình Baseline ban đầu hơn **+7.5% đến +11%**  
> * **Demo Live Streamlit Cloud:** [cifar10-resnet18-mixup...streamlit.app](https://cifar10-resnet18-mixup-mzibys6wqeqgqehrzj2mbx.streamlit.app/)

---

## 📂 Cấu trúc thư mục dự án

```text
CuoiKyDL_NR/
├── cuoiky.ipynb                     # Notebook chính chứa toàn bộ thực nghiệm từ Phần A -> E
├── baseline.ipynb                   # Notebook baseline gốc do giảng viên cung cấp
├── reproduce.py                     # Script dòng lệnh độc lập để tái lập đánh giá mô hình
├── app.py                           # Giao diện Web Native Streamlit (Deploy Streamlit Cloud)
├── demo/
│   ├── app_gradio.py                # Giao diện Web Demo ứng dụng nhận diện ảnh (Gradio)
│   └── requirements.txt             # Thư viện phục vụ chạy web demo
├── configs/                         # File cấu hình siêu tham số
│   ├── config.yaml                  # Cấu hình định dạng YAML
│   └── config.json                  # Cấu hình định dạng JSON
├── outputs/                         # Thư mục lưu kết quả và checkpoint mô hình
│   ├── baseline_resnet18.pt         # Checkpoint mô hình baseline đã huấn luyện (44.8 MB)
│   ├── best_model_resnet18.pt       # Checkpoint mô hình tốt nhất (Acc 90.04%)
│   ├── baseline_results.json        # Log kết quả mô hình baseline
│   ├── final_3seeds_results.json    # Log kết quả đánh giá 3 seed độc lập
│   └── mixup_ablation_results.json  # Log quét siêu tham số Mixup và ECE
├── requirements.txt                 # Danh sách thư viện Python cần thiết
├── .gitignore                       # Bỏ qua các file data và cache nặng
└── README.md                        # Tài liệu hướng dẫn chi tiết dự án
```

---

## 🔁 HƯỚNG DẪN TÁI LẬP KẾT QUẢ CUỐI CÙNG (REPRODUCIBILITY GUIDE)

Theo yêu cầu mục 5.2 trong đề bài (`doc.tex`), người chấm có thể tái lập kết quả theo một trong hai cách dưới đây:

### Cách 1: Chạy trực tiếp trên Google Colab / Kaggle (Khuyên dùng)
1. Mở file **`cuoiky.ipynb`** trên Google Colab (chọn Runtime GPU T4) hoặc Kaggle (GPU T4x2).
2. **Nếu muốn chạy toàn bộ đồ án:** Chọn `Runtime` ➔ `Run all` (chạy mượt từ Cell 0 đến Cell 40 không gặp bất kỳ lỗi nào).
3. **Nếu muốn tái lập nhanh mô hình cuối cùng:**
   - Chạy **Cell 0 đến Cell 6**: Thiết lập môi trường và tạo phân chia dữ liệu chuẩn 5.000 ảnh train / 5.000 ảnh val bằng hàm `make_splits(seed=42)`.
   - Nhảy thẳng xuống chạy **Cell 39**: Cell này sẽ thực hiện huấn luyện và đánh giá độc lập mô hình cuối cùng trên 3 seed `[42, 123, 999]`.
   - Xem kết quả tổng hợp chính thức và biểu đồ thanh sai số tại **Cell 40**:
     * Validation Accuracy: `89.67% ± 0.33%`
     * Test Accuracy: `89.38% ± 0.33%`
     * Checkpoint lưu tại `./outputs/best_model_resnet18.pt`.

### Cách 2: Chạy dòng lệnh độc lập từ Terminal (Script CLI)
Không cần mở Jupyter Notebook, bạn có thể chạy file script `reproduce.py` để tự động tải CIFAR-10 test set (10.000 ảnh), nạp checkpoint và in kết quả kiểm thử ngay lập tức:

```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy tái lập kết quả đánh giá mô hình
python reproduce.py --checkpoint ./outputs/baseline_resnet18.pt --batch_size 128
```

---

## 🔬 Tóm tắt các nội dung đã thực hiện

* **Phần A (Thuật toán tối ưu):** Tự cài đặt 4 lớp Optimizer (`SGD`, `Momentum`, `RMSProp`, `Adam`) kế thừa `torch.optim.Optimizer` và Unit test kiểm chứng sai lệch $< 10^{-5}$ so với PyTorch (`PASSED`).
* **Phần B (Khởi tạo, LR & Regularization):** Thực nghiệm LR Range Test tìm LR tối ưu ($0.01$); So sánh 3 LR Schedulers (Step Decay, Cosine, OneCycleLR); So sánh 4 phương pháp khởi tạo trọng số trên CNN 6 lớp; Ablation Study đo lường tác động của Dropout, BatchNorm, Weight Decay, Augmentation.
* **Phần C (Transfer Learning & Fine-tuning):** Thực nghiệm 4 chiến lược (*Scratch, Linear Probe, Full Fine-tuning, Discriminative LR*); Khảo sát đường cong hiệu quả dữ liệu (*Data Efficiency Curve*) tại 1%, 10% và 100% dữ liệu; Nghiên cứu sâu về hành vi của BatchNorm (Batch size 128 vs 8, Train vs Freeze).
* **Phần D (Huấn luyện & Gỡ lỗi):** Hoàn thành 4 Sanity Checks bắt buộc (Initial loss $\approx \ln(10)$, Overfit 1 batch $\to 0$, Augmentation visual check, Shuffled-label test $\approx 10\%$); Ghi nhận nhật ký gỡ lỗi thực tế.
* **Phần E (Nghiên cứu mở rộng):** Nghiên cứu kỹ thuật Regularization nâng cao **Mixup** (*Zhang et al., ICLR 2018*); Khảo sát quét siêu tham số $\alpha \in [0.0, 0.1, 0.2, 0.4, 0.8]$ kết hợp đánh giá đa chỉ số (*Accuracy, Macro F1, Val Loss, ECE Calibration Error*); Xác định điểm ngọt $\alpha = 0.1$.
* **Đánh giá 3 Seed:** Khảo sát độ ổn định thống kê qua 3 hạt giống độc lập ($42, 123, 999$).

---

## 🌐 Ứng dụng Demo Nhận diện Ảnh
* **Ứng dụng Native Streamlit (Live trên Cloud):** [cifar10-resnet18-mixup...streamlit.app](https://cifar10-resnet18-mixup-mzibys6wqeqgqehrzj2mbx.streamlit.app/)
* **Chạy cục bộ:**
  ```bash
  streamlit run app.py
  # hoặc chạy bản Gradio:
  python demo/app_gradio.py
  ```
