import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr

# ==============================================================================
# 1. CẤU HÌNH NHÃN LỚP VÀ TÊN TIẾNG VIỆT + ICON TRỰC QUAN
# ==============================================================================
CLASSES_RAW = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_LABELS_VI = {
    'airplane': '✈️ Máy bay (Airplane)',
    'automobile': '🚗 Ô tô con (Automobile)',
    'bird': '🐦 Loài chim (Bird)',
    'cat': '🐱 Con mèo (Cat)',
    'deer': '🦌 Hươu / Nai (Deer)',
    'dog': '🐶 Con chó (Dog)',
    'frog': '🐸 Con ếch (Frog)',
    'horse': '🐴 Con ngựa (Horse)',
    'ship': '🚢 Tàu thủy (Ship)',
    'truck': '🚚 Xe tải (Truck)'
}

# ==============================================================================
# 2. TỰ ĐỘNG PHÁT HIỆN THIẾT BỊ VÀ NẠP CHECKPOINT TỐT NHẤT
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Thiết bị suy luận: {device}")

def load_trained_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Ưu tiên tìm checkpoint mô hình cải tiến (Phần E/3-seed), nếu chưa có thì nạp baseline
    candidate_paths = [
        "./outputs/best_model_resnet18.pt",
        "../outputs/best_model_resnet18.pt",
        "./outputs/baseline_resnet18.pt",
        "../outputs/baseline_resnet18.pt"
    ]
    
    loaded_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                loaded_path = path
                print(f"[+] Đã nạp thành công checkpoint: {path}")
                break
            except Exception as e:
                print(f"[-] Lỗi khi đọc {path}: {e}")
                
    if loaded_path is None:
        print("[!] Cảnh báo: Chưa tìm thấy file checkpoint trong ./outputs/. Sử dụng trọng số mặc định để test giao diện.")
        
    model = model.to(device)
    model.eval()
    return model, loaded_path

model, current_checkpoint = load_trained_model()

# ==============================================================================
# 3. TIỀN XỬ LÝ ẢNH CHUẨN ĐÚNG VỚI QUÁ TRÌNH HUẤN LUYỆN RESNET-18
# ==============================================================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==============================================================================
# 4. HÀM SUY LUẬN DỰ ĐOÁN (INFERENCE)
# ==============================================================================
def predict(image, top_k=3):
    if image is None:
        return {}, "Vui lòng tải ảnh lên để nhận diện."
    
    start_time = time.time()
    
    # Chuẩn hóa ảnh RGB và đưa qua transform
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits[0], dim=0)
        
    latency_ms = (time.time() - start_time) * 1000
    
    # Lấy top k xác suất cao nhất
    top_values, top_indices = torch.topk(probs, k=int(top_k))
    top_predictions = {
        CLASS_LABELS_VI[CLASSES_RAW[idx]]: float(val) 
        for val, idx in zip(top_values, top_indices)
    }
    
    best_class = CLASS_LABELS_VI[CLASSES_RAW[top_indices[0]]]
    best_prob = float(top_values[0]) * 100
    
    info_text = (
        f"**Dự đoán cao nhất:** `{best_class}`\n\n"
        f"**Độ tự tin (Confidence):** `{best_prob:.2f}%`\n\n"
        f"**Thời gian suy luận:** `{latency_ms:.1f} ms`\n\n"
        f"**Checkpoint đang dùng:** `{os.path.basename(current_checkpoint) if current_checkpoint else 'Default Model'}`"
    )
    
    return top_predictions, info_text

# ==============================================================================
# 5. THIẾT KẾ GIAO DIỆN GRADIO (TONE MÀU XANH BIỂN & TRẮNG SÁNG)
# ==============================================================================
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#f4f9fd",       # Nền xanh biển nhạt sáng
    block_background_fill="#ffffff",      # Khối nội dung màu trắng tinh
    block_border_width="1px",
    block_border_color="#d0e3f7",
    button_primary_background_fill="#0284c7", # Xanh biển tươi nổi bật
    button_primary_background_fill_hover="#0369a1",
    button_primary_text_color="#ffffff",
)

custom_css = """
/* Tùy chỉnh hiệu ứng bo góc và đổ bóng nhẹ thanh lịch */
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}
.header-box {
    background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
    color: white !important;
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.2);
    text-align: center;
}
.header-box h1 {
    color: white !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
}
.header-box p {
    color: #e0f2fe !important;
    font-size: 15px !important;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="CIFAR-10 AI Classifier") as demo:
    # Header Banner xanh biển trắng
    gr.HTML("""
    <div class="header-box">
        <h1>CIFAR-10 AI IMAGE CLASSIFIER</h1>
        <p>Hệ thống nhận diện 10 lớp vật thể & động vật bằng mô hình Transfer Learning (ResNet-18 + Mixup)</p>
    </div>
    """)
    
    with gr.Row():
        # Cột trái: Tải ảnh
        with gr.Column(scale=5):
            gr.Markdown("###  Tải ảnh kiểm thử")
            input_image = gr.Image(type="pil", label="Kéo thả hoặc dán ảnh vào đây", height=320)
            
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1, 
                    label="Số lượng dự đoán hiển thị (Top-k)"
                )
            
            with gr.Row():
                btn_clear = gr.ClearButton(components=[input_image], value="Xóa ảnh", variant="secondary")
                btn_predict = gr.Button("🔍 Bắt đầu Nhận diện", variant="primary", scale=2)

        # Cột phải: Hiển thị kết quả
        with gr.Column(scale=5):
            gr.Markdown("###  Kết quả phân tích")
            output_labels = gr.Label(num_top_classes=3, label="Phân phối xác suất (Top-k)")
            output_info = gr.Markdown("*(Kết quả chi tiết và thời gian suy luận sẽ xuất hiện ở đây sau khi bạn bấm nhận diện)*")
            
    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align: center; color: #64748b; font-size: 13px;'>"
        "Môn học: Deep Neural Networks (Mạng Nơ-ron Sâu) • Mô hình: ResNet-18 Fine-tuned (ImageNet1K) • Kỹ thuật: Mixup Regularization"
        "</div>"
    )

    # Gán sự kiện cho các nút
    btn_predict.click(
        fn=predict, 
        inputs=[input_image, top_k_slider], 
        outputs=[output_labels, output_info]
    )
    # Tự động nhận diện khi người dùng vừa thả ảnh vào
    input_image.change(
        fn=predict, 
        inputs=[input_image, top_k_slider], 
        outputs=[output_labels, output_info]
    )

# ==============================================================================
# 6. KHỞI CHẠY APP
# ==============================================================================
if __name__ == "__main__":
    # share=True để sinh link công khai truy cập từ xa (rất tiện khi thuyết trình trên Colab)
    demo.launch(share=True)

