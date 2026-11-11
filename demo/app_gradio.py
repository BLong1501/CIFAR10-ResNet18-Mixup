import sys
import os

# ==============================================================================
# TỰ ĐỘNG PHÁT HIỆN: NẾU ĐƯỢC GỌI BỞI STREAMLIT THÌ CHUYỂN SANG STREAMLIT APP
# ==============================================================================
is_streamlit = "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv)

if is_streamlit:
    import runpy
    # Nạp và chạy giao diện Streamlit từ app.py
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    if not os.path.exists(app_path):
        app_path = "app.py"
    runpy.run_path(app_path, run_name="__main__")
else:
    # ==============================================================================
    # NẾU CHẠY BẰNG PYTHON THÔNG THƯỜNG (LOCAL HOẶC COLAB) -> CHẠY GRADIO
    # ==============================================================================
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    import gradio as gr

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_trained_model():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        candidate_paths = [
            "./outputs/best_model_resnet18.pt",
            "../outputs/best_model_resnet18.pt",
            "./outputs/baseline_resnet18.pt",
            "best_model_resnet18.pt"
        ]
        loaded_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    model.load_state_dict(state_dict)
                    loaded_path = path
                    break
                except Exception:
                    pass
        model = model.to(device)
        model.eval()
        return model, loaded_path

    model, current_checkpoint = load_trained_model()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    def predict(image, top_k=3):
        if image is None:
            return {}, "Vui lòng tải ảnh lên để nhận diện."
        start_time = time.time()
        img_rgb = image.convert("RGB")
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits[0], dim=0)
        latency_ms = (time.time() - start_time) * 1000
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

    custom_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#f4f9fd",
        block_background_fill="#ffffff",
        button_primary_background_fill="#0284c7",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(theme=custom_theme, title="CIFAR-10 AI Classifier") as demo:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🚀 CIFAR-10 AI IMAGE CLASSIFIER</h1>
            <p style="color: #e0f2fe; margin-top: 6px;">Hệ thống nhận diện 10 lớp vật thể & động vật (ResNet-18 + Mixup)</p>
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 📤 Tải ảnh kiểm thử")
                input_image = gr.Image(type="pil", label="Kéo thả hoặc dán ảnh vào đây", height=320)
                top_k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Số lượng dự đoán (Top-k)")
                with gr.Row():
                    btn_clear = gr.ClearButton(components=[input_image], value="Xóa ảnh")
                    btn_predict = gr.Button("🔍 Bắt đầu Nhận diện", variant="primary", scale=2)
            with gr.Column(scale=5):
                gr.Markdown("### 📊 Kết quả phân tích")
                output_labels = gr.Label(num_top_classes=3, label="Phân phối xác suất")
                output_info = gr.Markdown("*(Kết quả sẽ xuất hiện ở đây)*")
                
        btn_predict.click(fn=predict, inputs=[input_image, top_k_slider], outputs=[output_labels, output_info])
        input_image.change(fn=predict, inputs=[input_image, top_k_slider], outputs=[output_labels, output_info])

    if __name__ == "__main__":
        demo.launch(share=True)
