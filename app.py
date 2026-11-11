import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import streamlit as st

# ==============================================================================
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="CIFAR-10 AI Classifier",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS tone màu Xanh biển - Trắng sáng thanh lịch
st.markdown("""
<style>
    /* Nền ứng dụng màu xanh biển nhạt sáng */
    .stApp {
        background-color: #f4f9fd;
        color: #1e293b;
    }
    /* Banner tiêu đề gradient xanh biển */
    .header-banner {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.2);
        text-align: center;
    }
    .header-banner h1 {
        color: white !important;
        font-size: 28px !important;
        font-weight: 800;
        margin: 0;
    }
    .header-banner p {
        color: #e0f2fe !important;
        font-size: 15px;
        margin-top: 8px;
    }
    /* Khối thẻ màu trắng bo góc */
    .card-box {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d0e3f7;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CẤU HÌNH NHÃN LỚP VÀ TÊN TIẾNG VIỆT
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
# 3. NẠP MÔ HÌNH VÀ CHECKPOINT TỰ ĐỘNG
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    candidate_paths = [
        "./outputs/best_model_resnet18.pt",
        "./outputs/baseline_resnet18.pt",
        "best_model_resnet18.pt",
        "baseline_resnet18.pt"
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
            except Exception as e:
                pass
                
    model = model.to(device)
    model.eval()
    return model, loaded_path

model, current_checkpoint = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==============================================================================
# 4. GIAO DIỆN THANH BÊN (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/17/Google-flutter-logo.png", width=60)
    st.title("⚙️ Cấu hình Demo")
    st.markdown("---")
    
    top_k = st.slider("Số lượng dự đoán hiển thị (Top-k)", min_value=1, max_value=5, value=3, step=1)
    
    st.markdown("### 📌 Thông tin mô hình")
    st.write(f"• **Kiến trúc:** ResNet-18")
    st.write(f"• **Kỹ thuật:** Mixup Regularization")
    st.write(f"• **Thiết bị chạy:** `{device}`")
    st.write(f"• **Checkpoint:** `{os.path.basename(current_checkpoint) if current_checkpoint else 'Default Model'}`")
    st.markdown("---")
    st.caption("Bài tập lớn Mạng Nơ-ron Sâu (DNN)")

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN PAGE)
# ==============================================================================
st.markdown("""
<div class="header-banner">
    <h1>🚀 CIFAR-10 AI IMAGE CLASSIFIER</h1>
    <p>Hệ thống nhận diện 10 lớp vật thể & động vật sử dụng Transfer Learning (ResNet-18 + Mixup)</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Tải ảnh kiểm thử")
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn ảnh từ máy tính (JPG, PNG, JPEG)", 
        type=["jpg", "png", "jpeg"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đầu vào", use_container_width=True)

with col2:
    st.markdown("### 📊 Kết quả phân tích từ AI")
    
    if uploaded_file is not None:
        start_time = time.time()
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits[0], dim=0)
            
        latency_ms = (time.time() - start_time) * 1000
        
        top_values, top_indices = torch.topk(probs, k=top_k)
        
        best_label = CLASS_LABELS_VI[CLASSES_RAW[top_indices[0]]]
        best_confidence = float(top_values[0]) * 100
        
        # Hộp kết quả nổi bật
        st.success(f"🎯 **Dự đoán:** `{best_label}` | **Độ tự tin:** `{best_confidence:.2f}%`")
        st.caption(f"⏱️ Thời gian suy luận: **{latency_ms:.1f} ms**")
        
        st.markdown("#### Phân phối xác suất (Top-k):")
        for val, idx in zip(top_values, top_indices):
            class_name = CLASS_LABELS_VI[CLASSES_RAW[idx]]
            prob_percent = float(val) * 100
            st.write(f"**{class_name}**: `{prob_percent:.2f}%`")
            st.progress(float(val))
    else:
        st.info("👈 Hãy tải một bức ảnh ở cột bên trái để bắt đầu nhận diện!")
        
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px;'>"
    "Đồ án Mạng Nơ-ron Sâu • Sinh viên: BLong1501 • Deploy trên Streamlit Community Cloud"
    "</div>", 
    unsafe_allow_html=True
)
