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
# 1. CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="CIFAR-10 AI Classifier - BLong1501",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS phong cách Trắng - Xanh biển học thuật, thanh lịch
st.markdown("""
<style>
    /* Nền trang web */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header chính màu xanh biển */
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0 0 8px 0;
    }
    .main-header p {
        color: #e0f2fe !important;
        font-size: 14px;
        margin: 0;
    }
    .author-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin-top: 10px;
        font-weight: 500;
        color: #ffffff;
    }

    /* Khối thẻ hộp màu trắng */
    .content-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        height: 100%;
    }

    /* Tiêu đề mục */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0369a1;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0f2fe;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Kết quả dự đoán nổi bật */
    .prediction-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .prediction-label {
        font-size: 22px;
        font-weight: 700;
        color: #0369a1;
        margin-bottom: 4px;
    }
    .prediction-sub {
        font-size: 13px;
        color: #64748b;
    }

    /* Chỉnh màu thanh tiến trình Streamlit */
    .stProgress > div > div > div > div {
        background-color: #0284c7;
    }

    /* Ép tất cả nhãn widget, chữ hướng dẫn sang màu xanh đen đậm rõ nét */
    label, .stSlider label, .stFileUploader label, [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    /* Khung tải file Uploader: Nền trắng viền xanh, xóa bỏ hoàn toàn ô đen */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #93c5fd !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #0284c7 !important;
        background-color: #f0f9ff !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #334155 !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploader"] section button * {
        color: #ffffff !important;
    }

    /* Slider chữ và số */
    [data-testid="stSlider"] div, [data-testid="stThumbValue"] {
        color: #0369a1 !important;
        font-weight: 600 !important;
    }

    /* Ẩn bớt footer và menu mặc định */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH VÀ TIỀN XỬ LÝ ẢNH
# ==============================================================================
CLASSES_RAW = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_LABELS_VI = {
    'airplane': 'Máy bay (Airplane)',
    'automobile': 'Ô tô con (Automobile)',
    'bird': 'Loài chim (Bird)',
    'cat': 'Con mèo (Cat)',
    'deer': 'Hươu / Nai (Deer)',
    'dog': 'Con chó (Dog)',
    'frog': 'Con ếch (Frog)',
    'horse': 'Con ngựa (Horse)',
    'ship': 'Tàu thủy (Ship)',
    'truck': 'Xe tải (Truck)'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "outputs", "best_model_resnet18.pt"),
        "./outputs/best_model_resnet18.pt",
        "../outputs/best_model_resnet18.pt",
        "best_model_resnet18.pt"
    ]
    
    loaded_checkpoint = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                loaded_checkpoint = path
                break
            except Exception:
                pass
                
    # Nếu chưa có weights finetuned trên máy, nạp trọng số pre-trained mặc định để demo luôn hoạt động
    if loaded_checkpoint is None:
        try:
            pt_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.load_state_dict(pt_model.state_dict(), strict=False)
        except Exception:
            pass

    model = model.to(device)
    model.eval()
    return model, loaded_checkpoint

model, checkpoint_path = get_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>HỆ THỐNG PHÂN LỚP HÌNH ẢNH CIFAR-10</h1>
    <p>Mô hình ResNet-18 kết hợp kỹ thuật Mixup Regularization & Cosine Annealing Learning Rate</p>
    <div class="author-badge">Sinh viên thực hiện: BLong1501</div>
</div>
""", unsafe_allow_html=True)

# Bố cục 2 ô chính: Trái (Ảnh) - Phải (Xác suất & Kết quả)
col_left, col_right = st.columns([1, 1], gap="large")

# ----------------- Ô BÊN TRÁI: ĐẨY ẢNH TỪ THIẾT BỊ -----------------
with col_left:
    st.markdown('<div class="section-title">Tải ảnh từ thiết bị</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="Chọn hoặc kéo thả file ảnh từ máy tính (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="Hệ thống hỗ trợ các định dạng ảnh thông dụng."
    )
    
    top_k = st.slider(
        label="Số lượng lớp dự đoán hiển thị (Top-k)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đầu vào đã tải lên", use_container_width=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed #cbd5e1; border-radius: 10px; padding: 40px 20px; text-align: center; color: #64748b; background-color: #ffffff; margin-top: 10px;">
            <p style="font-size: 15px; margin: 0; font-weight: 500;">Chưa có ảnh nào được tải lên</p>
            <p style="font-size: 13px; color: #94a3b8; margin-top: 6px;">Hãy bấm "Browse files" ở trên hoặc kéo thả ảnh vào khung này để kiểm thử.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- Ô BÊN PHẢI: KẾT QUẢ VÀ XÁC SUẤT -----------------
with col_right:
    st.markdown('<div class="section-title">Kết quả phân tích & Phân phối xác suất</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        start_time = time.time()
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits[0], dim=0)
            
        latency_ms = (time.time() - start_time) * 1000
        top_values, top_indices = torch.topk(probs, k=top_k)
        
        best_label = CLASS_LABELS_VI[CLASSES_RAW[top_indices[0]]]
        best_conf = float(top_values[0]) * 100
        
        # Thẻ hiển thị nhãn dự đoán cao nhất
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-sub">NHÃN DỰ ĐOÁN CAO NHẤT</div>
            <div class="prediction-label">{best_label}</div>
            <div class="prediction-sub">Độ tự tin: <strong>{best_conf:.2f}%</strong> • Thời gian suy luận: <strong>{latency_ms:.1f} ms</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Chi tiết xác suất từng lớp:")
        for rank, (val, idx) in enumerate(zip(top_values, top_indices), start=1):
            class_name = CLASS_LABELS_VI[CLASSES_RAW[idx]]
            prob_percent = float(val) * 100
            
            col_label, col_val = st.columns([3, 1])
            with col_label:
                st.write(f"**#{rank}. {class_name}**")
            with col_val:
                st.write(f"`{prob_percent:.2f}%`")
                
            st.progress(float(val))
            
        # Thông tin mô hình và phần cứng
        st.markdown("---")
        ckpt_name = os.path.basename(checkpoint_path) if checkpoint_path else "Mô hình ResNet-18 (Weights mặc định)"
        st.caption(f"Trọng số sử dụng: `{ckpt_name}` | Thiết bị suy luận: `{device.type.upper()}`")
        
    else:
        st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 50px 24px; text-align: center; color: #64748b; margin-top: 10px;">
            <p style="font-size: 16px; font-weight: 500; color: #0369a1; margin-bottom: 8px;">Đang chờ dữ liệu đầu vào</p>
            <p style="font-size: 13px; color: #94a3b8; margin: 0;">Vui lòng tải một hình ảnh từ thiết bị ở ô bên trái để xem phân phối xác suất và kết quả nhận diện từ mô hình.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer trang
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px; padding-bottom: 20px;">
    Đồ án môn học: Mạng Nơ-ron Sâu (Deep Neural Networks) • <strong>Sinh viên: BLong1501</strong>
</div>
""", unsafe_allow_html=True)
