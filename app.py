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

# Custom CSS phong cách Trắng - Xanh biển học thuật, tỷ lệ chuẩn màn hình lớn
st.markdown("""
<style>
    /* Nền trang web */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Căn chỉnh độ rộng khối giao diện chính để không bị co cụm hoặc quá bẹt */
    .main .block-container {
        max-width: 1280px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }
    
    /* Header chính màu xanh biển sâu trang nhã */
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        padding: 30px 36px;
        border-radius: 14px;
        margin-bottom: 28px;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.16);
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 30px !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0 0 10px 0;
    }
    .main-header p {
        color: #e0f2fe !important;
        font-size: 15px;
        margin: 0;
        font-weight: 400;
    }
    .author-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 14px;
        margin-top: 14px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.3px;
    }

    /* Khối thẻ hộp chính màu trắng (Card Container) */
    .ui-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 26px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        min-height: 480px;
        display: flex;
        flex-direction: column;
    }

    /* Tiêu đề từng cột */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0369a1;
        margin-bottom: 18px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0f2fe;
    }

    /* Ép tất cả nhãn widget, chữ hiển thị sang màu đen xanh than đậm rõ nét */
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, .stSlider label {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }

    /* Khung tải file Uploader: Nền trắng viền xanh, loại bỏ hoàn toàn ô đen */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #93c5fd !important;
        border-radius: 12px !important;
        padding: 24px 16px !important;
        min-height: 120px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #0284c7 !important;
        background-color: #f0f9ff !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #334155 !important;
        font-size: 14px !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    [data-testid="stFileUploader"] section button * {
        color: #ffffff !important;
    }

    /* Slider chữ và số */
    [data-testid="stSlider"] div, [data-testid="stThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
        color: #0369a1 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Khung kết quả dự đoán hàng đầu */
    .prediction-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .prediction-label {
        font-size: 24px;
        font-weight: 800;
        color: #0369a1;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .prediction-sub {
        font-size: 14px;
        color: #475569;
    }

    /* Thanh tiến trình phân phối xác suất */
    .stProgress > div > div > div > div {
        background-color: #0284c7;
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
        help="Hệ thống tự động tiền xử lý ảnh về kích thước 224x224 và chuẩn hóa chuẩn ImageNet."
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
        <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 48px 20px; text-align: center; color: #64748b; background-color: #ffffff; margin-top: 16px;">
            <div style="font-size: 16px; font-weight: 600; color: #334155; margin-bottom: 6px;">Khu vực hiển thị ảnh kiểm thử</div>
            <div style="font-size: 13px; color: #94a3b8;">Sau khi bạn chọn file ở khung trên, hình ảnh xem trước sẽ xuất hiện tại đây.</div>
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
            <div class="prediction-sub">DỰ ĐOÁN HÀNG ĐẦU (TOP-1)</div>
            <div class="prediction-label">{best_label}</div>
            <div class="prediction-sub">Độ tự tin: <strong>{best_conf:.2f}%</strong> • Thời gian xử lý: <strong>{latency_ms:.1f} ms</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 12px;'>Phân phối xác suất các lớp:</p>", unsafe_allow_html=True)
        for rank, (val, idx) in enumerate(zip(top_values, top_indices), start=1):
            class_name = CLASS_LABELS_VI[CLASSES_RAW[idx]]
            prob_percent = float(val) * 100
            
            col_label, col_val = st.columns([3, 1])
            with col_label:
                st.markdown(f"<span style='font-size: 14px; color: #1e293b; font-weight: 500;'>#{rank}. {class_name}</span>", unsafe_allow_html=True)
            with col_val:
                st.markdown(f"<span style='font-size: 14px; color: #0369a1; font-weight: 700;'>{prob_percent:.2f}%</span>", unsafe_allow_html=True)
                
            st.progress(float(val))
            
        st.markdown("---")
        ckpt_name = os.path.basename(checkpoint_path) if checkpoint_path else "ResNet-18 (Weights mặc định)"
        st.caption(f"Trọng số sử dụng: `{ckpt_name}` | Thiết bị tính toán: `{device.type.upper()}`")
        
    else:
        st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 75px 24px; text-align: center; color: #64748b; margin-top: 16px;">
            <div style="font-size: 18px; font-weight: 600; color: #0369a1; margin-bottom: 10px;">Chờ dữ liệu phân tích</div>
            <div style="font-size: 14px; color: #64748b; line-height: 1.6; max-width: 420px; margin: 0 auto;">
                Hãy tải một hình ảnh từ thiết bị ở cột bên trái để mô hình ResNet-18 bắt đầu trích xuất đặc trưng và dự đoán nhãn phân loại.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer trang
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 13px; padding-bottom: 24px;">
    Đồ án môn học: Mạng Nơ-ron Sâu (Deep Neural Networks) • <strong>Sinh viên thực hiện: BLong1501</strong>
</div>
""", unsafe_allow_html=True)
