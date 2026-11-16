import os
import argparse
import json

# ==============================================================================
# SCRIPT TÁI LẬP KẾT QUẢ MÔ HÌNH CUỐI CÙNG (REPRODUCIBILITY SCRIPT)
# Đồ án môn học: Mạng Nơ-ron Sâu (Deep Neural Networks) - CIFAR-10
# Nhóm thực hiện: BLong1501, Lê Đức Vũ, Nguyễn Văn Trường
# ==============================================================================

def evaluate_checkpoint(checkpoint_path="./outputs/baseline_resnet18.pt", data_dir="./data", batch_size=128):
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> [Reproduce] Đang sử dụng thiết bị: {device}")
    
    # 1. Tìm checkpoint
    candidates = [
        checkpoint_path,
        "./outputs/best_model_resnet18.pt",
        "./outputs/baseline_resnet18.pt",
        "best_model_resnet18.pt"
    ]
    target_ckpt = None
    for c in candidates:
        if os.path.exists(c):
            target_ckpt = c
            break
            
    if target_ckpt is None:
        raise FileNotFoundError("Không tìm thấy file checkpoint (.pt) nào trong thư mục ./outputs!")
        
    print(f"--> [Reproduce] Đang nạp checkpoint từ: {target_ckpt}")
    
    # 2. Khởi tạo mô hình
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 3. Nạp tập Test 10.000 ảnh chính thức của CIFAR-10
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"--> [Reproduce] Đang đánh giá trên tập Test chính thức ({len(test_set)} ảnh)...")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    test_acc = (correct / total) * 100
    print("=" * 60)
    print(f"🎯 KẾT QUẢ TÁI LẬP TRÊN TẬP TEST CHÍNH THỨC:")
    print(f"   • File Checkpoint : {target_ckpt}")
    print(f"   • Số mẫu kiểm thử : {total} ảnh")
    print(f"   • Test Accuracy   : {test_acc:.2f}%")
    print("=" * 60)
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tái lập kết quả mô hình cuối cùng trên CIFAR-10")
    parser.add_argument("--checkpoint", type=str, default="./outputs/baseline_resnet18.pt", help="Đường dẫn file checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="Thư mục chứa dữ liệu CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size khi đánh giá")
    args = parser.parse_args()
    
    evaluate_checkpoint(checkpoint_path=args.checkpoint, data_dir=args.data_dir, batch_size=args.batch_size)
