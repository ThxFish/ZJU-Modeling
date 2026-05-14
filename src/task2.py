import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class YoloCropBinaryDataset(Dataset):
    """
    [Task 2 数据集]
    根据 YOLO 的标签，将原图中的瓶子裁剪出来作为分类任务的输入。
    标签映射机制:
    - 0 ('empty') 映射为 0 (无液体)
    - 1, 2, 3 ('high', 'low', 'medium') 映射为 1 (有液体)
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # 匹配 jpg, jpeg, png 格式的图片
        exts = ('*.jpg', '*.jpeg', '*.png')
        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            # 推倒出标签文件名（替换后缀扩展名）
            name_without_ext = os.path.splitext(basename)[0]
            label_path = os.path.join(label_dir, f"{name_without_ext}.txt")
            
            if not os.path.exists(label_path):
                continue
                
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # 二分类映射逻辑
                        binary_label = 0 if cls_id == 0 else 1
                        # 记录 图片路径、边界框与转换后的标签
                        self.samples.append((img_path, parts[1:5], binary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox_str, label = self.samples[idx]
        
        # 读取图片
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # 解析归一化坐标 cx, cy, bw, bh
        cx, cy, bw, bh = map(float, bbox_str)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        # 边界约束以免超出图片范围
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop_img = img[y1:y2, x1:x2]
        
        # 异常坐标保护（以防裁切失败）
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            crop_img = img
            
        crop_img_pil = Image.fromarray(crop_img)
        if self.transform:
            crop_img_pil = self.transform(crop_img_pil)
            
        return crop_img_pil, label

def build_binary_classifier():
    # 使用预训练模型进行泛化特征提取，缩短训练时间
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # 替换为当前任务需要的 2 分类输出（0:无液体 1:有液体）
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model

def main():
    print("====== 正在准备执行 Task 2: 瓶体是否有液体 (二分类) 训练 ======")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_img_dir = 'dataset/train/images'
    train_lbl_dir = 'dataset/train/labels'
    val_img_dir = 'dataset/valid/images'
    val_lbl_dir = 'dataset/valid/labels'
    
    train_dataset = YoloCropBinaryDataset(train_img_dir, train_lbl_dir, transform=transform)
    val_dataset = YoloCropBinaryDataset(val_img_dir, val_lbl_dir, transform=transform)
    print(f"提取完成 - 训练集包含 {len(train_dataset)} 个瓶子截图，验证集包含 {len(val_dataset)} 个。")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✅ 检测到专用显卡: {torch.cuda.get_device_name(0)}，即将使用 CUDA 狂飙加速训练！")
    else:
        print("⚠️ 未检测到可用的 CUDA 环境，默认回退到 CPU (速度较慢)。请确认已经安装了 GPU 版的 PyTorch！")
        
    model = build_binary_classifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    print(f"开始使用 {device} 训练ResNet18，总轮次：{epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # 验证集评估
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                v_outputs = model(v_inputs)
                _, v_pred = v_outputs.max(1)
                val_total += v_labels.size(0)
                val_correct += v_pred.eq(v_labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    os.makedirs('runs/task2', exist_ok=True)
    save_path = 'runs/task2/resnet18_binary.pth'
    torch.save(model.state_dict(), save_path)
    print(f"训练完成！包含本地数据学习特征的权重已保存至: {save_path}")
    
    # ======== 新增：训练后对若干验证集图片进行可视化预测 ========
    print("\n>>> 开始可视化本次 Task 2 模型的预测结果...")
    model.eval()
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    images_gpu = images.to(device)
    
    with torch.no_grad():
        outputs = model(images_gpu)
        _, preds = torch.max(outputs, 1)
        
    fig = plt.figure(figsize=(12, 6))
    for i in range(min(8, len(images))):
        ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
        # 反归一化
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        pred_text = "Liquid" if preds[i].item() == 1 else "Empty"
        true_text = "Liquid" if labels[i].item() == 1 else "Empty"
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"Pred: {pred_text}\nTrue: {true_text}", color=color)
        
    plt.tight_layout()
    viz_path = 'runs/task2/val_visualization.png'
    plt.savefig(viz_path)
    print(f"✅ 可视化结果已保存！你可以打开 {viz_path} 查看模型识别效果。")

if __name__ == '__main__':
    main()