import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    [新增] CBAM (Convolutional Block Attention Module) 双重注意力机制模块
    空间+通道的双重注意力，找出“什么是液体特征”的同时“特征在哪”。
    增强抗背景干扰能力（如折射横线，液面边缘）。
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class ResNet18WithAttention(nn.Module):
    def __init__(self):
        super(ResNet18WithAttention, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        # 加入 CBAM 注意力机制模块 (通道维数为 512)
        self.cbam = CBAM(512)
        self.avgpool = base_model.avgpool
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class GradCAM:
    """[论文用] 轻量级纯 PyTorch 手写 Grad-CAM 实现用于获取特征热力图"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册向前与向后传播钩子 (Hook)，截取流经特征层的数据
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        loss = output[0, target_class]
        loss.backward()
        
        # 全局平均池化梯度，获得各个特征通道对预测结果的“权重”
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # 将权重叠加到对应的特征热力图上
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        # ReLU 激活函数过滤掉负面干扰像素
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().detach().numpy()

def build_binary_classifier():
    # 替换为带 CBAM 双重注意力机制的网络
    return ResNet18WithAttention()

def main():
    print("====== 正在准备执行 Task 2: 瓶体是否有液体 (二分类) 训练 ======")
    
    # 训练集特有：包含数据增强（旋转、变色、翻转）以提升泛化能力
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0, inplace=False) # 新增: 随机擦除对抗遮挡
    ])
    
    # 验证/测试集特有：绝对不能包含随机增强，必须原图直出（仅做缩放和归一化）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_img_dir = 'dataset/train/images'
    train_lbl_dir = 'dataset/train/labels'
    val_img_dir = 'dataset/valid/images'
    val_lbl_dir = 'dataset/valid/labels'
    
    train_dataset = YoloCropBinaryDataset(train_img_dir, train_lbl_dir, transform=train_transform)
    val_dataset = YoloCropBinaryDataset(val_img_dir, val_lbl_dir, transform=val_transform)
    print(f"提取完成 - 训练集包含 {len(train_dataset)} 个瓶子截图，验证集包含 {len(val_dataset)} 个。")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✅ 检测到专用显卡: {torch.cuda.get_device_name(0)}，即将使用 CUDA 狂飙加速训练！")
    else:
        print("⚠️ 未检测到可用的 CUDA 环境，默认回退到 CPU (速度较慢)。请确认已经安装了 GPU 版的 PyTorch！")
        
    model = build_binary_classifier().to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 修改: 加入标签平滑
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # 修改: 加入L2正则化
    
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # 新增: 余弦退火学习率调度器
    best_val_acc = 0.0 # 新增: 跟踪最佳准确率
    save_path = 'runs/task2/resnet18_二分类权重.pth'
    os.makedirs('runs/task2', exist_ok=True)
    
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
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {curr_lr:.6f} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        scheduler.step() # 步进更新学习率
        
        # 保存最佳模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
    
    print(f"训练完成！最佳验证集准确率为: {best_val_acc:.2f}%。包含本地数据学习特征的权重已保存至: {save_path}")
    
    # 在计算指标和绘图之前，确保我们加载的是这 20 轮里面准确率最高的一轮模型，而不是最终可能过拟合的模型
    model.load_state_dict(torch.load(save_path))
    
    # 设置 matplotlib 支持中文显示和负号正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ======== 1. 验证集全面综合评价：分类报告、混淆矩阵、ROC/AUC ========
    print("\n>>> 开始在验证集上计算评价指标 (Confusion Matrix, ROC Curve, AUC)...")
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for v_inputs, v_labels in val_loader:
            v_inputs = v_inputs.to(device)
            v_outputs = model(v_inputs)
            # 通过 Softmax 提取分类置信度用来给截距 ROC/AUC 定锚
            probs = F.softmax(v_outputs, dim=1)[:, 1] 
            _, v_pred = torch.max(v_outputs, 1)
            
            all_labels.extend(v_labels.numpy())
            all_preds.extend(v_pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # ------ 分类报告 ------
    print("\n【论文指标】验证集统等精确指标与召回汇报库:")
    print(classification_report(all_labels, all_preds, target_names=["Empty (0/空瓶)", "Liquid (1/有液体)"]))

    # ------ 混淆矩阵 (Confusion Matrix) 绘制 ------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["空瓶", "有液体"], yticklabels=["空瓶", "有液体"])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵 (Task2)')
    cm_path = 'runs/task2/混淆矩阵.png'
    plt.savefig(cm_path)
    plt.close()

    # ------ ROC及AUC 绘制 ------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('受试者工作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    roc_path = 'runs/task2/ROC曲线.png'
    plt.savefig(roc_path)
    plt.close()
    
    # ======== 2. 绘制 Grad-CAM 热力遮罩叠加分析图 ========
    print("\n>>> 开始为若干验证集图片探寻判断因果并生成 Grad-CAM 热力图...")
    model.train() # Grad-CAM由于需提取反向梯度反响所以要临时启反流抓取
    
    # 改为挂载在 ResNet18 的最后一层真正卷积特征上，而不是直接挂载在整个 CBAM 类上，使得梯度的纯粹性更强
    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer)
    
    # 【修改点 1】为了使得每次画图都能随机抽验证集，临时创建一个 shuffle=True 的 DataLoader
    random_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    dataiter = iter(random_val_loader)
    images, labels = next(dataiter)
    images_gpu = images.to(device)
    
    # 【修改点 2】更改画布比例为 2行 x 4列，共 8 张图片
    fig = plt.figure(figsize=(16, 9)) 
    
    # 【修改点 3】增加抓取数量上限至 8 
    for i in range(min(6, len(images))):
        img_tensor = images_gpu[i:i+1] # 取得切片
        img_label = labels[i].item()
        
        # 重新获取预测结果用于可视化标题
        with torch.no_grad():
            pred_out = model(img_tensor)
            _, img_pred = torch.max(pred_out, 1)
            img_pred = img_pred.item()
            
        # 考虑到 GradCAM 需要梯度
        img_tensor.requires_grad_(True)
        # 将张量送回给 Cam类要求打热量特征评分
        cam_map = cam_extractor.generate(img_tensor)
        
        # 准备原格式图片并使用OpenCV作为着色剂背景图去拼合热力区
        img_np = images[i].numpy().transpose((1, 2, 0))
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_np = np.clip(std * img_np + mean, 0, 1)
        
        # 将缩小或放大了的特质图拉伸重塑回去以适应 224x224 原图，增加 INTER_CUBIC 双三次插值消除马赛克方块感
        cam_resize = cv2.resize(cam_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resize), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 增加阈值掩膜（Mask）：完全剔除低于 20% 重视度的低热量蓝色无效区域
        mask = cam_resize > 0.2
        mask = np.expand_dims(mask, axis=-1)

        # 仅在有明显响应的高光区域叠加图层（40%红黄渐变），其他区域完全透过底层的高清未改变实景图
        overlay = np.where(mask, 0.6 * img_np + 0.4 * heatmap, img_np)
        
        # 【修改点 4】更改坐标轴矩阵分配为 2x4
        # 第一排原图
        ax = fig.add_subplot(2, 8, i+1, xticks=[], yticks=[])
        ax.imshow(img_np)
        
        # 按照预测对错设置标题颜色与内容 (包含 Pred 和 True)
        pred_text = "有液体" if img_pred == 1 else "空瓶"
        true_text = "有液体" if img_label == 1 else "空瓶"
        color = "green" if img_pred == img_label else "red"
        ax.set_title(f"预测: {pred_text}\n真实: {true_text}", color=color, pad=10)
        
        # 第二排热力图
        ax2 = fig.add_subplot(2, 8, i+9, xticks=[], yticks=[])
        ax2.imshow(overlay)
        ax2.set_title("Grad-CAM 焦点区", pad=10)
        
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0) # 增加各个子图和标题的间距
    viz_path = 'runs/task2/验证集GradCAM可视化结果.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight') # 满足论文要求，输出 300DPI 并且移除页面多余白边
    plt.close()
    
    print(f"✅ 数学建模专用论证图表均已成功保存在 runs/task2/ 目录下。准备随时拿去排版论文：")
    print(f" - [混淆矩阵用于证明无极端误判情况] {cm_path}")
    print(f" - [ROC曲线与AUC用于提供坚实模型置信区间数学参考] {roc_path}")
    print(f" - [GradCAM热点图用于直观展示模型“看见反光和液底”时的机理抓取有效性] {viz_path}")

if __name__ == '__main__':
    main()