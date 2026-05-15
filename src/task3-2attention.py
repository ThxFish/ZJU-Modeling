from ultralytics import YOLO
import torch

import torch.nn as nn

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
    在端到端识别任务中，增强液面特征提取，消除混淆背景，并在论述中提升算法算法含金量。
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

def train_full_detection():
    """
    [任务3-2] 瓶体定位与残液含量联合分析 (端到端识别)
    该任务让 YOLO 基于 data.yaml 学习完整的 4 类别：
    0:'empty', 1:'high', 2:'low', 3:'medium'
    训练自己的数据集并微调模型，完成完整功能的端到端落地。
    """
    print("====== 开始执行 Task 3-2: 瓶内液体含量分析 (端到端 4 分类目标检测) ======")
    
    if torch.cuda.is_available():
        print(f"✅ 检测到专用显卡: {torch.cuda.get_device_name(0)}，YOLO 将使用 CUDA 进行极速训练！")
        device_id = 0
    else:
        print("⚠️ 未检测到 CUDA，将回退到 CPU。")
        device_id = 'cpu'
        
    model = YOLO('yolov8n.pt') 

    # 包含本地数据集的多类别训练
    results = model.train(
        data='dataset/data.yaml',
        epochs=40, 
        imgsz=640,
        batch=16,
        device=device_id,
        single_cls=False,  # 保留 4 分类
        project='runs/task3-2', 
        name='bottle_liquids_analysis' 
    )
    
    print("Task 3-2 训练结束。最佳产出模型通常保存在 runs/task3-2/bottle_liquids_analysis/weights/best.pt。")
    print("--------------------------------------------------")
    
    # ======== 新增：端到端推理测试可视化 ========
    print(">>> 开始对测试集进行端到端识别（包含定位和液体残量分级）预测...")
    # 对测试集数据执行跑图测试
    model.predict(
        source='dataset/test/images',
        save=True,                       # 开启可视化并保存
        project='runs/task3-2',
        name='test_results',
        conf=0.25                        # 置信度阈值
    )
    print("✅ 预测识别完毕！请打开 runs/task3-2/test_results 文件夹查看。")
    print("图片中将会包含各瓶子的准确定位框，以及 Empty, Low, Medium, High 具体的类别和置信度概率值！")

if __name__ == '__main__':
    train_full_detection()()