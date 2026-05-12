import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

def build_task2_classifier():
    """
    [任务2] 瓶体区域残液识别 (二分类)
    用于已经用 OpenCV 裁剪好的纯瓶身图片，判断是否有液体。
    利用 PyTorch 构建 ResNet18 二分类模型。
    """
    print("正在初始化 Task 2 - ResNet18 分类模型...")
    # 加载预训练的 ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 替换最后的全连接层，适应我们的二分类任务 (0: 无液体, 1: 有液体)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model


def train_task3_yolo():
    """
    [任务3] 瓶体自动定位与残液分级 (端到端目标检测)
    完整原图输入，YOLO 同时输出边界框位置及 4 个类别：
    类别定义例如: 0-no_liquid, 1-low_liquid, 2-medium_liquid, 3-high_liquid
    """
    print("正在初始化 Task 3 - YOLOv8 目标检测模型...")
    # 加载轻量级的预训练模型 YOLOv8n
    model = YOLO('yolov8n.pt') 

    # 训练模型
    # data.yaml 需配置为包含以上 4 个类别的数据集说明文件
    results = model.train(
        data='dataset/data.yaml',
        epochs=50,  # 演示用，可根据需要加大
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        name='yolov8_bottle_detection' # 训练结果保存的文件夹名
    )
    return model

def main():
    print("====== 建模竞赛 A 题演示代码 ======")
    
    # --- 演示：获取任务 2 的模型实例 ---
    resnet_model = build_task2_classifier()
    print("Task 2 模型结构(最后的全连接层):", resnet_model.fc)
    
    # --- 演示：触发任务 3 的 YOLO 训练（取消注释即可运行） ---
    # print("\n>>> 准备开始训练 Task 3 目标检测模型...")
    # train_task3_yolo()

if __name__ == '__main__':
    main()
