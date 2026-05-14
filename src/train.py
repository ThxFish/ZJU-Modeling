# 已废弃,仅作为备用

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
    类别定义例如: 0-empty, 1-high, 2-low, 3-medium
    """
    print("正在初始化 Task 3 - YOLOv8 目标检测模型...")
    # 加载轻量级的预训练模型 YOLOv8n
    model = YOLO('yolov8n.pt') 

    # 训练模型
    results = model.train(
        data='dataset/data.yaml',
        epochs=30,  # 演示用，具体看拟合情况调整
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/train',       # 显式指定保存位置
        name='yolov8_bottle_detection' 
    )
    return model

def test_task3_yolo():
    """
    测试与验证：使用训练好的模型对测试集进行评估和推理
    """
    print("正在加载训练好的模型进行评估...")
    # 注意：这里的模型路径根据实际训练产出的 best.pt 路径进行修改
    # 默认路径通常为 runs/train/yolov8_bottle_detection/weights/best.pt
    model = YOLO('runs/train/yolov8_bottle_detection/weights/best.pt')
    
    # 1. 在测试集上评估指标 (mAP等)
    metrics = model.val(data='dataset/data.yaml', split='test')
    print(f"测试集 mAP50-95: {metrics.box.map}")
    
    # 2. 对测试集中的几张图片进行推理预测并保存结果可视化图片
    print("正在对单张/多张图片进行推理测试...")
    results = model.predict(source='dataset/test/images', save=True, project='runs/detect', name='test_results')
    print("测试完成，预测结果图片已保存在 runs/detect/test_results 文件夹内。")

def test_bottle_localization_only():
    """
    仅测试瓶体定位功能 (Zero-shot / 预训练模型直接测试)
    如果还没有训练属于自己的 best.pt，可以使用 COCO 预训练的 YOLOv8n 直接定位瓶子。
    COCO 数据集中瓶子 (bottle) 的类别 ID 为 39。
    """
    print("正在加载官方预训练 YOLOv8n 模型，直接测试瓶体会定位功能...")
    model = YOLO('yolov8n.pt') 

    # 对测试集图片进行推理，并仅保留类别为 39 (bottle) 的检测框
    # save=True 会将画好框的图片存下来
    print("正在对 dataset/test/images 进行预测...")
    results = model.predict(
        source='dataset/test/images', 
        classes=[39],   # 仅检测瓶子
        save=True, 
        project='runs/detect', 
        name='localization_only_results'
    )
    print("瓶子定位置信度及边界框测试完成，结果已保存在 runs/detect/localization_only_results 文件夹内。")

def main():
    print("====== 建模竞赛 A 题演示代码 ======")
    
    # --- 演示：纯测试预训练模型的瓶体定位 ---
    print("\n>>> 准备测试纯瓶子定位功能...")
    test_bottle_localization_only()
    
    # --- 演示：触发任务 3 的 YOLO 训练 ---
    # print("\n>>> 准备开始训练 Task 3 目标检测模型...")
    # train_task3_yolo()
    
    # --- 演示：触发任务 3 的 YOLO 测试与推理 ---
    # print("\n>>> 准备测试 Task 3 目标检测模型...")
    # test_task3_yolo()

if __name__ == '__main__':
    main()
