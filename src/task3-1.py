from ultralytics import YOLO
import torch

def train_localization():
    """
    [任务3-1] 仅定位瓶子
    基于用户标注的自有数据集 (data.yaml) 训练，不完全依赖通用模型。
    技巧：通过配置 `single_cls=True`，让 YOLO 在读取数据集时无视 4 个分类差异。
    它会自动将所有的类视为单个统一的类别（即“瓶体/目标”），集中学习如何框出瓶子轮廓。
    """
    print("====== 开始执行 Task 3-1: 瓶体定位 (单一目标回归) ======")
    
    if torch.cuda.is_available():
        print(f"✅ 检测到专用显卡: {torch.cuda.get_device_name(0)}，YOLO 将使用 CUDA 进行极速训练！")
        device_id = 0
    else:
        print("⚠️ 未检测到 CUDA，将回退到 CPU。")
        device_id = 'cpu'

    # 仍然加载 YOLO 预训练模型(提取基础纹理能力)，但在本地数据集上微调坐标定位
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data='dataset/data.yaml',
        epochs=30,  # 这里的 epochs 可以根据数据集大小调整
        imgsz=640,
        batch=16,
        device=device_id,
        single_cls=True,  # 核心参数：将 'empty', 'low', 'medium', 'high' 合为 1 个类
        project='runs/task3-1', 
        name='bottle_localization',
        hsv_h=0.015,  # 增强色调
        degrees=10.0   # 增强旋转
    )
    
    print("Task 3-1 训练结束。模型已经掌握在你特有数据集背景下寻找瓶体的能力。")
    print("--------------------------------------------------")
    
    # ======== 新增：训练后进行自动可视化测试 ========
    print(">>> 开始用已训练好的模型进行瓶体定位测试...")
    # 可单独选一些验证集/测试集进行图片批量识别，并保存绘框结果
    model.predict(
        source='dataset/test/images',   # 直接加载测试集文件夹
        save=True,                      # 保存成画好定位框的图片
        project='runs/task3-1',
        name='test_results',
        conf=0.25                       # 只画置信度>0.25的框
    )
    print("✅ 测试画面绘制完毕！你可以直接打开 runs/task3-1/test_results 文件夹，查看所有标注了矩形框的测试原图。")

if __name__ == '__main__':
    train_localization()