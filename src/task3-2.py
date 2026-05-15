from ultralytics import YOLO
import torch

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
        name='瓶子液体分析模型' 
    )
    
    print("Task 3-2 训练结束。最佳产出模型通常保存在 runs/task3-2/瓶子液体分析模型/weights/best.pt。")
    print("--------------------------------------------------")
    
    # ======== 新增：端到端推理测试可视化 ========
    print(">>> 开始对测试集进行端到端识别（包含定位和液体残量分级）预测...")
    # 对测试集数据执行跑图测试
    model.predict(
        source='dataset/test/images',
        save=True,                       # 开启可视化并保存
        project='runs/task3-2',
        name='测试集结果',
        conf=0.25                        # 置信度阈值
    )
    print("✅ 预测识别完毕！请打开 runs/task3-2/测试集结果 文件夹查看。")
    print("图片中将会包含各瓶子的准确定位框，以及 Empty, Low, Medium, High 具体的类别和置信度概率值！")

if __name__ == '__main__':
    train_full_detection()()