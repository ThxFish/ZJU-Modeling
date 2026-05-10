from ultralytics import YOLO
import torch

def main():
    # 建立模型
    # 加载预训练配置，比如 yolov11n.pt 或 yolov8n.pt 等
    model = YOLO('yolov8n.pt') 

    # 训练模型
    # data参数应该指向你的 dataset/data.yaml (确保修改 yaml 里的绝对/相对路径)
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    main()
