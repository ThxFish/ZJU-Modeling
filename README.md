# 比赛名称 / ZJU-Modeling

## 项目简介
本项目旨在利用YOLO和PyTorch框架，结合OpenCV进行数据预处理，训练一个塑料瓶目标检测模型，并完成相关比赛论文的辅助撰写。

## 目录结构
- `archive1/` & `archive2/`: 原始数据集存档
- `dataset/`: 规划好的训练、验证数据集
- `src/`: 源代码目录
  - `data/`: 数据预处理脚本（基于OpenCV）
  - `models/`: 模型配置文件
  - `train.py`: 模型训练脚本
  - `inference.py`: 模型推理与测试脚本
- `paper/`: LaTeX 论文撰写目录
- `runs/`: 训练产生的日志和权重保存目录
- `notebooks/`: Jupyter Notebook，用于数据探索和可视化
- `requirements.txt`: 环境依赖
