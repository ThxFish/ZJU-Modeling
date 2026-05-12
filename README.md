# 比赛名称 / ZJU-Modeling

## 项目简介
本项目旨在利用YOLO和PyTorch框架，结合OpenCV进行数据预处理，训练一个塑料瓶目标检测模型，并完成相关比赛论文的辅助撰写。

## 目录结构
- `dataset/`: 规划好的训练、验证、测试数据集
- `docs/`: 项目文档（论文提纲、数据集构建规范等）
  - `pics/`: 记录项目过程的图片文件
- `src/`: 源代码目录
  - `data/`: 数据预处理脚本（如 preprocess.py）
  - `models/`: 模型配置文件
  - `train.py`: 模型训练脚本
- `paper/`: LaTeX 论文撰写目录
- `notebooks/`: Jupyter Notebook，用于数据探索和可视化
- `tools/`: 辅助工具集（如数据标注工具 labelImg 等）
- `runs/`: 训练产生的日志和权重保存目录（运行训练后自动生成）
- `requirements.txt`: 环境依赖
