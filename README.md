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

## 团队协作与 Git 工作流

> 其实你可以直接使用VSCode的图形化界面哦

为了保证代码的稳定与安全，本项目严格实行分支管理。**所有人（包括发起人）都必须在自己的分支上进行开发**，经过检查和 Code Review 后再合并到 `main` 主分支。具体操作规范如下：

### 1. 初始配置与下载仓库
如果你是第一次参与本项目，请先将仓库克隆到本地，并配置运行环境：
```bash
# 1. 克隆仓库到本地
git clone <仓库地址>
cd ZJU-Modeling

# 2. 配置虚拟环境并安装项目依赖
conda create -n yolov11-env python=3.10 -y
conda activate yolov11-env
pip install -r requirements.txt
```

### 2. 同步主干最新代码
在每天开始工作前，或者准备创建新分支前，**务必**拉取远程 `main` 分支的最新代码，以免造成后续合并冲突：
```bash
git checkout main
git pull origin main
```

### 3. 创建并切换到个人开发分支
**严禁直接在 `main` 分支上修改和提交代码！** 
接手新任务时，请基于最新的 `main` 创建并切换到自己的分支。分支命名建议为 `姓名缩写-任务名`（如 `zs-preprocess-data`）：
```bash
git checkout -b <你的分支名称>
```

### 4. 提交你的更改
在你的分支上完成代码编写或论文修改后，在本地暂存并提交：
```bash
# 查看当前修改了哪些文件
git status 

# 暂存所有你更改的文件（按需修改，也可单独 add 某个文件）
git add .

# 提交并写明清晰的提交信息（例如：新增预处理脚本 / 修改Introduction）
git commit -m "你的提交信息"
```

### 5. 推送分支并请求合并 (Pull Request)
当本阶段开发完成，准备将代码合并到主干时，执行以下命令推送到远端：
```bash
# 将本地分支 push 到远程仓库
git push origin <你的分支名称>
```
推送完成后，请前往 GitHub/Gitee 的网页端：
1. 发起一个 **Pull Request (PR)**，请求将 `<你的分支名称>` 合并到 `main` 分支。
2. 通知队友帮你 Review（检查）代码或论文修改。
3. 检查无误且不产生冲突后，点击 Merge 合并入主干，并删掉已经完成使命的远程开发分支。
