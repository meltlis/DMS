# 疲劳驾驶监控系统 (DMS System)

本项目是一个基于 `PyTorch`, `YOLOv11`, 和 `MediaPipe` 构建的高精度疲劳驾驶检测系统。系统结合了**空间特征检测（Spatial Features）**和**时序深度学习（Temporal LSTM）**两阶段识别模型，以克服传统阈值判断的局限性。

## 目前已完成的工作
1. **基础环境搭建**：使用 `uv` 高效构建了 Python 运行环境及依赖管理。
2. **视觉特征提取器 (Visual Engine)**：
   - 接入了 YOLOv11 (`yolov11n.pt`) 用作基础人脸、手机与危险物体的 Bounding Box 追踪与提取。
   - 集成 MediaPipe Face Landmarker，实时提取每一帧的高精度脸部状态数据：EAR (眼睛闭合度)、MAR (嘴巴张开度)、Pitch (低头/抬头角度)、Yaw (侧转头) 与 Roll 等。
3. **时序与动作缓冲区 (Temporal Aggregator)**：
   - 使用滑动窗口（Deque）机制记录过去 30 帧 (1~3秒) 的空间特征数据，消除了部分帧的检测抖动，追踪了点头(nod)、打哈欠(yawn)、闭眼(perclos)等物理逻辑动作。
4. **PyTorch LSTM 深度学习特征整合**：
   - 将原来纯粹的 IF-ELSE 静态硬阈值规则替换为预训练的 `lstm_model.pth` 模型。
   - 构建了适配器，将纯量特征 (EAR, MAR, Pitch 等) 二值化/离散映射为模型所需对齐的 9 维空间表示。
   - `src/decision/lstm_classifier.py` 每一帧抽取历史 30 帧时序数据进行网络前向传播推理疲劳状态，显著提高了泛化能力。
5. **本地数据集测评与评测集同步**：
   - 接入了 DROZY 数据集进行自动化评估脚本 (`scripts/evaluate_dataset_accuracy.py`) 并修正了基于视频原生 `cv2.CAP_PROP_POS_MSEC` 的同步时间戳问题。

---

## 项目代码架构

项目的主目录如下，具有高度模块化的分层设计：

```text
dms-system/
├── configs/                   # 包含系统各项运行阈值的 YAML 配置 (thresholds.yaml, runtime.yaml)
├── scripts/                   # 各项离线评测、诊断工具 (evaluate_dataset_accuracy.py, grid_search.py 等)
├── src/                       # 生产环境管线源码
│   ├── decision/              # 决策层：
│   │   ├── danger.py          # - 检测危险品 (如是否玩手机/未系安全带)
│   │   ├── fsm.py             # - 追踪目标丢失与状态有限状态机 
│   │   ├── rules.py           # - 传统的静态规则判别逻辑 (备用/组合条件)
│   │   └── lstm_classifier.py # - 基于 PyTorch LSTM 的时序疲劳推理引擎 (核心)
│   │
│   ├── features/              # 特征提取层：
│   │   └── face_analyzer.py   # - MediaPipe 人脸 468 关键点解析，计算 EAR/MAR/Pose 等
│   │
│   ├── perception/            # 感知层：
│   │   ├── detector.py        # - YOLOv11 实体及人脸包围盒定位
│   │   └── tracker.py         # - ByteTrack 多目标持续追踪
│   │
│   ├── temporal/              # 时序聚合层：
│   │   └── aggregator.py      # - 汇总特征序列、保存历史帧上下文、为 LSTM 提供输入 Tensor
│   │
│   └── pipeline.py            # 主核心管道：串联感知提取、时序缓存与深度网络判决的总入口
│
├── weights/                   # 存放 YOLO 模型权重 (如 yolov11n.pt) 和其它预训练结构
└── pyproject.toml             # UV / Pip 配置文件，声明所有第三方包依赖
```

## Quickstart (快速开始)

```powershell
cd ddd/ddd2/dms-system

# 使用 uv 构建并激活环境
uv venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
uv pip install -e .
uv add torch # 如果需要手动安装 torch

# 启动系统进行真实摄像头检测（按 Q 退出）
python -m src.pipeline --source 0 --show

# 进行后台静默视频评估 (如 DROZY 集)
python scripts/evaluate_dataset_accuracy.py
```