# 疲劳驾驶监控系统 (DMS System)

本项目是一个基于 `PyTorch`、`YOLOv11` 和 `MediaPipe` 构建的高精度疲劳驾驶检测系统。系统结合了**空间特征检测（Spatial Features）**和**时序深度学习（Temporal LSTM）**两阶段识别模型，以克服传统阈值判断的局限性。

## 环境要求

- **Python**: >= 3.10
- **操作系统**: Windows / Linux / macOS
- **GPU** (可选): 支持 CUDA 的 NVIDIA 显卡，推荐用于实时推理

## 项目结构

```text
dms-system/
├── configs/                   # 系统阈值与运行配置 (thresholds.yaml, runtime.yaml)
├── scripts/                   # 离线评测、诊断、网格搜索工具
├── src/                       # 核心管线源码
│   ├── decision/              # 决策层
│   │   ├── danger.py          # - 危险行为检测 (手机、安全带)
│   │   ├── fsm.py             # - 状态机与去抖逻辑
│   │   ├── rules.py           # - 静态规则判别逻辑
│   │   └── lstm_classifier.py # - PyTorch LSTM 时序疲劳推理引擎
│   ├── features/
│   │   └── face_analyzer.py   # - MediaPipe 人脸 468 关键点解析 (EAR/MAR/Pose)
│   ├── perception/
│   │   ├── detector.py        # - YOLOv11 检测器
│   │   └── tracker.py         # - 目标追踪器
│   ├── temporal/
│   │   └── aggregator.py      # - 时序特征聚合与滑动窗口
│   └── pipeline.py            # - 主流程入口
├── tests/                     # 单元测试与集成测试
├── web/                       # FastAPI Web 服务
│   ├── app.py                 # - WebSocket + HTTP API 后端
│   └── static/
│       └── index.html         # - 前端监控页面
├── weights/                   # 模型权重存放目录
├── requirements.txt           # pip 依赖列表
├── pyproject.toml             # uv / setuptools 配置
└── start_cuda.bat / .ps1      # CUDA 环境快速启动脚本
```

## 从零开始配置

### 1. 准备代码

```bash
cd dms-system
```

### 2. 创建虚拟环境（强烈推荐）

**使用 venv:**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

**使用 uv (更快):**
```bash
uv venv .venv
.\.venv\Scripts\activate.ps1   # Windows
```

### 3. 安装依赖

**方式 A：使用 pip + requirements.txt**

```bash
# CPU 版本
pip install -r requirements.txt

# CUDA 12.1 版本 (推荐，需 NVIDIA 显卡)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**方式 B：使用 uv**

```bash
uv pip install -e .

# 如果需要 CUDA 版本的 torch
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch / CUDA:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 4. 准备模型权重

系统运行时依赖以下模型文件，已内置或自动下载的无需额外准备：

| 模型 | 路径 | 说明 | 是否必须 |
|------|------|------|----------|
| `yolov11n.pt` | `weights/yolov11n.pt` | YOLO 基础检测模型 | **是**（或保留 fallback 合成模式） |
| `face_landmarker.task` | `web/face_landmarker.task` | MediaPipe 人脸关键点模型 | 首次运行自动下载 |
| `best.pt` (9类行为) | 外部路径 `../Drowsiness-Detection-based-on-yolo11-and-LSTM/runs/detect/train16/weights/best.pt` | 高精度行为检测模型 | 否（有 fallback） |
| `lstm_model.pth` | 外部路径 `../Drowsiness-Detection-based-on-yolo11-and-LSTM/lstm_model.pth` | LSTM 时序推理模型 | 否（加载失败则禁用 LSTM） |

> **提示**：如果你没有额外的 `best.pt` 和 `lstm_model.pth`，系统会自动降级使用标准的 `yolov11n.pt` 和纯规则引擎，仍可正常运行。

### 5. 配置参数

修改 `configs/` 下的 YAML 文件来调整检测灵敏度：

- **`configs/thresholds.yaml`**：检测阈值（如 `ear_threshold`、`perclos_alert`、`phone_duration_seconds` 等）
- **`configs/runtime.yaml`**：运行参数（如 `device: cuda` / `cpu`、`fps`、`width`、`height`）

## 运行项目

### 启动 Web 监控服务（推荐）

Web 界面支持实时摄像头推流、视频文件上传分析、WebSocket 低延迟预览。

```bash
cd web
python app.py
```

服务启动后访问：
- **本地地址**: http://localhost:8000
- **分析接口**: `POST /analyze` (上传图片)
- **实时推流**: `WebSocket /ws` (摄像头帧流)
- **视频文件**: `WebSocket /ws/file` + `POST /upload-video`

### 命令行实时检测

使用本地摄像头：
```bash
python -m src.pipeline --source 0 --show
```

使用视频文件：
```bash
python -m src.pipeline --source path/to/video.mp4 --show
```

静默模式（后台处理，不显示窗口）：
```bash
python -m src.pipeline --source 0 --max-frames 300
```

### 离线数据集评估

```bash
python scripts/evaluate_dataset_accuracy.py
```

运行测试：
```bash
pytest tests/
```

## 快速启动脚本

Windows 用户可直接双击或运行：

```powershell
# PowerShell
.\start_cuda.ps1

# CMD
start_cuda.bat
```

## 核心特性

1. **视觉特征提取器**
   - YOLOv11 实时检测人脸、手机、安全带等目标
   - MediaPipe Face Landmarker 提取 EAR、MAR、头部姿态 (Pitch/Yaw/Roll)

2. **时序聚合**
   - 滑动窗口记录历史帧特征，消除单帧抖动
   - 自动识别点头、打哈欠、持续闭眼等动作

3. **双引擎决策**
   - **规则引擎**：基于 PERCLOS、连续闭眼时长等硬阈值（高可靠）
   - **LSTM 引擎**：基于 30 帧时序数据的深度学习推理（高泛化）
   - 两引擎融合：规则引擎优先，LSTM 辅助升级预警

4. **Web 可视化**
   - 实时视频流标注（BBox、疲劳等级、特征数值）
   - 支持图片分析、视频文件批量处理
   - 一键重置管线状态
## External YOLO And Sequence Model Notes

The optional external project is resolved from `configs/runtime.yaml`:

```yaml
external_project_dir: ../Drowsiness-Detection-based-on-yolo11-and-LSTM-main
behavior_model_path: ""
sequence_model_path: ""
```

When these paths are empty, the pipeline searches the external project for
`runs/detect/train16/weights/best.pt`, `lstm_model.pth`, and
`transformer_model.pth`. The sequence checkpoint is loaded for diagnostics, but
it does not change fatigue state unless `lstm_can_warn` or `lstm_can_alert` is
enabled in `configs/thresholds.yaml`. The bundled external training data is very
small, so do not use the sequence score as a standalone accuracy claim.
