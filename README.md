# GazeFlame — GazeTR × FLAME_PyTorch 合并仓库

把 **GazeTR**（Transformer 视线估计）与 **FLAME_PyTorch**（3D 头部参数模型）合并到一个仓库，
形成 **视线估计 → 头部渲染** 的完整流水线：

```
人脸图片 (224×224)
   │  GazeTR (gazetr/inference.py)          视线估计
   ▼
Pitch / Yaw (弧度)
   │  run_gaze_to_flame.py                  坐标换算 (common/geometry.py)
   ▼
3D 视线向量 (x, y, z)
   │  FLAME (flame/gaze_viewer.py)          眼球关节驱动
   ▼
带视线箭头的 3D 头部 (pyrender)
```

## 目录结构

```
GazeFlame/
├── common/                  共享工具（几何换算等，流水线各阶段的唯一数据源）
│   └── geometry.py          pitch/yaw → 3D 视线向量等
├── environment.yml          conda 环境定义 (name: gaze_flame)
├── run.sh                   一键运行脚本（自动定位 conda 环境）
├── run_gaze_to_flame.py     一键编排脚本 (GazeTR → FLAME)
├── gazetr/                  原 GazeTR 项目
│   ├── inference.py         推理入口
│   ├── model.py / resnet.py GazeTR 网络定义
│   ├── model/GazeTR-H-ETH.pt   预训练模型 (ETH-XGaze)
│   └── images/inference/    示例图片
└── flame/                   原 FLAME_PyTorch 项目
    ├── FLAME.py             FLAME 模型层
    ├── config.py            FLAME 超参数 (FLAMEConfig dataclass)
    ├── gaze_viewer.py       视线渲染脚本
    ├── main.py              演示脚本（生成 outputs/frame_XX.png）
    └── model/               generic_model.pkl 等
```

## 环境安装

```bash
conda env create -f environment.yml      # 环境名 gaze_flame (Python 3.8)
conda activate gaze_flame
```

## 使用

### 1. 一键串联（GazeTR → FLAME）

```bash
./run.sh                                  # 默认 CPU + 示例图片
./run.sh --image /path/to/face.jpg        # 自定义图片
DEVICE=cuda ./run.sh                      # 有 GPU 时走 cuda
./run.sh --skip-viewer                    # 只算角度, 不启动 3D 窗口
./run.sh --headless --save-image out.png  # 无窗口渲染到 PNG (适合自动化)
```

### 2. 只看 GazeTR 视线角度

```bash
conda activate gaze_flame
cd gazetr
python inference.py --model model/GazeTR-H-ETH.pt --image images/inference/test.jpg --device cpu
# --output results.json  输出机器可读的 JSON
# --image_folder dirs    批量处理整个文件夹
# --popup                每张图弹出窗口（默认关闭，避免阻塞流水线）
```

### 3. 只看 FLAME 渲染（手动给方向）

```bash
cd flame
python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1
# --save-image out.png --headless   无窗口渲染 PNG
```

## 性能说明

- CPU 推理单张图片仅需数秒（此前记录的 1–2 分钟来自推理脚本中默认启用的
  tkinter 弹窗阻塞，现已默认关闭）。
- 推理采用 `weights_only=True` 加载纯 state_dict，并批量前向（`--batch_size`）。

## 合并说明

- 两个项目顶层有 `model`、`config.py`、`README.md` 等重名文件，
  因此用 `gazetr/` 与 `flame/` 子目录隔离，避免冲突。
- 子进程 cwd 分别为 `gazetr/`、`flame/`，脚本内 import 依赖该目录结构；
  `run_gaze_to_flame.py` 通过 JSON 临时文件在阶段间传递结果。
- 视线几何换算集中在 `common/geometry.py`，避免在多处重复定义。

## 原始项目

- GazeTR: https://github.com/hikvision-research/opera (GazeTR)
- FLAME_PyTorch: https://github.com/soubhiksanyal/FLAME_PyTorch
