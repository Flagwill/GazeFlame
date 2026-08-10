# GazeFlame — GazeTR × FLAME_PyTorch 合并仓库

把 **GazeTR**（Transformer 视线估计）与 **FLAME_PyTorch**（3D 头部参数模型）合并到一个仓库，
形成 **视线估计 → 头部渲染** 的完整流水线：

```
人脸图片 (224×224)
   │  GazeTR (gazetr/inference.py)          视线估计
   ▼
Pitch / Yaw (弧度)
   │  run_gaze_to_flame.py                  坐标换算
   ▼
3D 视线向量 (x, y, z)
   │  FLAME (flame/gaze_viewer.py)          眼球关节驱动
   ▼
带视线箭头的 3D 头部 (pyrender)
```

## 目录结构

```
GazeFlame/
├── environment.yml          conda 环境定义 (name: gaze_flame)
├── run_gaze_to_flame.py     一键编排脚本 (Linux 版)
├── gazetr/                  原 GazeTR 项目
│   ├── inference.py         推理入口
│   ├── model/GazeTR-H-ETH.pt   预训练模型 (ETH-XGaze)
│   └── images/inference/    示例图片
└── flame/                   原 FLAME_PyTorch 项目
    ├── FLAME.py             FLAME 模型层
    ├── gaze_viewer.py       视线渲染脚本 (新增)
    ├── main.py              演示脚本
    └── model/               generic_model.pkl 等
```

## 环境安装

```bash
# conda 在 /opt/anaconda（若未在 PATH 中先 export）
export PATH="/opt/anaconda/bin:$PATH"

conda env create -f environment.yml
conda activate gaze_flame
```

## 使用

### 1. 只看 GazeTR 视线角度

```bash
conda activate gaze_flame
cd gazetr
python inference.py --model model/GazeTR-H-ETH.pt --image images/inference/test.jpg
# CPU: 加 --device cpu
```

### 2. 只看 FLAME 渲染（手动给方向）

```bash
cd flame
python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1
```

### 3. 一键串联（GazeTR → FLAME）

```bash
cd ..
python run_gaze_to_flame.py --image gazetr/images/inference/test.jpg
```

## 合并说明

- 两个项目顶层有 `model`、`config.py`、`requirements.txt`、`README.md` 等重名文件，
  因此用 `gazetr/` 与 `flame/` 子目录隔离，避免冲突。
- 原始两个项目目录 (`~/Projects/GazeTR`、`~/Projects/FLAME_PyTorch`) 保持不动。
- `run_gaze_to_flame.py` 已从 Windows/PowerShell 版改为 Linux 版（直接调用 python 进程）。

## 原始项目

- GazeTR: https://github.com/hikvision-research/opera (GazeTR)
- FLAME_PyTorch: https://github.com/soubhiksanyal/FLAME_PyTorch