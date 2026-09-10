# GazeFlame — GazeTR × FLAME_PyTorch 合并仓库

把 **GazeTR**（Transformer 视线估计）与 **FLAME_PyTorch**（3D 头部参数模型）合并到一个仓库，
形成 **视线估计 → 头部渲染** 的完整流水线：

```
任意人脸图片
   │  common/face.py (dlib)                  人脸检测 + ETH-XGaze 归一化
   ▼
224×224 标准正面人脸裁剪
   │  GazeTR (gazetr/inference.py)          视线估计
   ▼
Pitch / Yaw (弧度)
   │  run_gaze_to_flame.py                  坐标换算 (common/geometry.py)
   ▼
3D 视线向量 (x, y, z)
    │  common/flame_mirror.py + pyrender     FLAME 头部眼球驱动（离屏渲染）
    ▼
带视线箭头的 3D 头部 + 标注输入图（合成到同一个窗口）
```


## 目录结构

```
GazeFlame/
├── common/                  共享工具（几何换算 / 渲染 / 标注，流水线各阶段的唯一数据源）
│   ├── geometry.py          pitch/yaw → 3D 视线向量等
│   ├── face.py              人脸检测 + ETH-XGaze 归一化对齐 (dlib)
│   ├── flame_mirror.py      FLAME 离屏渲染 + 单窗口合成 (compose_with_head)
│   ├── viz.py               2D 视线标注 (draw_overlay, 摄像头/图片共用)
│   ├── download_dlib_model.py 下载 dlib 68点模型 (需联网, 一次性)
│   └── models/              dlib shape_predictor 模型 (gitignore)
├── environment.yml          conda 环境定义 (name: gaze_flame)
├── run.sh                   一键运行脚本（自动定位 conda 环境）
├── run_gaze_to_flame.py     一键编排脚本 (GazeTR → FLAME, 统一单窗口)
├── run_webcam.py            实时摄像头/视频流视线追踪 (统一单窗口)
├── run_webcam.sh            摄像头一键脚本 (自动定位 conda 环境)
├── gazetr/                  原 GazeTR 项目
│   ├── inference.py         推理入口 (含人脸对齐)
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
# dlib 会把 numpy 升到 1.24, 需降回 1.23.5 以兼容 trimesh:
pip install --force-reinstall numpy==1.23.5
# 下载 dlib 68 点人脸关键点模型 (~157MB, 解压后):
python -m common.download_dlib_model
```

## 使用

### 1. 一键串联（GazeTR → FLAME）

```bash
./run.sh                                  # 默认 CPU + 示例图片, 统一窗口显示
./run.sh --image /path/to/face.jpg        # 自定义图片
DEVICE=cuda ./run.sh                      # 有 GPU 时走 cuda
./run.sh --skip-viewer                    # 只算角度, 不显示窗口
./run.sh --headless --save-image out.png  # 无窗口, 把合成图存到 PNG (适合自动化)
./run.sh --no-align                       # 关闭人脸检测/对齐 (输入已是裁剪图时)
```

统一窗口把**标注后的输入图**（视线箭头 + pitch/yaw 读数）和**跟随视线的 FLAME 头部**
左右拼在同一个画面里（不再分别用控制台输出和独立的 pyrender 窗口）。
默认弹出一个交互窗口（需 X11，按任意键关闭）；`--headless --save-image` 则直接存 PNG。

默认会对任意照片做**人脸检测 + ETH-XGaze 归一化对齐**：dlib 检出最大人脸 →
68 点关键点 → solvePnP 头部姿态 → 透视 warp 成 224×224 标准正面裁剪，
与 GazeTR 的训练分布一致（裸图不再被直接拉伸变形）。

### 2. 只看 GazeTR 视线角度

```bash
conda activate gaze_flame
cd gazetr
python inference.py --model model/GazeTR-H-ETH.pt --image images/inference/test.jpg --device cpu
# --no-align          输入已是 224x224 裁剪图时跳过对齐
# --align-out dir     把对齐后的裁剪图(带关键点)存到 dir 便于调试
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

### 4. 实时摄像头 / 视频流（GazeTR + 可选 FLAME 头部，统一单窗口）

```bash
./run_webcam.sh                         # 打开摄像头 0, 实时显示视线箭头
./run_webcam.sh --flame                 # 同一窗口右侧并排 FLAME 头部, 眼球跟随视线
./run_webcam.sh --video clip.mp4        # 也可以处理视频文件
./run_webcam.sh --camera 1 --no-mirror  # 指定摄像头 / 关闭自拍镜像
DEVICE=cuda ./run_webcam.sh             # 有 GPU 时用 cuda
# 无显示环境 (服务器/CI):
./run_webcam.sh --headless --save-frames out/
```

交互控制：`q`/`ESC` 退出，`s` 保存快照。逐帧执行 人脸检测(降采样, ~30ms) →
ETH-XGaze 对齐 → GazeTR 推理（CPU 下本机实测约 **10fps**，受摄像头帧率上限约束）；
加 `--flame` 后 FLAME 头部离屏渲染并按 `--head-fps`（默认 8）节流刷新，
与标注后的画面合成到**同一个** `GazeFlame` 窗口，不拖慢主画面。

## 性能说明

- CPU 推理单张图片仅需数秒（此前记录的 1–2 分钟来自推理脚本中默认启用的
  tkinter 弹窗阻塞，现已默认关闭）。启用人脸对齐后会在推理前额外运行一次
  dlib 检测 + 关键点 + solvePnP，通常增加不到一秒。
- 推理采用 `weights_only=True` 加载纯 state_dict，并批量前向（`--batch_size`）。

## 合并说明

- 两个项目顶层有 `model`、`config.py`、`README.md` 等重名文件，
  因此用 `gazetr/` 与 `flame/` 子目录隔离，避免冲突。
- `run_gaze_to_flame.py` 仍把 GazeTR 推理作为子进程（cwd=`gazetr/`），通过 JSON
  临时文件传递 pitch/yaw；FLAME 头部改为**进程内离屏渲染**（`common/flame_mirror`），
  以便与标注画面合成到同一窗口。
- 统一可视化的公共件：`common/geometry.py`（几何换算）、`common/viz.py`（2D 标注
  `draw_overlay`）、`common/flame_mirror.py`（`FlameMirror` 离屏渲染 + `compose_with_head`
  单窗口合成），两条流水线（图片 / 摄像头）共用，避免重复实现。
- `flame/gaze_viewer.py` 保留为独立调试入口（手动给方向、单看 3D 头部），流水线不再依赖它。

## 原始项目

- GazeTR: https://github.com/hikvision-research/opera (GazeTR)
- FLAME_PyTorch: https://github.com/soubhiksanyal/FLAME_PyTorch
