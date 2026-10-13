# GazeFlame — GazeTR × FLAME_PyTorch 合并仓库

把 **GazeTR**（Transformer 视线估计）与 **FLAME_PyTorch**（3D 头部参数模型）合并到一个仓库，
形成 **视线估计 → 头部渲染** 的完整流水线，并提供面向飞行模拟器的
**驾驶员状态感知与辅助控制概念验证层**：

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

可选概念层：视线落点 → 驾驶舱关注区域 → 操作意图 → 辅助状态
```


## 结题 / 答辩 Demo（含 webcam 实时全链路）

```bash
./run_demo.sh --camera 0                 # 推荐：webcam → 对齐 → GazeTR → FLAME → 辅助意图
./run_demo.sh --camera 0 --windowed      # 普通窗口，便于现场录屏
./run_demo.sh --video driver.mp4         # 视频逐帧真实推理，同一全链路面板
./run_demo.sh                            # 真实单图开场 + 明确标注的合成状态回放（约27秒）
./run_demo.sh --headless --save-video demo_outputs/report_demo.mp4
./run_demo.sh --skip-image --no-flame --headless  # 无摄像头/GL的规则演示
```

实时演示默认启用 FLAME 和驾驶员辅助概念层，展示原始输入、实际网络对齐裁剪、三维头眼、
驾驶舱关注区域、注视意图、原始/滤波角度曲线及事件记录。默认 CPU，并保留原有 ROCm
安全保护。`ASSIST READY` 仅表示软件准备状态，不发送机械臂、飞控或执行器命令。
详细操作、个人校准、答辩讲解顺序、输出文件和研究边界见 **[docs/DEMO.md](docs/DEMO.md)**。

## 目录结构

```
GazeFlame/
├── common/                  共享工具（几何换算 / 渲染 / 标注，流水线各阶段的唯一数据源）
│   ├── geometry.py          pitch/yaw → 3D 视线向量等
│   ├── face.py              人脸检测 + ETH-XGaze 归一化对齐 (dlib)
│   ├── flame_mirror.py      FLAME 离屏渲染 + 单窗口合成 (compose_with_head)
│   ├── viz.py               2D 视线标注 (draw_overlay, 摄像头/图片共用)
│   ├── assistive.py         视线目标/意图/辅助状态概念层与状态面板
│   ├── demo.py              全链路演示面板与明确标注的合成回放场景
│   ├── download_dlib_model.py 下载 dlib 68点模型 (需联网, 一次性)
│   └── models/              dlib shape_predictor 模型 (gitignore)
├── environment.yml          conda 环境定义 (name: gaze_flame_rocm)
├── run.sh                   一键运行脚本（自动定位 conda 环境）
├── run_gaze_to_flame.py     一键编排脚本 (GazeTR → FLAME, 统一单窗口)
├── run_webcam.py            实时摄像头/视频流视线追踪 (统一单窗口)
├── run_webcam.sh            摄像头一键脚本 (自动定位 conda 环境)
├── run_demo.py / run_demo.sh 结题演示：真实webcam/视频全链路 + 单图/合成回放
├── docs/DEMO.md              演示使用与答辩讲解说明
├── profiles/                本地个人校准 JSON（gitignore，首次使用需自行校准）
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
conda env create -f environment.yml      # 环境名 gaze_flame_rocm (Python 3.10 + TheRock 原生 gfx1035)
conda activate gaze_flame_rocm
# 若 dlib 安装后 numpy 仍偏高，需降回 1.23.5 以兼容 trimesh:
pip install --force-reinstall numpy==1.23.5 --no-deps
# 下载 dlib 68 点人脸关键点模型 (~157MB, 解压后):
python -m common.download_dlib_model
```

环境用 AMD **TheRock** 原生 gfx1035 的 torch（`environment.yml` 里的
`torch[device-gfx1035]`，来自 `https://nightly.repo.amd.com/rocm/whl-next/`），
rocBLAS/MIOpen 自带 gfx1035 kernel，无需 `HSA_OVERRIDE_GFX_VERSION`。TheRock 是
nightly，torch 版本会滚动。

## 使用

### 1. 一键串联（GazeTR → FLAME）

```bash
./run.sh                                  # 默认 CPU + 示例图片, 统一窗口显示
./run.sh --image /path/to/face.jpg        # 自定义图片
DEVICE=cuda ./run.sh                      # 有 GPU 时走 cuda
./run.sh --skip-viewer                    # 只算角度, 不显示窗口
./run.sh --headless --save-image out.png  # 无窗口, 把合成图存到 PNG (适合自动化)
./run.sh --assistive --headless --save-image assistive.png  # 增加辅助概念面板
./run.sh --no-align                       # 关闭人脸检测/对齐 (输入已是裁剪图时)
```

统一窗口把**标注后的输入图**（视线箭头 + pitch/yaw 读数）和**跟随视线的 FLAME 头部**
左右拼在同一个画面里（不再分别用控制台输出和独立的 pyrender 窗口）。
默认弹出一个**全屏**交互窗口（需 X11，按任意键关闭），合成画面会用十字准星标出
估计视点位置；`--windowed` 可恢复普通可调整大小的窗口，`--headless --save-image`
则直接存 PNG。

默认会对任意照片做**人脸检测 + ETH-XGaze 归一化对齐**：dlib 检出最大人脸 →
68 点关键点 → solvePnP 头部姿态 → 透视 warp 成 224×224 标准正面裁剪，
与 GazeTR 的训练分布一致（裸图不再被直接拉伸变形）。即使输入恰好是 224×224，
也会先检测并对齐；只有明确传入 `--no-align` 才跳过，适合已经由 ETH-XGaze
生成的标准 crop。

### 2. 只看 GazeTR 视线角度

```bash
conda activate gaze_flame_rocm
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
./run_webcam.sh --assistive             # 增加软件-only 辅助控制概念面板
./run_webcam.sh --video clip.mp4 --flame --assistive --save-frames out/
./run_webcam.sh --video clip.mp4        # 也可以处理视频文件
./run_webcam.sh --camera 1 --no-mirror  # 指定摄像头 / 关闭自拍镜像
DEVICE=cuda ./run_webcam.sh             # 有 GPU 时用 cuda
# 无显示环境 (服务器/CI):
./run_webcam.sh --headless --save-frames out/
# 九点个人校准（按画面中的目标注视，自动保存 JSON）:
./run_webcam.sh --calibrate profiles/me.json
# 加载校准并输出 fixation/saccade/blink 事件:
./run_webcam.sh --calibration-file profiles/me.json --events-out events.json
# 关闭 FLAME 眼睛上的调试箭头，只看虹膜/瞳孔/眼睑:
./run_webcam.sh --flame --hide-gaze-arrows
# 调整视频流的人脸检测/跟踪容错（默认每 6 帧检测，最多跟踪 30 帧）:
./run_webcam.sh --face-detect-interval 6 --face-track-frames 30
```

视频流默认使用 dlib correlation tracker 桥接短暂的运动模糊、遮挡或单帧检测失败，
并按周期重新运行全图检测以校正漂移；完全离开画面后会自动清除旧跟踪并重新寻找人脸。
自拍镜像只作用于显示，模型始终使用未镜像的摄像头帧。

在 ROCm AMD 核显上，实时入口的 `auto` 默认使用 GPU，并启用占空比护栏，避免
摄像头长时间运行时连续 HIP kernel 触发驱动卡死。默认相邻 forward 间隔至少
100ms，约 8-10fps；可用 `GAZE_ROCM_SAFE_INTERVAL=0.20` 增加安全余量。程序会按
设备名称和显存容量区分核显与独显：稳定 AMD 独显默认不受护栏限制；如硬件识别
不准确，可用 `GAZE_ROCM_IGPU=1` 强制启用或 `GAZE_ROCM_IGPU=0` 强制关闭。

此外，启动时默认关闭 MIOpen 的 GCN 汇编卷积 kernel，避免 gfx1035 通过 ISA
兼容模式运行数百次后触发 page fault；这仍是 GPU 推理，只是卷积速度会降低。
稳定 AMD 独显可用 `GAZE_ROCM_SAFE_KERNELS=0` 恢复汇编 kernel。

ROCm GPU 推理默认使用 fp16 autocast：GazeTR 的计算几乎全在 ResNet18 backbone，
半精度实测单张约 1.3x、批量约 1.7x，角度漂移 <0.1°，同时缩短每帧 GPU 忙碌时间、
降低核显持续负载。可用 `GAZE_ROCM_FP16=0` 回退 fp32；CPU/NVIDIA 路径不受影响。
注意实时窗口的 fps 仍由上面的占空比护栏决定（约 10fps），fp16 主要让 GPU 更快
空出来（更稳），批量/纯 GPU 场景的吞吐提升更明显。

关于 GPU hang：经底层排查，旧版 torch 2.4.1+rocm6.1 捆绑的 rocBLAS 只有 gfx1030
等内核，必须用 `HSA_OVERRIDE_GFX_VERSION=10.3.0` 伪装指令集，背靠背满载会触发
`[gfxhub] page fault` → `MODE2 reset`。后来实测了 AMD **TheRock** 的原生 gfx1035
构建（`gfx1035` 原生 rocBLAS/MIOpen、无需 override、数值正确、背靠背余量约 6x）：
它消除了“坏地址”故障，但**持续 GPU 满载最终仍会挂死**（`Queue preemption failed →
MODE2 reset`，与 kernel 集/精度无关，仅成倍延缓）——即这是 Radeon 680M APU 在持续
满载下的固件/驱动级挂死，没有软件配置能根治。且每次故障后 GPU 会进入"无异常但慢
几倍"的劣化态直到重启。因此占空比护栏是必要缓解，不能为了跑满帧率去掉。实时入口
还会在 GPU 连续报错或健康探针发现前向远慢于基线时，自动切换到 CPU 继续会话。
详见 `AGENTS.md` 的 "GPU Hang 根因排查" 与 "原生 gfx1035 (TheRock) 实测与迁移"。

有窗口时，ROCm 实时入口还会默认把 HighGUI 的显示端切到软件光栅化，GazeTR
仍然使用 GPU；如确认桌面/驱动稳定，可用 `GAZE_ROCM_SOFTWARE_DISPLAY=0` 恢复
原显示路径。

`--flame` 会让 pyrender 的 OpenGL 与 ROCm 共用核显，此模式仍自动把 GazeTR
放到 CPU；不带 `--flame` 的上述命令则使用 GPU 加速。若明确要 GPU，可写
`DEVICE=cuda ./run_webcam.sh --calibration-file profiles/me.json`。

交互控制：`q`/`ESC` 退出，`s` 保存快照。逐帧执行 人脸检测(降采样, ~30ms) →
ETH-XGaze 对齐 → GazeTR 推理（CPU 下本机实测约 **10fps**，受摄像头帧率上限约束）；
加 `--flame` 后 FLAME 头部离屏渲染并按 `--head-fps`（默认 8）节流刷新，
与标注后的画面合成到**同一个** `GazeFlame` 窗口，不拖慢主画面。

加上 `--assistive` 后会增加 `ASSISTIVE CONTROL CONCEPT` 面板：它把归一化视点映射到
抽象驾驶舱区域（SYSTEM STATUS、ATTITUDE、NAVIGATION、CONTROL AREA），结合注视时长
和跟踪置信度显示 `MONITORING`、`INTENT DETECTED` 或 `ASSIST READY`。该层只表达
“驾驶员状态感知 → 意图识别 → 辅助介入准备”的软件闭环，不模拟真实机械臂、飞行控制器
或物理执行器；因此适合作为“机械臂辅助控制飞行模拟器”课题的概念验证界面。使用
`--flame --assistive` 可同时看到输入图、FLAME 头部和辅助状态三类证据；使用视频文件
作为输入时不需要连接摄像头。

视点十字准星会先把 ETH-XGaze 标准化人脸坐标还原到原始相机坐标，再估算屏幕位置；加载
`--calibration-file profiles/me.json` 后会使用九点个人校准得到的坐标，标记精度更高。
校准只影响屏幕落点，不会改写 FLAME 的真实视线向量；每个目标点取稳定阶段的中位数，
可减少眨眼、跟踪抖动对 profile 的影响。RMSE 大于 0.15 的校准文件会被忽略，需重新校准。
新校准同时检查留一目标验证误差、角度覆盖与映射增益；失败不会覆盖原文件。实时窗口按
`c` 可重新校准。demo 请用 `./run_demo.sh --camera 0 --calibrate profiles/demo_user.json`
在实际演示窗口完成校准，并保持相同窗口大小和坐姿，见 [demo 校准说明](docs/DEMO.md)。
旧版 profile 使用的是未补偿头姿的角度坐标，首次升级后请重新运行一次 `--calibrate`。

### 实时增强

- `common.temporal.GazeTemporalState` 先还原相机系视线再用自适应 One Euro 滤波，默认
  cutoff=0.5Hz、beta=1.2；闭眼/低质量帧不更新，超过 12° 的单帧跳变须下一帧确认。
  `TemporalEstimate.camera_gaze` 是完整滤波后的相机向量，勿再次乘归一化旋转。时域状态
  同时计算角速度并输出 `face_found/face_lost`、`blink_start/blink_end`、`fixation_start/fixation_end`
  和 `saccade_start/saccade_end` 事件。事件可用 `--events-out` 保存为 JSON。
- `common.face` 从同一组 68 点关键点提供 EAR、solvePnP 头部轴角和重投影误差；
  FLAME 的头部、左右眼、上下眼睑是分层节点，头部转动不会再被误当成眼球转动。
- `--calibrate` 是九点定时校准，会保存一个从相机坐标角度到屏幕坐标的二维仿射 profile；
  用 `--calibration-file` 加载后，校准结果只用于屏幕落点，滤波后的真实视线仍用于 FLAME。
- `FlameMirror.render(gaze, head_pose, blink)` 复用原有场景和缓冲，只更新头部/眼睛/眼睑
  节点，加入左右眼会聚、虹膜、瞳孔高光和眨眼遮挡；`--hide-gaze-arrows` 可隐藏辅助箭头。

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
