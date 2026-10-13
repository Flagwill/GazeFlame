# GazeFlame 结题 / 答辩演示

本 demo 根据桌面《GazeFlame_结题报告》设计，特别对应报告的 2.1、4.4、5.1–5.4 和
6.3 节。提供**真实 webcam 全链路**、**真实视频逐帧推理**和**真实单图 + 合成状态回放**
三种入口，不需要安装新的 GUI 框架或更改现有模型版本。

## 1. 推荐：webcam 实时全链路

```bash
./run_demo.sh --camera 0
# 普通窗口，便于录屏：
./run_demo.sh --camera 0 --windowed
# 在实际演示面板内做个人校准（完成后继续实时演示）：
./run_demo.sh --camera 0 --calibrate profiles/demo_user.json
./run_demo.sh --camera 0 --calibration-file profiles/demo_user.json
# 录制整个演示面板：
./run_demo.sh --camera 0 --save-video demo_outputs/webcam.mp4
```

默认开启 **FLAME + 辅助状态**，同一窗口包含：

1. **驾驶员输入**：摄像头真实帧，复用 `draw_overlay` 的人脸框、关键点、视线角度。
2. **网络输入**：当前帧实际送入 GazeTR 的 ETH-XGaze 224×224 对齐裁剪，不是示意图。
3. **三维头部**：FLAME 头部、双眼朝向、眨眼及头眼解耦，场景仍只创建一次。
4. **辅助控制概念**：四个驾驶舱关注区域、目标、意图、置信度和当前区域注视时长。
5. **数值与事件证据**：相机系三维视线向量、眼动速度、原始/One Euro 滤波角度曲线、
   注视/扫视/眨眼/丢脸及 `assistive_` 状态事件。

实际数据流为：

```text
webcam RGB 帧
  → common.face: dlib 检测 / 跟踪 / 68点 / solvePnP / ETH-XGaze 对齐
  → gazetr.inference.predict_gaze: GazeTR 每帧真实推理
  → common.geometry + temporal: 先还原相机系，再 One Euro / 跳变检查 / 眼动事件
  → common.geometry: 转换 FLAME 坐标
  ├→ common.flame_mirror: 头部 / 双眼 / 眼睑实时渲染
  └→ 个人校准或未校准回退屏幕映射
      → common.assistive: 驾驶舱目标 / 意图 / 辅助准备
  → common.demo.DemoDashboard: 同一演示窗口
```

webcam 模式复用 `run_webcam.py`，没有另写一条简化的推理流水线。
`--demo` 也可以直接给旧入口：`./run_webcam.sh --demo --flame`。
其他已有参数如 `--camera-fov`、`--head-fps`、`--hide-gaze-arrows`、`--no-mirror`、
`--face-detect-interval`、`--events-out` 均可传给实时 demo。

### 现场讲解建议（约 1 分钟）

- 正脸进入画面：说明原图、对齐裁剪、3D 头部对应同一帧推理；演示转头和眨眼。
- 稳定看向某个区域：指出 `MONITORING → INTENT DETECTED → ASSIST READY`，
  以及 0.35s / 0.80s 阈值。
- 看向另一区域：解释注视证据重置，不能沿用上一目标的停留时间。
- 闭眼或离开画面：说明 `PAUSED` / `NO TRACKING`，不能在感知失效时宣称辅助准备。
- 展示角度曲线和事件日志，结束时强调**没有发送任何控制指令**。

**未校准落点只是线性回退估计，可能无法精确命中希望注视的区域。** 请提前个人校准，
不要把四个宽泛区域表述为真实座舱精确标定结果。校准仅影响屏幕视点，不修改送入 FLAME
的物理视线向量。若现场落点不稳定，用下述可重复回放展示规则，而不是伪造实时识别结果。

### 已校准但视点抖动或偏移

实时模式按 `c` 可直接在当前 demo 窗口重新做九点校准。加载过 profile 时，合格的新结果
保存回该路径；没有指定路径时保存到 `profiles/demo_user.json`。原文件在校准通过前保留。
保持演示时的摄像头位置、坐姿、窗口大小和全屏/普通窗口模式；普通 webcam 窗口与 demo
面板比例不同，旧窗口的校准不能保证适合新面板。注视依次出现的红色目标，不要追随估计准星。

每个目标默认等待 0.7s，再收集至少 1.5s、10 个有效样本。眨眼、丢脸、低质量跟踪或扫视
会重新开始当前目标的稳定采样；离散程度过大时重复该目标。拟合使用去离群后的中位数，并
检查留一目标验证误差、二维角度覆盖和映射增益，避免“拟合分数还可以，但微小抖动被放大”
的 profile。拒绝后按 `c` 重试；`CALIBRATION READY` 才代表新结果已保存。已有 v2 相机
角度 profile 仍可加载，新保存的 v3 增加质量诊断信息。

视线现在先还原到**相机坐标**再滤波，避免对齐坐标系逐帧变化重新引入抖动；闭眼和低质量帧
不进入视线滤波器。超过 12° 的单帧跳变需下一帧确认，会增加约一帧响应时间；持续转移仍会
更新，长时间丢失后重新初始化，不把上一目标拖进新目标。滤波参数可在实时入口透传：

```bash
# 默认：静止截止频率 0.5Hz、运动响应 beta=1.2
./run_demo.sh --camera 0 --calibration-file profiles/demo_user.json
# 需要更强的静止降噪时，可降低 cutoff；小幅移动的响应也会变慢：
./run_demo.sh --camera 0 --calibration-file profiles/demo_user.json --gaze-min-cutoff 0.35
```

这些检查改善信号稳定性与个人映射质量，不等于重新训练了 GazeTR，也不提供真实被试的
屏幕误差保证。画面中的 confidence 是几何/跟踪质量代理，不是网络预测准确率。

## 2. 没有摄像头：真实视频逐帧推理

```bash
./run_demo.sh --video /path/to/driver.mp4 --windowed
./run_demo.sh --video /path/to/driver.mp4 --headless \
  --output demo_outputs/video --save-video demo_outputs/video.mp4
```

视频中的每一帧仍经真实 dlib / GazeTR 推理，数据标签为 `VIDEO / REAL GazeTR INFERENCE`。
注视时间与事件时间使用**视频帧时间**，而不是解码/保存图片的速度；视频文件帧率不可用时
回退到 30fps。有窗口的 demo 按视频时间节奏播放，推理慢于源视频时则慢速播放、不跳帧。

## 3. 无外置硬件：报告场景回放（约 27 秒）

```bash
./run_demo.sh
./run_demo.sh --windowed
./run_demo.sh --headless --save-video demo_outputs/report_demo.mp4
# 更换真实单图开场：
./run_demo.sh --image /path/to/face.jpg
```

- 前 3 秒：对仓库 `test.jpg` 运行一次**真实人脸对齐 + GazeTR 推理 + FLAME 渲染**。
  显示真实输入和对齐裁剪，辅助状态不累积注视时间。单张图不能凭空产生时间证据。
- 后 24 秒：明确标注为 `SYNTHETIC REPLAY` 的确定性眼动输入。依次展示 ATTITUDE、
  SYSTEM STATUS、NAVIGATION、CONTROL AREA；随后展示眨眼、恢复、低置信度、人脸丢失、
  重新跟踪和区域外注视。
- 合成输入只驱动真实公共时域滤波器、事件检测器、辅助状态机和 FLAME 渲染器，
  **不是 GazeTR 从图片预测出的动态视线，不代表被试实验、模型精度或控制性能**。
- `SPACE` 暂停/继续，`s` 保存截图，`q` / `ESC` 退出。暂停期间不累计回放时间证据。
  webcam / 视频模式支持 `s`、`q`、`ESC`；webcam 额外支持 `c` 校准，不支持暂停实时感知。

回放不循环，完成后正常退出；方便答辩和自动化生成素材。

## 4. 无 X11 / GL：只演示感知与辅助概念

```bash
# 真实单图推理 + 2D 合成回放（无需 GL）：
./run_demo.sh --no-flame --headless --output demo_outputs/2d
# 完全不加载神经网络 / 不访问摄像头的规则验证：
./run_demo.sh --skip-image --no-flame --headless --output demo_outputs/rules
# 无 GL 的真实视频推理：
./run_demo.sh --video driver.mp4 --no-flame --headless --output demo_outputs/video2d
# 有显示服务的离屏全链路验证：
DISPLAY=:1 ./run_demo.sh --camera 0 --headless --max-frames 60 \
  --output demo_outputs/live_check
```

`--headless` 只关闭 OpenCV 窗口，**不会消除 pyrender 对 OpenGL 的要求**。
本机可用 `DISPLAY=:1`；如果没有可用 GL，请显式 `--no-flame`。
默认 CPU，以免本机 ROCm 核显 HIP 推理与持续 GL 共载。
显式 `DEVICE=cuda` 时仍经原有设备保护策略；不会移除 GPU 占空比护栏或绕过
`flame_realtime_device` 的 CPU 回退。

## 5. 输出、参数与验证

默认输出目录为仓库的 `demo_outputs/`（gitignore，可用 `--output` 更换）：

| 文件 | 内容 |
| --- | --- |
| `events.json` | 眼动与辅助状态事件；回放的每条事件注明真实单图或合成数据来源 |
| `summary.json` | 仅回放模式：真实开场推理结果、合成来源声明、帧数、实际出现的状态 |
| `stage_*.png` | 仅回放模式：各讲解阶段的截图 |
| `state_*.png` | 仅回放模式：状态 / 目标变化时截图 |
| `last_frame.png` | 仅回放模式：最后显示的面板 |
| `frames/` | 实时/视频 headless 模式：原有标注帧、`head/` 渲染帧、`assistive/` 完整 demo 面板 |
| 自定义 `.mp4` | 使用 `--save-video PATH` 导出的 1440×900 演示面板 |

导出使用 OpenCV 的 `mp4v` 编码，无音轨；实时摄像头视频以 `--fps`（默认 15）作为固定输出
帧率，不保证录制视频时长等于墙钟时长。JSON 的 webcam 事件时间保留实际感知时间。
PNG 逐帧导出会降低处理吞吐，不能用它测量实时 GUI 帧率。

```bash
./run_demo.sh --help
# 使用已有环境以外的明确解释器：
GAZE_PYTHON=/path/to/python ./run_demo.sh --help
# 单元测试无需摄像头或 GL：
conda activate gaze_flame_rocm
python -m unittest discover -s tests -v
```

新增测试覆盖：所有四个区域的辅助准备、扫视/眨眼/丢脸事件、低置信度拒绝辅助、
跨区域注视证据重置、恢复后重新积累、无效数值拒绝、固定面板尺寸和回放可重复性。
`tests/test_tracking_accuracy.py` 另外覆盖坐标系变化时的固定视线、8/15/30fps 降噪与
目标转移延迟、单帧尖峰、闭眼保持、丢失重建、校准离群样本、退化映射和旧 profile 兼容。

## 安全与研究边界

本 demo 实现的是**驾驶员状态感知 → 注视意图识别 → 辅助介入准备**的软件闭环。
`ASSIST READY` 是候选准备提示，不是控制命令、自动驾驶决策或已经完成的机械臂动作。
项目没有接入真实机械臂、飞控、飞行动力学或执行器，也没有真实驾驶安全验证。
