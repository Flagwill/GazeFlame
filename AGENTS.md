# AGENTS.md — GazeFlame 仓库记忆

## 项目概述

GazeTR（Transformer 视线估计）× FLAME_PyTorch（3D 头部参数模型）的合并流水线：

```
人脸图片 → GazeTR (gazetr/inference.py) → Pitch/Yaw (弧度)
        → run_gaze_to_flame.py 坐标换算 (common/geometry.py) → 3D 视线向量
        → FLAME (common/flame_mirror.FlameMirror) + pyrender 离屏渲染
        → 标注输入图 + 3D 头部合成到同一个 cv2 窗口 (common/viz + compose_with_head)
```

两个子项目用 `gazetr/`、`flame/` 子目录隔离（顶层有重名的 model/、config.py 等）。
共享工具放顶层 `common/`：几何换算唯一数据源 `geometry.py`、2D 视线标注唯一实现
`viz.py`（draw_overlay，图片/摄像头共用）、FLAME 离屏渲染 + 单窗口合成
`flame_mirror.py`（FlameMirror / compose_with_head）、软件-only 辅助控制概念层
`assistive.py`（视线目标 / 意图状态 / 辅助面板）；编排脚本从仓库根目录导入。
人脸检测/对齐也放在 `common/face.py`（dlib + ETH-XGaze 归一化，GazeTR 训练同款），
68 点模型在 `common/models/shape_predictor_68_face_landmarks.dat`（~96MB，gitignore，
用 `python -m common.download_dlib_model` 下载，断点续传可重跑）。

## 环境

- conda 环境名：`gaze_flame_rocm`，Python **3.10** + AMD **TheRock 原生 gfx1035**
  torch（rocBLAS/MIOpen 自带 gfx1035 kernel，无需 `HSA_OVERRIDE`；2026-09-11 从旧的
  Python 3.8 + torch 2.4.1+rocm6.1 环境迁移而来，旧 `gaze_flame` 环境已删除）
- 关键版本锁定（勿随意升级）：numpy==1.23.5（trimesh 3.6.18 需要 np.bool 等旧别名）、
  pyrender==0.1.39、trimesh==3.6.18；这套老栈在 Python 3.10 上同样可用（已验证）
- dlib 19.24.2（conda-forge）：**conda 安装 dlib 会把 numpy 升到 1.24+**，
  `environment.yml` 已把 pip 的 numpy==1.23.5 排在 dlib 之后；若 import trimesh 报
  `AttributeError: module 'numpy' has no attribute 'bool'`，执行
  `pip install --force-reinstall numpy==1.23.5 --no-deps` 降回（已知坑）
- 创建/更新环境：`conda env create -f environment.yml`（含 TheRock nightly 索引；
  环境更新后记得 numpy 降级 + 下载 dlib 模型）

## 运行方式

```bash
# 一键脚本 (自动定位 conda + gaze_flame_rocm 环境, 无需手动 activate; 默认 CPU)
./run.sh                                  # 默认示例图片 test.jpg (统一单窗口)
./run.sh --image /path/to/face.jpg        # 自定义图片
./run.sh --skip-viewer                    # 只算角度, 不显示窗口
./run.sh --headless --save-image out.png  # 无 X11: 把合成图存 PNG (自动化推荐)
DEVICE=cuda ./run.sh                      # 有 GPU 时走 cuda (无 GPU 自动回退 cpu)

# 或手动方式:
conda activate gaze_flame_rocm
python run_gaze_to_flame.py --image gazetr/images/inference/test.jpg --device cpu

# 仅 GazeTR 推理
cd gazetr && python inference.py --model model/GazeTR-H-ETH.pt --image images/inference/test.jpg --device cpu
# 注意: --popup 默认关闭(会阻塞)。需要弹窗时显式加 --popup。

# 仅 FLAME 渲染（手动方向）
cd flame && python gaze_viewer.py --gaze-x 0 --gaze-y 0.2 --gaze-z 1
# 无窗口渲染: 加 --save-image out.png --headless

# 实时摄像头 / 视频流
./run_webcam.sh --flame                       # 摄像头 + FLAME 头部镜像 (统一单窗口)
./run_webcam.sh --video clip.mp4              # 视频文件
./run_webcam.sh --video clip.mp4 --assistive  # 无摄像头的辅助控制概念验证
./run_webcam.sh --video clip.mp4 --flame --assistive --save-frames out/
./run_webcam.sh --headless --save-frames out/ # 无窗口, 存帧 (自动化)
```

## 已验证事实（2026-09-08，CPU）

- 全链路跑通：test.jpg（启用人脸对齐，2026-09-09 后为默认）→
  Pitch=0.0434 rad, Yaw=-0.3421 rad → gaze 向量 (-0.0409, 0.3355, -0.9412)
  （旧基线 `--no-align` 直接缩放：Pitch=-0.2064, Yaw=-0.7247 → (0.1534, 0.6629, -0.7328)；
  test.jpg 本身已是归一化裁剪，二次对齐会改变输入分布，两者均有效）
- 任意照片端到端验证：把 test.jpg 人脸旋转 12° 嵌入 640×900 背景图后推理，
  对齐后得到 Pitch=-0.0156, Yaw=-0.3502，与直接裁剪人脸的对齐结果一致
  （不对齐时漂移到 Yaw=-0.2149）
- CPU 下 GazeTR 单张推理约 **数秒**（2026-09-08 重构后；此前 1–2 分钟是因
  inference.py 默认弹 tkinter 窗口阻塞，已改为 --popup 显式启用）
- `pyrender.Viewer` 需要 **X11 显示环境**（弹出 GLFW 窗口，阻塞至手动关闭）；
  无 X11/后台会话中会静默退出 → 自动化请用 `--headless --save-image`
- 实时 webcam 已实测（2026-09-09 建，2026-09-10 夜修订）：本机 `/dev/video0`
  存在、`DISPLAY=:1` 可用；**裸摄像头读取 30fps（~33ms/帧，非旧测的 ~100ms），
  旧"上限 ~10fps"作废**——当时 10fps 的真凶是 FLAME 渲染：每帧重建整个 pyrender
  场景（build_scene ~44ms + gl_render ~85ms，其中 ~28ms 是无人使用的深度回读），
  8Hz 更新时每次阻塞主循环 ~140ms。修复后 `--flame` 实测 ~14.5–15fps（30s 连续，
  含 face=1 渲染段，无尖峰）：FlameMirror 场景只建一次、render 仅对两个
  gaze-root 节点 set_pose（480² 下 140ms→**5.4ms**，与旧实现逐像素比对差
  0.02%，仅在眼球边缘），并跳过深度缓冲 glReadPixels。剩余下限：dlib
  HOG@384 ~34ms + torch CPU 前向 12ms（默认 8 线程实测最优，勿减）+ GUI ~10ms
  ≈ 理论 20fps
- dlib HOG 检测开销随面积增长：640x480 全图检测 ~458ms，降采样到 384 后仅 ~25ms
  （`common/face.py` 的 `DETECT_MAX_DIM=384` 已自动处理大图，勿改大）
- pyrender 离屏渲染的 OpenGL 上下文**绑定创建它的线程**，不能跨线程渲染
  → `run_webcam.py` 的 FLAME 镜像在主循环内按 `--head-fps`(默认 8) 节流，不用线程
- opencv-python 5.0 已移除 `cv2.CascadeClassifier`/objdetect，勿用 Haar 方案
- 可视化已统一为单窗口（2026-09-10，CPU 验证）：
  `./run.sh --headless --save-image` 产出左=标注输入图/右=FLAME头 的合成 PNG（exit 0）；
  `run_webcam.py --flame --headless --save-frames` 逐帧存标注图 + head/ 节流帧，均正常。
  注意：pyrender **OffscreenRenderer 仍需可用 GL 上下文**（本机 `DISPLAY=:1`）；
  `DISPLAY=`（空）会让其创建失败并报 Pyglet NoSuchDisplayException——这不是代码问题。
- **AMD GPU (ROCm, TheRock 原生 gfx1035，2026-09-11 迁移）**：
  本机 Radeon 680M = gfx1035，现用 TheRock 原生构建（`amd-torch-device-gfx1035`，
  当前 torch 2.14.0+rocm10.2.0a / HIP 7.16；见 environment.yml 与本文「原生 gfx1035
  (TheRock) 实测」）。**无需 `HSA_OVERRIDE`**（旧式官方 ROCm 才需，显式设
  `GAZE_ROCM_GFX_OVERRIDE=10.3.0` 复现）。前向 batch=1 约 4.8ms、batch=16 约 36ms。
  **坑（关键）**：核显与显示共享内存，**背靠背满载** forward 到 ~2000 次(b1) 必现
  GPU Hang + amdgpu MODE2 reset（旧 build 是 ~300 且伴 page fault；`HSA_ENABLE_SDMA=0`、
  禁用 fused SDPA 均无效；挂死后重启才恢复）。
  **对策 = 占空比护栏**：`common/gpu.py:rocm_igpu_guard(device, elapsed)` 把相邻两次
  GPU forward 的间隔补足到 `ROCM_SAFE_INTERVAL`(默认 100ms)，实测可稳定数千帧。
  只有走 ISA 伪装（`GAZE_ROCM_GFX_OVERRIDE`）时才默认设
  `MIOPEN_DEBUG_GCN_ASM_KERNELS=0`；native gfx1035 保留 ASM kernel（更快，非 ASM
  只把挂死从 ~3650 延到 ~5850，治标不治本）。稳定 AMD 独显可用
  `GAZE_ROCM_SAFE_MODE=0` 关闭占空比护栏。快速视频回放 / 大批量文件夹仍会被限速到
  约 6fps 以保安全。单张图 run.sh 仍默认 cpu（GPU 上下文初始化多花 ~1.5s，单图不划算；
  DEVICE=cuda 显式开启则走 GPU）。
  **第二个坑（2026-09-10 夜修复）：HIP 与持续 OpenGL 渲染互斥**。`run_webcam
  --flame` 有人脸时 pyrender 每帧做 GL 渲染（光栅化在同一块核显的 3D 引擎上），
  护栏留出的"空闲"被 GL 占满 → 与 HIP 共载，实测 21–55s（55–300 帧）必现
  Memory access fault，与 FLAME 形变放哪无关、`LIBGL_ALWAYS_SOFTWARE=1` 也救不了。
  **对策 = `common/gpu.py:flame_realtime_device(requested, flame_enabled)`**：
  ROCm 构建 + `--flame` 时推理自动回退 cpu（GazeTR CPU 前向 12ms 本就满足
  10fps，GPU 在此场景无收益；GL 回到纯图形使用的稳定老路），打印提示可懂；
  无 --flame 的实时/批量仍走 GPU+护栏。已验证修复后 1500 帧/155s 连续 FLAME
  渲染零故障。另外 `flame_render_device()` 在强制用 GPU 跑 FLAME 时把形变挪回
  cpu（纵深防御）。挂死过 1–2 次后 GPU 可能进入"无 fault 的静默 NaN 劣化态"
  （fix 前的 NaN 即此），再跑几分钟/复位后会恢复——遇到莫名 NaN 先怀疑这个。
- **推理提速（2026-09-10，CPU 实测）**：`run_gaze_to_flame.py` 默认改为**进程内**
  加载 GazeTR（同 run_webcam 的 sys.path 模式，先 `import model` 再让
  flame_mirror 把 flame/ 加进 sys.path，避免与 `flame/model/` 数据目录撞名）。
  `--skip-viewer` 纯推理路径 3.40s → **2.06s**（省一次解释器启动 + torch/dlib
  二次导入）。旧子进程+JSON 临时文件链路保留为 `--subprocess` 回退选项。
  run_webcam 增加 load 后 3 次 dummy forward 预热（GPU 首帧 455ms→24ms 级别；
  CPU 首帧亦稳定）。当前分段耗时：GazeTR 阶段 ~1.8s（torch import 1.0s +
  dlib 模型 0.65s + 其余），FLAME 阶段 ~1.0s；模型前向本身仅 13ms（300 帧
  持续压测 median 13.0 / max 15.1ms，无劣化）。
- **fp16 autocast GPU 推理（2026-09-11，ROCm 默认开启）**：GazeTR 的耗时几乎全在
  ResNet18 backbone（b1 前向 ~8ms 中 transformer encoder 仅 ~2ms；逐层 profiling
  见下），fp16 autocast 实测 b1 **1.3x** / b16 **1.7x**，ETH-XGaze 样例角度漂移
  <0.1°（pitch/yaw 各 ~0.001 rad）。默认在 ROCm 构建开启，`GAZE_ROCM_FP16=0`
  关闭；CPU/NVIDIA 不受影响。**重要认知**：核显实时路径的墙钟 fps 仍被 100ms
  占空比护栏钉在 ~10fps，fp16 的收益是**每帧 GPU 忙碌时间缩短**（b1 8→5ms、
  b16 100→60ms，持续负载更低、更远离 Hang）以及在护栏不生效场景（独显 /
  `GAZE_ROCM_SAFE_MODE=0` / 纯 GPU-bound 批处理）的吞吐提升，而不是实时 fps。
  切勿因此调大 `--batch_size`：b32+ 实测更慢，且无护栏背靠背大 batch 已复现
  `Memory access fault`（护栏必须保留）。
- **推理健壮性（2026-09-11）**：`predict_gaze` / 批量推理拦截非有限输出
  （静默 NaN 劣化态按丢脸 / 跳过该图处理，不再把 NaN 传进 UI）；`run_webcam`
  若 GPU 连续 3 帧抛 HIP/CUDA 运行时错误，会自动把模型重载到 CPU 继续会话
  （死掉的 HIP 上下文无法进程内恢复；`common.gpu.is_gpu_runtime_error` 判别）。

## GPU Hang 根因排查（2026-09-11，旧 build ROCm 6.1 / gfx1035）

用户怀疑"GPU hang 不正常"。旧环境排查结论：**这是 ROCm 6.1 对 Radeon 680M
(gfx1035) 不支持的后果，不是本项目代码的 bug，也无法靠改 Python 代码根治。**
（后续用原生 TheRock 复测，发现还有更深一层原因，见下节。）

证据链：
- 内核 7.2.3-arch，amdgpu 驱动；torch 2.4.1+rocm6.1 是 Python 3.8 可用的最后一版。
- 随 torch 捆绑的 rocBLAS Tensile kernel 只有 gfx1030 / gfx1100 / gfx900 / gfx906 /
  gfx908 / gfx90a / gfx942，**没有 gfx1035**；MIOpen 3.1.0 与 hipBLASLt 同样没有。
  只能 `HSA_OVERRIDE_GFX_VERSION=10.3.0` 把 gfx1030 预编译二进制跑在 gfx1035 上；
  `torch.cuda.get_device_properties(0).gcnArchName` 实际报 `gfx1030`。
- 复现：无间隔背靠背 forward，约 20s–5min 内必现 GPU 页错误；内核日志签名固定：
  ```
  amdgpu ...: [gfxhub] page fault (src_id:0 ring:24 ...)
  amdgpu ...: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
  amdgpu ...:   Faulty UTCL2 client ID: TCP (0x8)
  amdgpu ...:   PERMISSION_FAULTS: 0x3       # 访问了未映射的 GPU VA
  amdgpu ...: GPU reset begin!. Source: 4    # VM 层触发
  amdgpu ...: MODE2 reset
  amdgpu ...: [drm] device wedged, but no recovery needed
  ```
- 内存无泄漏（allocated/reserved 恒定），温度 ~80°C 不是热保护；故障地址在进程
  VA 空间里跳变 → gfx1030 内核在 gfx1035 APU 上算出了错误的地址（APU 的 GTT/
  carve-out 地址布局与 dGPU 不同）。
- 故障是**概率性**的：空闲间隔越长概率越低，但 20ms 空闲仍约 5min 复现；
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 在本 build 直接
  "not supported on this platform" 被忽略。所以 100ms 占空比护栏是**降低概率**
  而非根治（这也解释了为什么它当初有效、却不能宣称"安全"）。
- **关键副作用：每次 fault 后 GPU 进入静默劣化态**（实测 batch16 前向
  105ms→705ms、batch1 ~8→20ms），不会自愈，需复位/重启。所以宁可多留空闲、
  避免 fault，也不要为了 fps 去掉护栏。

处置：
- **不要移除 `common.gpu.rocm_igpu_guard` 占空比护栏**，它是目前最有效的缓解；
  想更保守可调大 `GAZE_ROCM_SAFE_INTERVAL`。
- `run_webcam` 在 GPU 连续 3 帧报 HIP/CUDA 错误、或健康探针连续两次发现前向
  时间 > `max(6×基线, 基线+100ms)` 时，会把模型重载到 CPU 继续会话
  （`fallback_to_cpu`）。硬 fault（SIGABRT）进程仍会被杀，完全兜底需要外层
  supervisor 重启。
- 根治路径（优先级）：① 换 ROCm 官方支持的 GPU；② 带 gfx1035 原生支持的
  ROCm 构建（已实测 = TheRock，见下节：能去掉 override/坏地址故障，但**不能**消除
  持续满载挂死，仍需护栏）；③ 纯 CPU（GazeTR CPU 前向 ~12ms，实时 10fps 足够）；
  ④ 维持现状：护栏 + fp16 + 自动降级。

## 原生 gfx1035 (TheRock) 实测与迁移（2026-09-11，最终结论）

官方 ROCm（直到文档中的 10.0）始终没有 gfx1035 的 rocBLAS/MIOpen kernel；但 AMD
新的 **TheRock** 构建按 gfx 架构发布原生 wheel，gfx1035 在支持列表里。**项目已迁移
到该构建**（conda 环境 `gaze_flame_rocm`，Python 3.10；见 environment.yml）：

```bash
pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1035]"
```

实测（torch 2.14.0+rocm10.2.0a，HIP 报 7.16）：`gcnArchName=gfx1035`（**无需**
`HSA_OVERRIDE_GFX_VERSION`），`get_arch_list()` 含 gfx1035，捆绑 rocBLAS 的 Tensile
**只有 gfx1035**、MIOpen 有 gfx1035 kpack。数值正确：fp32 GPU vs CPU 最大差 2.7e-6、
fp16 8.7e-4；b1 ~4.8ms、b16 ~36ms。

**但它不能解决持续满载挂死。** 背靠背（无护栏）压测阈值：

| 工作负载（native gfx1035） | 挂死迭代 | 故障 |
|---|---|---|
| 整个 GazeTR 模型 b1 fp32 | ~2000 | 无 page fault，`Queue preemption failed → MODE2 reset` |
| 仅 ResNet18 卷积栈 fp32 + MIOpen ASM | ~3650 | 同上 |
| 仅卷积栈 fp32 + `MIOPEN_DEBUG_GCN_ASM_KERNELS=0` | ~5850 | 同上 |
| 仅卷积栈 fp16 + 非 ASM | ~10350 | 同上 |

对比旧 build（rocBLAS gfx1030 伪装）~300 次且伴 `[gfxhub] page fault`：native 消除了
ISA 不匹配与“坏地址”故障，余量约 6x；但换 kernel 集（非 ASM）/降精度只**成倍延缓**
挂死，无法根除，触发点是 MIOpen 卷积栈。结论：**这是 Radeon 680M APU 在持续 GPU
满载下的固件/驱动级挂死，任何 ROCm 版本/内核配置都只是降低概率**。挂死后 GPU 进入
静默劣化态（b1 5ms→20ms），MODE2 复位不恢复，需重启。

影响：**占空比护栏继续保留**（native 也救不了）。native 把背靠背余量从旧 build 的
~300 拉到 ~2000（~6x），再叠加非 ASM + fp16 可到 ~10350（~2.8x）；理论上可把
`GAZE_ROCM_SAFE_INTERVAL` 调小于 100ms 换取更高实时 fps，但默认仍用 100ms 保守值。
迁移已完成：Python 3.10 + TheRock，numpy 1.23.5 / trimesh 3.6.18 / pyrender 0.1.39
老 pins 在 cp310 可用；旧 `gaze_flame`（Py3.8）与 `.venv_rocm1035` 实验环境已删除。
复现脚本在 `~/.cache/gazeflame_gfx1035/`（不在仓库，已指向新环境 `gaze_flame_rocm`）。

## Agent 注意事项

- 本机 GPU：Radeon 680M 核显（gfx1035，TheRock 原生 torch 2.14+rocm10.2，`cuda` 即 ROCm）。
  可安全用于实时/批量推理，**但必须保留 `rocm_igpu_guard` 占空比护栏**（背靠背
  满载会 GPU Hang，见上文；TheRock 原生 gfx1035 也一样，别以为换个 build 就能去掉）；
  **`--flame` 实时模式必须经 `flame_realtime_device`
  回退 cpu**（HIP 与持续 GL 渲染共载必挂）；单张图 run.sh 默认仍 cpu（GPU 初始化不划算）
- 交互式窗口（`run_gaze_to_flame` 的 cv2 窗口 / `run_webcam` / `gaze_viewer` 的
  pyrender.Viewer）都是阻塞式 GUI：自动化验证请用 `--headless --save-image`（或
  `--save-frames`），可正常退出（退出码 0）；`--headless` 需配合保存路径。
  无显示环境若要离屏渲染，确保有可用 GL（本机 `DISPLAY=:1`）
- 设备选择统一走 `common/gpu.py`：`resolve_device("auto"/"cuda")` 用**真实 kernel
  探测**（ROCm 对不支持芯片 `is_available()` 会误报 True，kernel 才报错，勿改回）；
  `prepare_amd_gpu_env()` 必须在任何 `import torch` 之前调用（所有入口脚本头部已
  接好）：仅在显式设 `GAZE_ROCM_GFX_OVERRIDE=10.3.0`（旧式无 gfx1035 kernel 的官方
  ROCm）时才设 HSA override + 禁用 MIOpen 汇编 kernel；当前 native gfx1035 默认不设
  override，只抢注系统 libdrm——ROCm 版 torch 会连带 import 旧 libdrm 破坏 pyrender 的
  GLX（NoSuchConfigException），换回 CPU/CUDA 构建后此调用无副作用
- `run_gaze_to_flame.py`：GazeTR 推理默认**进程内**（sys.path 加 gazetr/ 后
  import model/inference，与 run_webcam 同款）；`--subprocess` 保留旧
  subprocess + JSON 临时文件链路作回退。FLAME 头部进程内 `FlameMirror` 离屏渲染，
  `compose_with_head` 合成单窗口。路径已做绝对化解析（勿改回相对）
- 统一可视化的公共件：几何 `common/geometry.py`、2D 标注 `common/viz.draw_overlay`
  （图片与摄像头共用，勿在 run_webcam/run_gaze_to_flame 里各写一份）、
  FLAME 渲染+合成 `common/flame_mirror.py`（`FlameMirror` 从 run_webcam 抽出，二者共用）
- `common/flame_mirror.py` 会把 `flame/` 加入 sys.path 并复用 `gaze_viewer` 的
  estimate_eye_centers/forward_to_axis_angle/make_cylinder_between 等函数，
  因此 `flame/gaze_viewer.py` 仍是单看 3D 的入口
- **FlameMirror 的场景只在 `__init__` 建一次，`render()` 仅 set_pose 一个
  head-root、两个 gaze-root 和四个眼睑节点 + 复用缓存的 GPU buffer**——实时
  `--flame` 的 fps 全靠这一点（旧实现每帧重建场景实测 140ms/帧 → 5.4ms），
  勿改回每帧 `build_scene`。头部网格冻结在 eye_pose=0（FLAME 眼姿只动插座内
  皮肤色眼球顶点，可见的虹膜/瞳孔/高光是独立的眼部网格，像素级比对差
  <0.03%）。`_install_color_only_readback`
  是对 **pyrender 0.1.39 内部**（`Renderer._main_fb/_main_fb_ms/_main_fb_dims/
  _read_main_framebuffer`）的实例级补丁，跳过无人用的深度 glReadPixels（~28ms/帧）；
  升级 pyrender 时若属性对不上会自动回退到原始路径（只是变慢，不会出错）
- run_webcam 的 `--head-size` 默认 480=摄像头高度（compose 1:1 显示，GL 回读
  开销随面积增长，开大只亏帧率）；无 --flame 的 GPU 路径被占空比护栏限速到
  ~8-10fps 属预期（见上文），ROCm 实时 `auto` 默认走 GPU 并启用 100ms 占空比
  护栏；可用 `GAZE_ROCM_SAFE_INTERVAL=0.20` 增加余量，稳定 AMD 独显可用
  `GAZE_ROCM_SAFE_MODE=0` 关闭护栏。`--flame` 仍按 `flame_realtime_device` 回退
  cpu，以避免 HIP 与持续 GL 渲染共载，不受该护栏影响。
- native gfx1035 默认保留 MIOpen GCN 汇编卷积 kernel（`MIOPEN_DEBUG_GCN_ASM_KERNELS`
  不设，更快）；只有旧式 ISA override 模式才默认禁用（`GAZE_ROCM_SAFE_KERNELS=0`
  关闭该禁用）。两种都仍需占空比护栏。
- ROCm GPU 推理默认走 fp16 autocast（`common.gpu.autocast_gpu`，约 1.3-1.7x，
  见上），`GAZE_ROCM_FP16=0` 可回退 fp32；仅 ROCm 构建生效，CPU/NVIDIA 不受影响。
  前向与 `run_webcam` 的 warmup 都走同一精度，避免首帧 kernel 重编译。
- ROCm 核显的实时 GUI 默认由 `prepare_rocm_display_env` 设置 HighGUI/Qt 使用
  XCB + 软件光栅化（`LIBGL_ALWAYS_SOFTWARE=1`），而 GazeTR 仍运行在 GPU；这是
  为隔离桌面合成与 HIP 负载。确认稳定桌面后可设 `GAZE_ROCM_SOFTWARE_DISPLAY=0`
  关闭；`--headless` 和 `--flame` 不套用此显示变量。
- 视线几何换算统一在 `common/geometry.py`；`gazetr/gtools.py` 保留上游实现
  （gazetr 内部使用），不要在别处再复制一份数学公式
- 人脸检测/对齐统一在 `common/face.py`（dlib 检测 + ETH-XGaze 6点归一化，
  复刻官方 demo.py 的 solvePnP + warpPerspective 参数，勿在 gazetr/flame 里另写一份）
- `gazetr/inference.py` 默认开启对齐（`--no-align` 关闭）；无 dlib 模型或检测
  不到人脸时会打印警告并回退直接缩放，不会中断流水线
- `flame/config.py` 是 `FLAMEConfig` dataclass，无模块级 parser 副作用；
  `gaze_viewer.py` 用 `add_flame_args + config_from_args` 构造配置

## 辅助控制概念验证层（2026-09-12）

为拉近项目与“机械臂辅助控制飞行模拟器”的关系，新增了一个**纯软件、概念验证级**
的驾驶员状态感知层。它只表达以下闭环，不模拟真实机械臂、飞行动力学、飞控或物理
执行器：

```
GazeTR / FLAME 视线与头姿感知
        → 抽象驾驶舱关注区域
        → 注视意图状态
        → 辅助介入准备状态
```

- 实现位置：`common/assistive.py`。唯一职责是把归一化视点映射到四个宽泛的概念区域
  （`SYSTEM STATUS`、`ATTITUDE`、`NAVIGATION`、`CONTROL AREA`），并根据跟踪置信度、
  视线速度、注视时长和眨眼状态输出可解释的辅助状态。
- 状态机默认阈值：置信度至少 `0.25`；同一区域注视达到 `0.35s` 后为
  `INTENT DETECTED`，达到 `0.80s` 后为 `ASSIST READY`。切换到新区域必须重新获得
  注视时间，不能沿用上一个区域的 fixation 证据。
- `run_webcam.py --assistive`：实时/视频模式增加 `ASSISTIVE CONTROL CONCEPT` 面板，
  复用 `common.temporal.GazeTemporalState` 的滤波、注视时长和置信度。使用视频文件
  （`--video`）时不需要摄像头或其他外置硬件；`--flame --assistive` 会显示输入图、
  FLAME 头部和辅助状态三类面板。
- `run_gaze_to_flame.py --assistive`：单图模式显示当前关注区域，但单张图没有时间证据，
  只显示 `MONITORING`，不能伪造 `INTENT DETECTED` 或 `ASSIST READY`。
- 视频模式的辅助状态事件以 `assistive_` 前缀写入现有 `--events-out` JSON；启用
  `--save-frames` 时，标注帧仍写入输出根目录，含辅助面板的合成帧写入 `assistive/`。
- 面板合成通过 `common.assistive.compose_with_assistive` 完成；FLAME 仍只由
  `common.flame_mirror` 负责，不能在辅助模块中复制或重建 FLAME 场景。
- 这是课题的“驾驶员状态感知 → 操作意图识别 → 辅助介入准备”软件闭环，不能在论文或
  演示中表述为已经完成真实机械臂控制。真实机械臂、飞行模拟器通信和执行器接口属于
  后续扩展边界。
- 测试：`tests/test_assistive.py` 覆盖状态机推进、无效跟踪拒绝辅助、面板尺寸稳定性。
  在 `gaze_flame_rocm` 环境中使用
  `python -m unittest discover -s tests -v` 验证通过（3 tests）。
- FLAME 组合图仍继承原有 pyrender/OpenGL 前置条件；没有可用 X11/GL 时，使用
  `--assistive` 且不加 `--flame` 的视频模式验证概念层，或只运行辅助面板单元测试，
  不要把 OpenGL 初始化失败误判为辅助状态机错误。

## 结题报告演示 Demo

- 新入口 `./run_demo.sh --camera 0` 默认 CPU + FLAME + 辅助状态，委托
  `run_webcam.main(argv)` 的完整真实推理流水线；`--video` 同样逐帧真实推理。
  原入口也可用 `./run_webcam.sh --demo --flame`。实时参数和个人校准可透传。
- `common/demo.py:DemoDashboard` 为统一 1440×900 演示面板，展示标注输入、实际网络
  对齐裁剪、FLAME、辅助状态、滤波曲线和事件；不加载模型、不创建 GL 场景。
- 无输入源的 `./run_demo.sh` 为约 27s 的报告回放：前3s单图真实推理（不累积注视），
  随后明确标注合成眼动输入。`ReportReplay` 仅验证公共时域/辅助规则，不代表模型精度。
  SPACE 暂停回放时不能累计注视时间。摄像头和视频模式没有合成输入。
- 视频事件/注视时间已改用源帧率的媒体时钟，避免快速离线处理或保存图片改变时间证据。
- `AssistiveIntentState` 使用当前区域的本地停留时间与时域注视证据的较小值；区域切换、
  眨眼、低置信度、丢脸、扫视均重新累计，不能只重置第一帧后继承旧区域的长注视。
- 三个启动脚本共用 `common/conda_env.sh` 选择解释器：优先 `GAZE_PYTHON`，其次已激活
  的 `CONDA_PREFIX`；回退解析 conda env list 时按绝对路径识别，不把名称后的 `*` / `+`
  标记当作路径（支持路径含空格）。回归测试在 `tests/test_launchers.py`。
- 使用说明在 `docs/DEMO.md`；默认输出 `demo_outputs/`（gitignore），回放保存阶段/状态
  PNG、events.json、summary.json；`--save-video PATH` 保存 MP4。实时 headless 保存帧。
  `--no-flame` 不需要 GL，`--skip-image --no-flame` 不需要 torch 权重或摄像头。
- 演示仍不发送任何机械臂、飞控、执行器命令，仍须保留 ROCm/GL CPU 回退及 GPU 护栏。
  相关测试在 `tests/test_demo.py` 和扩展的 `tests/test_assistive.py`。

## 实时视点稳定性与校准（2026-10-13）

- `GazeTemporalState` 内部先用当前 `normalization_rotation.T` 还原相机向量，再做
  One Euro（默认 min_cutoff=0.5Hz、beta=1.2，角度单位弧度）；消费
  `estimate.camera_gaze`，不要再乘一次当前 R.T，否则会重新引入头姿/对齐抖动。
  `estimate.gaze` 和 pitch/yaw 保留当前归一化系的兼容输出。
- 闭眼从首帧阻止测量入滤波；低几何/跟踪质量帧也不更新。12° 以上尖峰须后续相邻帧
  确认，持续移动仍跟随；长于 max_missing_seconds 的无有效测量期重置滤波与注视证据。
  `measurement_valid` 表示该帧是否真正进入滤波，不能仅用 face_count 采集校准样本。
- demo/webcam 的 `c` 在当前布局启动九点校准；默认 settle=0.7s、hold=1.5s、至少10帧。
  blink/低质量/扫视打断当前 hold；中位数去离群、留一目标验证、角度覆盖和增益检查通过
  才保存/使用新 profile，失败保留旧文件。v3 增加验证误差和采样诊断，v2 相机系兼容。
  校准目标相对当前窗口内容定义，需在实际 demo 布局和相同坐姿/窗口模式内校准。
- `tests/test_tracking_accuracy.py` 是确定性信号与映射回归，不是被试精度基准；不能把
  合成降噪结果表述为神经网络识别准确率提升。模型权重、训练归一化和 GPU 护栏未改变。
  `tests/test_webcam_calibration.py` 在 mock 摄像头/GUI 的真实入口验证保存只执行一次、
  `c` 复用已加载路径及拒绝结果不覆盖原文件。

## 仓库产物约定

- `profiles/*.json` 为本机/个人校准，`events.json`、`gaze_snapshot_*.png` 和
  `demo_outputs/` 为运行产物，均 gitignore；不要把个人 profile 当作通用示例提交。
  初次运行按 `profiles/README.md` 在实际 demo 中校准。
- 必需的 `gazetr/model/*.pt`、`flame/model/*` 及已有示例素材继续保留，不按临时产物清理。
