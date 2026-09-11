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
`flame_mirror.py`（FlameMirror / compose_with_head）；编排脚本从仓库根目录导入。
人脸检测/对齐也放在 `common/face.py`（dlib + ETH-XGaze 归一化，GazeTR 训练同款），
68 点模型在 `common/models/shape_predictor_68_face_landmarks.dat`（~96MB，gitignore，
用 `python -m common.download_dlib_model` 下载，断点续传可重跑）。

## 环境

- conda 环境名：`gaze_flame`，Python **3.8**（FLAME_PyTorch 依赖较老）
- 关键版本锁定（勿随意升级）：numpy==1.23.5（trimesh 3.6.18 需要 np.bool 等旧别名）、pyrender==0.1.39、trimesh==3.6.18
- dlib 19.24.2（conda-forge）：**conda 安装 dlib 会把 numpy 升到 1.24.4**，
  必须再执行 `pip install --force-reinstall numpy==1.23.5` 降回，否则 trimesh import 报
  `AttributeError: module 'numpy' has no attribute 'bool'`（这是已知坑，已验证）
- 创建/更新环境：`conda env create -f environment.yml`（环境更新后记得上面的 numpy 降级 + 下载 dlib 模型）

## 运行方式

```bash
# 一键脚本 (自动定位 conda + gaze_flame 环境, 无需手动 activate; 默认 CPU)
./run.sh                                  # 默认示例图片 test.jpg (统一单窗口)
./run.sh --image /path/to/face.jpg        # 自定义图片
./run.sh --skip-viewer                    # 只算角度, 不显示窗口
./run.sh --headless --save-image out.png  # 无 X11: 把合成图存 PNG (自动化推荐)
DEVICE=cuda ./run.sh                      # 有 GPU 时走 cuda (无 GPU 自动回退 cpu)

# 或手动方式:
conda activate gaze_flame
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
- **AMD GPU (ROCm) 已启用（2026-09-10 重启后复测，带占空比护栏）**：
  本机 Radeon 680M = gfx1035，装 torch 2.4.1+rocm6.1（cp38 wheel，官方索引或
  上交镜像 mirror.sjtu.edu.cn/pytorch-wheels/rocm6.1/），需
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`（common/gpu.py 自动设）。前向 batch=1 约
  5ms、batch=16 约 35ms。**坑（关键）**：核显与显示共享内存，**背靠背满载**
  forward 到 ~300 次(b1)/60-120 次(b16) 必现 GPU Hang + amdgpu MODE2 reset
  （`HSA_ENABLE_SDMA=0`、禁用 fused SDPA 均无效；挂死后重启才恢复）。
  **对策 = 占空比护栏 + 兼容 kernel**：`common/gpu.py:rocm_igpu_guard(device,
  elapsed)` 把相邻两次 GPU forward 的间隔补足到 `ROCM_SAFE_INTERVAL`(默认 100ms)，
  并由 `prepare_amd_gpu_env` 默认设置 `MIOPEN_DEBUG_GCN_ASM_KERNELS=0`，避免
  gfx1035 的 GCN 汇编 kernel 在长时间重复 forward 后 page fault。通用 HIP kernel
  仍然使用 GPU；实测兼容模式下 220 帧无窗口实时流水线完整通过。稳定 AMD 独显可用
  `GAZE_ROCM_SAFE_KERNELS=0` 恢复汇编 kernel，再按需用 `GAZE_ROCM_SAFE_MODE=0`
  关闭占空比护栏。快速视频回放 / 大批量文件夹仍会被限速到约 6fps 以保安全。
  单张图 run.sh 仍默认 cpu（GPU 上下文初始化多花 ~1.5s，单图不划算；
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

## Agent 注意事项

- 本机 GPU：Radeon 680M 核显（gfx1035，torch 已装 2.4.1+rocm6.1，`cuda` 即 ROCm）。
  可安全用于实时/批量推理，**但必须保留 `rocm_igpu_guard` 占空比护栏**（背靠背
  满载会 GPU Hang，见上文）；**`--flame` 实时模式必须经 `flame_realtime_device`
  回退 cpu**（HIP 与持续 GL 渲染共载必挂）；单张图 run.sh 默认仍 cpu（GPU 初始化不划算）
- 交互式窗口（`run_gaze_to_flame` 的 cv2 窗口 / `run_webcam` / `gaze_viewer` 的
  pyrender.Viewer）都是阻塞式 GUI：自动化验证请用 `--headless --save-image`（或
  `--save-frames`），可正常退出（退出码 0）；`--headless` 需配合保存路径。
  无显示环境若要离屏渲染，确保有可用 GL（本机 `DISPLAY=:1`）
- 设备选择统一走 `common/gpu.py`：`resolve_device("auto"/"cuda")` 用**真实 kernel
  探测**（ROCm 对不支持芯片 `is_available()` 会误报 True，kernel 才报错，勿改回）；
  `prepare_amd_gpu_env()` 必须在任何 `import torch` 之前调用（所有入口脚本头部已
  接好）：设 `HSA_OVERRIDE_GFX_VERSION=10.3.0` + 抢注系统 libdrm——ROCm 版 torch
  会连带 import 旧 libdrm 破坏 pyrender 的 GLX（NoSuchConfigException），换回
  CPU/CUDA 构建后此调用无副作用
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
- 由于当前环境的 gfx1035 需要 ISA override，实时入口默认禁用 MIOpen GCN 汇编
  kernel（`MIOPEN_DEBUG_GCN_ASM_KERNELS=0`），仍是 GPU 推理；稳定 AMD 独显可用
  `GAZE_ROCM_SAFE_KERNELS=0` 恢复。
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
