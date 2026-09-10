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
- 实时 webcam 已实测（2026-09-09）：本机 `/dev/video0` 存在、`DISPLAY=:1` 可用；
  摄像头读取 ~100ms/帧（首次 ~700ms 预热，分辨率无关，上限 ~10fps），
  单帧流水线（降采样检测 25ms + 对齐 + GazeTR 16ms）约 **10fps**
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

## Agent 注意事项

- 本机无 GPU：所有 torch 推理加 `--device cpu`
- 交互式窗口（`run_gaze_to_flame` 的 cv2 窗口 / `run_webcam` / `gaze_viewer` 的
  pyrender.Viewer）都是阻塞式 GUI：自动化验证请用 `--headless --save-image`（或
  `--save-frames`），可正常退出（退出码 0）；`--headless` 需配合保存路径。
  无显示环境若要离屏渲染，确保有可用 GL（本机 `DISPLAY=:1`）
- `run_gaze_to_flame.py`：GazeTR 推理仍走 subprocess + `sys.executable`（cwd=gazetr/），
  结果经 **JSON 临时文件**传递 pitch/yaw；FLAME 头部改为**进程内** `FlameMirror` 离屏渲染，
  再用 `compose_with_head` 与标注图合成单窗口。路径已做绝对化解析（勿改回相对）
- 统一可视化的公共件：几何 `common/geometry.py`、2D 标注 `common/viz.draw_overlay`
  （图片与摄像头共用，勿在 run_webcam/run_gaze_to_flame 里各写一份）、
  FLAME 渲染+合成 `common/flame_mirror.py`（`FlameMirror` 从 run_webcam 抽出，二者共用）
- `common/flame_mirror.py` 会把 `flame/` 加入 sys.path 并复用 `gaze_viewer` 的
  build_scene/estimate_eye_centers 等函数，因此 `flame/gaze_viewer.py` 仍是单看 3D 的入口
- 视线几何换算统一在 `common/geometry.py`；`gazetr/gtools.py` 保留上游实现
  （gazetr 内部使用），不要在别处再复制一份数学公式
- 人脸检测/对齐统一在 `common/face.py`（dlib 检测 + ETH-XGaze 6点归一化，
  复刻官方 demo.py 的 solvePnP + warpPerspective 参数，勿在 gazetr/flame 里另写一份）
- `gazetr/inference.py` 默认开启对齐（`--no-align` 关闭）；无 dlib 模型或检测
  不到人脸时会打印警告并回退直接缩放，不会中断流水线
- `flame/config.py` 是 `FLAMEConfig` dataclass，无模块级 parser 副作用；
  `gaze_viewer.py` 用 `add_flame_args + config_from_args` 构造配置
