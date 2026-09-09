# AGENTS.md — GazeFlame 仓库记忆

## 项目概述

GazeTR（Transformer 视线估计）× FLAME_PyTorch（3D 头部参数模型）的合并流水线：

```
人脸图片 → GazeTR (gazetr/inference.py) → Pitch/Yaw (弧度)
        → run_gaze_to_flame.py 坐标换算 (common/geometry.py) → 3D 视线向量
        → FLAME (flame/gaze_viewer.py) + pyrender → 带视线箭头的 3D 头部交互窗口
```

两个子项目用 `gazetr/`、`flame/` 子目录隔离（顶层有重名的 model/、config.py 等）。
共享工具放顶层 `common/`（几何换算的唯一数据源，编排脚本从仓库根目录导入）。

## 环境

- conda 环境名：`gaze_flame`，Python **3.8**（FLAME_PyTorch 依赖较老）
- 关键版本锁定（勿随意升级）：numpy==1.23.5（trimesh 3.6.18 需要 np.bool 等旧别名）、pyrender==0.1.39、trimesh==3.6.18
- 创建/更新环境：`conda env create -f environment.yml`

## 运行方式

```bash
# 一键脚本 (自动定位 conda + gaze_flame 环境, 无需手动 activate; 默认 CPU)
./run.sh                                  # 默认示例图片 test.jpg
./run.sh --image /path/to/face.jpg        # 自定义图片
./run.sh --skip-viewer                    # 只算角度, 不启动 3D 窗口
./run.sh --headless --save-image out.png  # 无 X11 环境下渲染 PNG (自动化推荐)
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
```

## 已验证事实（2026-09-08，CPU）

- 全链路跑通：test.jpg → Pitch=-0.2064 rad, Yaw=-0.7247 rad → gaze 向量 (0.1534, 0.6629, -0.7328)
- CPU 下 GazeTR 单张推理约 **数秒**（2026-09-08 重构后；此前 1–2 分钟是因
  inference.py 默认弹 tkinter 窗口阻塞，已改为 --popup 显式启用）
- `pyrender.Viewer` 需要 **X11 显示环境**（弹出 GLFW 窗口，阻塞至手动关闭）；
  无 X11/后台会话中会静默退出 → 自动化请用 `--headless --save-image`

## Agent 注意事项

- 本机无 GPU：所有 torch 推理加 `--device cpu`
- viewer 是阻塞式 GUI：自动化验证时建议 `timeout N python gaze_viewer.py ...`，
  退出码 124 = 窗口正常打开并被 timeout 杀掉（成功），非零其他退出码 = 失败；
  更推荐 `--headless --save-image`（可正常退出，退出码 0）
- `run_gaze_to_flame.py` 用 subprocess + `sys.executable` 串联两阶段，结果经
  **JSON 临时文件**传递 pitch/yaw；路径已做绝对化解析（勿改回相对）
- 子进程 cwd 分别为 `gazetr/`、`flame/`，脚本内 import 依赖该目录结构
- 视线几何换算统一在 `common/geometry.py`；`gazetr/gtools.py` 保留上游实现
  （gazetr 内部使用），不要在别处再复制一份数学公式
- `flame/config.py` 是 `FLAMEConfig` dataclass，无模块级 parser 副作用；
  `gaze_viewer.py` 用 `add_flame_args + config_from_args` 构造配置
