#!/usr/bin/env bash
# GazeFlame 实时摄像头/视频流视线追踪 (GazeTR + 可选 FLAME 头部镜像)
#
# 用法:
#   ./run_webcam.sh                        # 摄像头 0, 自动镜像, auto 选择可用 GPU
#   ./run_webcam.sh --flame                # 打开 FLAME 头部镜像窗口
#   ./run_webcam.sh --video clip.mp4       # 处理视频文件
#   ./run_webcam.sh --camera 1 --no-mirror # 指定摄像头, 关闭镜像
#   DEVICE=cpu ./run_webcam.sh         # 强制 CPU (默认 auto: 有 GPU 自动用)
#
# 交互: q/ESC 退出, s 保存快照
# 无显示环境 (服务器/CI): 加 --headless --save-frames out/
# 注意: ROCm 核显上无 --flame 的 auto 使用 GPU+占空比护栏；--flame 因 GL+HIP
# 共载风险自动回退 cpu 推理 (见 common/gpu.py)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${REPO_ROOT}/common/conda_env.sh"
PY="$(gazeflame_find_python)"

# GPU 探测与 AMD ROCm 环境变量由 Python 端 common/gpu.py 负责
# (shell 里 torch.cuda.is_available() 对 ROCm 不支持的芯片会误报 True)。
DEVICE="${DEVICE:-auto}"

exec "${PY}" "${REPO_ROOT}/run_webcam.py" --device "${DEVICE}" "$@"
