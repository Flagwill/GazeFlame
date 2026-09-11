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

CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" && -x /opt/anaconda/bin/conda ]]; then
  CONDA_BIN=/opt/anaconda/bin/conda
fi
if [[ -z "${CONDA_BIN}" ]]; then
  echo "错误: 找不到 conda, 请先安装或将其加入 PATH" >&2
  exit 1
fi

ENV_NAME="gaze_flame"
ENV_PATH="$(awk -v n="${ENV_NAME}" '
  $1 == n {print $2; exit}
  $1 == "*" && $2 == n {print $3; exit}' <<< "$("${CONDA_BIN}" env list)")"
if [[ -z "${ENV_PATH}" ]]; then
  echo "错误: conda 环境 ${ENV_NAME} 不存在, 请先运行:" >&2
  echo "  conda env create -f ${REPO_ROOT}/environment.yml" >&2
  exit 1
fi

PY="${ENV_PATH%/}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "错误: 找不到环境 python (env path: ${ENV_PATH})" >&2
  exit 1
fi

# GPU 探测与 AMD ROCm 环境变量由 Python 端 common/gpu.py 负责
# (shell 里 torch.cuda.is_available() 对 ROCm 不支持的芯片会误报 True)。
DEVICE="${DEVICE:-auto}"

exec "${PY}" "${REPO_ROOT}/run_webcam.py" --device "${DEVICE}" "$@"
