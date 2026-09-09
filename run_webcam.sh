#!/usr/bin/env bash
# GazeFlame 实时摄像头/视频流视线追踪 (GazeTR + 可选 FLAME 头部镜像)
#
# 用法:
#   ./run_webcam.sh                        # 摄像头 0, 自动镜像, CPU
#   ./run_webcam.sh --flame                # 打开 FLAME 头部镜像窗口
#   ./run_webcam.sh --video clip.mp4       # 处理视频文件
#   ./run_webcam.sh --camera 1 --no-mirror # 指定摄像头, 关闭镜像
#   DEVICE=cuda ./run_webcam.sh            # 有 GPU 时用 cuda
#
# 交互: q/ESC 退出, s 保存快照
# 无显示环境 (服务器/CI): 加 --headless --save-frames out/

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

DEVICE="${DEVICE:-auto}"
if [[ "${DEVICE}" == "cuda" ]]; then
  if ! "${PY}" -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "警告: DEVICE=cuda 但未检测到可用 GPU, 回退到 cpu" >&2
    DEVICE="cpu"
  fi
fi

exec "${PY}" "${REPO_ROOT}/run_webcam.py" --device "${DEVICE}" "$@"
