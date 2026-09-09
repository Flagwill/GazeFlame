#!/usr/bin/env bash
# GazeFlame 一键运行脚本: GazeTR 视线估计 -> FLAME 3D 头部渲染
#
# 用法:
#   ./run.sh                          # 默认 CPU, 使用示例图片 test.jpg
#   ./run.sh --image /path/to/face.jpg
#   DEVICE=cuda ./run.sh              # 有 GPU 时用 cuda (需本机存在 NVIDIA 驱动)
#
# 脚本会:
#   1. 自动定位 conda (PATH 或 /opt/anaconda/bin)
#   2. 直接使用 gaze_flame 环境的 python, 避免 shell activation 问题
#   3. 默认 --device cpu; 检测到 GPU 且 DEVICE=cuda 时改用 cuda

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

# 从 conda env list 解析环境真实路径 (envs/ 可能在 ~/.conda/envs 或 $CONDA_ROOT/envs)
# 兼容 active 标记在名称前 ("* gaze_flame ...") 或在行尾 ("gaze_flame ... *") 两种格式
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

# 设备选择: 默认 cpu (本机无 GPU); DEVICE=cuda 时, 若 torch 检测到 GPU 则用 cuda
DEVICE="${DEVICE:-cpu}"
if [[ "${DEVICE}" == "cuda" ]]; then
  if ! "${PY}" -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "警告: DEVICE=cuda 但未检测到可用 GPU, 回退到 cpu" >&2
    DEVICE="cpu"
  fi
fi

exec "${PY}" "${REPO_ROOT}/run_gaze_to_flame.py" --device "${DEVICE}" "$@"
