#!/usr/bin/env bash
# GazeFlame 一键运行脚本: GazeTR 视线估计 -> FLAME 3D 头部渲染
#
# 用法:
#   ./run.sh                          # 默认 cpu, 使用示例图片 test.jpg
#   ./run.sh --image /path/to/face.jpg
#   DEVICE=cuda ./run.sh              # 强制 GPU (NVIDIA CUDA 或 AMD ROCm 都写 cuda;
#                                     #  单图不一定更快, 批量文件夹/验证 GPU 时用)
#
# 脚本会:
#   1. 自动定位 conda (PATH 或 /opt/anaconda/bin)
#   2. 直接使用 gaze_flame 环境的 python, 避免 shell activation 问题
#   3. --device 透传给 run_gaze_to_flame.py (auto 时由 Python 端探测 GPU)

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

# 设备选择: 默认 cpu —— 单张图片的前向只占 ~12ms (cpu) / ~5ms (GPU),
# 而 ROCm 上下文初始化要多花 ~1.5s, 单图场景 GPU 反而更慢。
# 批量图片文件夹或想验证 GPU 时: DEVICE=cuda ./run.sh ... (自动含 gfx override)。
# GPU 探测/AMD ROCm 的 HSA_OVERRIDE_GFX_VERSION 都在 Python 端
# (common/gpu.py): torch.cuda.is_available() 对 ROCm 不支持的芯片会误报
# True(kernel 执行才报错), 所以 shell 里不做探测。DEVICE 别名: gpu/rocm/cuda。
DEVICE="${DEVICE:-cpu}"

exec "${PY}" "${REPO_ROOT}/run_gaze_to_flame.py" --device "${DEVICE}" "$@"
