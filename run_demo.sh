#!/usr/bin/env bash
# 报告演示（默认真实单图 + 合成场景回放）；全链路实时：--camera 0
# 完整说明：docs/DEMO.md。DEVICE=cpu 为默认，避免 ROCm/HIP 与 GL 共载。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/common/conda_env.sh"
PY="$(gazeflame_find_python)"
exec "${PY}" "${ROOT}/run_demo.py" --device "${DEVICE:-cpu}" "$@"
