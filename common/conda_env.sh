#!/usr/bin/env bash
# Shared interpreter lookup for run.sh / run_webcam.sh / run_demo.sh.
# Source this file, then PY="$(gazeflame_find_python)". No shell activation needed.

gazeflame_find_python() {
  local env_name="gaze_flame_rocm" py conda_bin env_path
  if [[ -n "${GAZE_PYTHON:-}" ]]; then
    py="${GAZE_PYTHON}"
  elif [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" &&
          -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    # Avoid conda startup (and its base-Python warnings) when already active.
    py="${CONDA_PREFIX%/}/bin/python"
  else
    conda_bin="$(command -v conda || true)"
    if [[ -z "${conda_bin}" && -x /opt/anaconda/bin/conda ]]; then
      conda_bin=/opt/anaconda/bin/conda
    fi
    if [[ -z "${conda_bin}" ]]; then
      echo "错误: 找不到 conda；请安装 conda 或设置 GAZE_PYTHON=/path/to/python" >&2
      return 1
    fi
    # conda prints NAME PATH, NAME * PATH, NAME * + PATH, etc. Select the
    # absolute Unix path, NOT a fixed field; preserve spaces in the path.
    env_path="$(awk -v n="${env_name}" '
      $1 == n || ($1 ~ /^[*+]+$/ && $2 == n) {
        start = index($0, "/")
        if (start) {
          path = substr($0, start)
          sub(/[[:space:]]+[*+]+[[:space:]]*$/, "", path)
          sub(/[[:space:]]+$/, "", path)
          print path
          exit
        }
      }' <<< "$("${conda_bin}" env list)")"
    if [[ -z "${env_path}" ]]; then
      echo "错误: conda 环境 ${env_name} 不存在；请运行 conda env create -f environment.yml" >&2
      return 1
    fi
    py="${env_path%/}/bin/python"
  fi
  if [[ ! -x "${py}" ]]; then
    echo "错误: Python 不可执行: ${py}" >&2
    return 1
  fi
  printf '%s\n' "${py}"
}
