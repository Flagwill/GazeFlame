"""Download the dlib 68-point shape predictor model into common/models/.

Usage:
    python -m common.download_dlib_model

Skips the download if the model is already present.  Supports resuming a
partially downloaded file (the connection to the hosting servers is often
flaky), so you can safely re-run it until it reports success.
"""

from __future__ import annotations

import bz2
import sys
import urllib.request

from common.face import DLIB_MODEL_URL, PREDICTOR_BZ2_PATH, PREDICTOR_PATH

EXPECTED_SIZE = 64_040_097  # bytes of the .bz2 file served by the mirrors


def _fetch_with_resume(url: str, dest, expected_size: int) -> None:
    """Download ``url`` to ``dest``, resuming across connection drops."""
    for attempt in range(1, 200):
        current = dest.stat().st_size if dest.exists() else 0
        if current >= expected_size:
            return

        req = urllib.request.Request(url)
        if current > 0:
            req.add_header("Range", f"bytes={current}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(
                dest, "ab"
            ) as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:  # network is flaky; just retry
            print(f"  第 {attempt} 次下载中断 ({type(exc).__name__}): 继续重试...")
            continue
    raise RuntimeError(f"下载超时/失败: {url} (已下载 {dest.stat().st_size} 字节)")


def main() -> int:
    if PREDICTOR_PATH.exists():
        print(f"✓ 模型已存在: {PREDICTOR_PATH}")
        return 0

    PREDICTOR_BZ2_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 dlib 68点模型 ({EXPECTED_SIZE / 1e6:.0f} MB)...")
    _fetch_with_resume(DLIB_MODEL_URL, PREDICTOR_BZ2_PATH, EXPECTED_SIZE)

    got = PREDICTOR_BZ2_PATH.stat().st_size
    if got != EXPECTED_SIZE:
        print(f"错误: 文件大小不匹配 ({got} != {EXPECTED_SIZE})", file=sys.stderr)
        return 1

    print(f"✓ 下载完成，正在解压...")
    with bz2.open(PREDICTOR_BZ2_PATH, "rb") as src, open(PREDICTOR_PATH, "wb") as dst:
        while True:
            chunk = src.read(1 << 16)
            if not chunk:
                break
            dst.write(chunk)
    PREDICTOR_BZ2_PATH.unlink()
    print(f"✓ 模型就绪: {PREDICTOR_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
