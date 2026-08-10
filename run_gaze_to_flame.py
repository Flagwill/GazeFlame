"""Run GazeTR inference and forward the resulting gaze vector to FLAME.

This is the Linux-friendly orchestrator for the merged GazeFlame repo.
It chains the two stages directly with subprocess (no PowerShell / conda-hook):

1. Run GazeTR inference.py on a single image.
2. Read pitch/yaw from a temporary result file.
3. Convert pitch/yaw to a normalized 3D gaze vector.
4. Run flame/gaze_viewer.py with that vector to render the FLAME head.

By default the sub-projects are assumed to live at <this script dir>/gazetr and
<this script dir>/flame, but every path can be overridden via CLI args.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GAZETR_ROOT = REPO_ROOT / "gazetr"
DEFAULT_FLAME_ROOT = REPO_ROOT / "flame"
DEFAULT_MODEL = DEFAULT_GAZETR_ROOT / "model" / "GazeTR-H-ETH.pt"
DEFAULT_IMAGE = DEFAULT_GAZETR_ROOT / "images" / "inference" / "test.jpg"


def pitch_yaw_to_gaze_xyz(pitch: float, yaw: float) -> dict[str, float]:
    """Convert gaze pitch/yaw (radians) into a normalized 3D direction vector."""
    x = -math.cos(yaw) * math.sin(pitch)
    y = -math.sin(yaw)
    z = -math.cos(yaw) * math.cos(pitch)
    return {"x": x, "y": y, "z": z}


def parse_result_file(result_file: Path) -> dict[str, float]:
    """Parse the last data line of inference.py's --output file as pitch/yaw."""
    lines = [
        line.strip()
        for line in result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"结果文件内容不足: {result_file}")

    last_line = lines[-1]
    parts = last_line.split()
    if len(parts) < 3:
        raise ValueError(f"无法解析推理结果: {last_line}")

    return {"pitch": float(parts[1]), "yaw": float(parts[2])}


def run_python(args: list[str], cwd: Path) -> None:
    print(f"\n[GazeTR→FLAME] 运行: python {' '.join(args)}  (cwd: {cwd})")
    completed = subprocess.run([sys.executable, *args], cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GazeTR inference and forward gaze to FLAME."
    )
    parser.add_argument(
        "--gazetr-root",
        default=str(DEFAULT_GAZETR_ROOT),
        help="GazeTR (merged gazetr/) project root",
    )
    parser.add_argument(
        "--flame-root",
        default=str(DEFAULT_FLAME_ROOT),
        help="FLAME (merged flame/) project root",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="GazeTR model path"
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument(
        "--image_folder", default=None, help="Input image folder (alternative to --image)"
    )
    parser.add_argument(
        "--gaze-viewer",
        default=None,
        help="FLAME gaze viewer script (default: <flame-root>/gaze_viewer.py)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Inference device"
    )
    args = parser.parse_args()

    gazetr_root = Path(args.gazetr_root)
    flame_root = Path(args.flame_root)
    model_path = Path(args.model)
    image_path = Path(args.image)

    gaze_viewer = Path(args.gaze_viewer) if args.gaze_viewer else flame_root / "gaze_viewer.py"

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not gaze_viewer.exists():
        raise FileNotFoundError(f"gaze_viewer.py 不存在: {gaze_viewer}")

    with tempfile.TemporaryDirectory(prefix="gazetr_") as temp_dir:
        result_file = Path(temp_dir) / "results.txt"

        inference_args = [
            str(gazetr_root / "inference.py"),
            "--model", str(model_path),
        ]
        if args.image_folder:
            inference_args += ["--image_folder", args.image_folder]
        else:
            if not image_path.exists():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            inference_args += ["--image", str(image_path)]
        inference_args += ["--output", str(result_file), "--device", args.device]
        run_python(inference_args, cwd=gazetr_root)

        pitch_yaw = parse_result_file(result_file)
        gaze = pitch_yaw_to_gaze_xyz(pitch_yaw["pitch"], pitch_yaw["yaw"])

        print(
            f"\nPitch: {pitch_yaw['pitch']:.6f} rad, "
            f"Yaw: {pitch_yaw['yaw']:.6f} rad"
        )
        print(
            f"Gaze vector: x={gaze['x']:.6f}, y={gaze['y']:.6f}, z={gaze['z']:.6f}"
        )

        gaze_viewer_args = [
            str(gaze_viewer),
            "--gaze-x", f"{gaze['x']:.6f}",
            "--gaze-y", f"{gaze['y']:.6f}",
            "--gaze-z", f"{gaze['z']:.6f}",
        ]
        run_python(gaze_viewer_args, cwd=flame_root)


if __name__ == "__main__":
    main()