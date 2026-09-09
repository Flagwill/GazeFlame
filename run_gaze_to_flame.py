"""Run GazeTR inference and forward the resulting gaze vector to FLAME.

Orchestrates the two stages of the GazeFlame pipeline with subprocesses
(keeping the two sub-projects' own import layouts intact):

1. Run ``gazetr/inference.py`` on a single image (or folder).
2. Read pitch/yaw from a temporary JSON result file.
3. Convert pitch/yaw to a normalized 3D gaze vector (``common.geometry``).
4. Run ``flame/gaze_viewer.py`` with that vector to render the FLAME head.

Paths are resolved to absolute at startup so the subprocesses (whose cwd is
the gazetr/flame root) never mis-resolve relative arguments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from common.geometry import gaze_to_dict, pitch_yaw_to_gaze

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GAZETR_ROOT = REPO_ROOT / "gazetr"
DEFAULT_FLAME_ROOT = REPO_ROOT / "flame"
DEFAULT_MODEL = DEFAULT_GAZETR_ROOT / "model" / "GazeTR-H-ETH.pt"
DEFAULT_IMAGE = DEFAULT_GAZETR_ROOT / "images" / "inference" / "test.jpg"


def run_python(args: list[str], cwd: Path) -> None:
    """Run ``sys.executable`` with ``args`` from ``cwd``; exit on failure."""
    print(f"\n[GazeTR→FLAME] 运行: python {' '.join(args)}  (cwd: {cwd})")
    completed = subprocess.run([sys.executable, *args], cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_gaze_result(result_file: Path) -> dict[str, float]:
    """Read pitch/yaw from the inference JSON output file."""
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析推理结果文件: {result_file}") from exc

    results = payload.get("results")
    if not results:
        raise ValueError(f"推理结果为空: {result_file}")
    return {"pitch": float(results[0]["pitch"]), "yaw": float(results[0]["yaw"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GazeTR inference and forward gaze to FLAME."
    )
    parser.add_argument("--gazetr-root", default=str(DEFAULT_GAZETR_ROOT),
                        help="GazeTR (gazetr/) project root")
    parser.add_argument("--flame-root", default=str(DEFAULT_FLAME_ROOT),
                        help="FLAME (flame/) project root")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="GazeTR model path")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument("--image_folder", default=None,
                        help="Input image folder (alternative to --image)")
    parser.add_argument("--gaze-viewer", default=None,
                        help="FLAME gaze viewer script "
                             "(default: <flame-root>/gaze_viewer.py)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (auto-falls back to cpu)")
    parser.add_argument("--skip-viewer", action="store_true",
                        help="Only run GazeTR and print pitch/yaw/gaze; "
                             "do not launch the FLAME viewer")
    parser.add_argument("--save-image", default=None,
                        help="Passed to the viewer: also render the scene to a PNG")
    parser.add_argument("--headless", action="store_true",
                        help="Passed to the viewer: no interactive window "
                             "(requires --save-image)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    gazetr_root = Path(args.gazetr_root).resolve()
    flame_root = Path(args.flame_root).resolve()
    # 绝对化解析: 子进程的 cwd 是 gazetr_root/flame_root, 相对路径必须基于调用方 cwd 解析。
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    image_folder = Path(args.image_folder).resolve() if args.image_folder else None
    gaze_viewer = (
        Path(args.gaze_viewer).resolve()
        if args.gaze_viewer
        else flame_root / "gaze_viewer.py"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not args.skip_viewer and not gaze_viewer.exists():
        raise FileNotFoundError(f"gaze_viewer.py 不存在: {gaze_viewer}")

    with tempfile.TemporaryDirectory(prefix="gazetr_") as temp_dir:
        result_file = Path(temp_dir) / "results.json"

        inference_args = [
            str(gazetr_root / "inference.py"),
            "--model", str(model_path),
        ]
        if image_folder:
            inference_args += ["--image_folder", str(image_folder)]
        else:
            if not image_path.exists():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            inference_args += ["--image", str(image_path)]
        inference_args += ["--output", str(result_file), "--device", args.device]
        run_python(inference_args, cwd=gazetr_root)

        pitch_yaw = read_gaze_result(result_file)
        gaze = pitch_yaw_to_gaze(pitch_yaw["pitch"], pitch_yaw["yaw"])
        gaze_dict = gaze_to_dict(gaze)

        print(
            f"\nPitch: {pitch_yaw['pitch']:.6f} rad, "
            f"Yaw: {pitch_yaw['yaw']:.6f} rad"
        )
        print(
            f"Gaze vector: x={gaze_dict['x']:.6f}, "
            f"y={gaze_dict['y']:.6f}, z={gaze_dict['z']:.6f}"
        )

        if args.skip_viewer:
            return

        gaze_viewer_args = [
            str(gaze_viewer),
            "--gaze-x", f"{gaze_dict['x']:.6f}",
            "--gaze-y", f"{gaze_dict['y']:.6f}",
            "--gaze-z", f"{gaze_dict['z']:.6f}",
            "--device", args.device,
        ]
        if args.save_image:
            gaze_viewer_args += ["--save-image", str(Path(args.save_image).resolve())]
        if args.headless:
            gaze_viewer_args += ["--headless"]
        run_python(gaze_viewer_args, cwd=flame_root)


if __name__ == "__main__":
    main()
