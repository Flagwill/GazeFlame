"""Run GazeTR inference and forward the resulting gaze vector to FLAME.

Orchestrates the two stages of the GazeFlame pipeline:

1. Run ``gazetr/inference.py`` as a subprocess on a single image (or folder),
   keeping the two sub-projects' own import layouts intact.
2. Read pitch/yaw from a temporary JSON result file.
3. Convert pitch/yaw to a normalized 3D gaze vector (``common.geometry``).
4. Render the FLAME head toward that vector (``common.flame_mirror``) and show
   it side-by-side with the annotated input image in ONE cv2 window.

By default the unified window is shown interactively (needs an X11 display).
With ``--headless --save-image out.png`` the same composed image is written to
disk instead (recommended for automation / servers).

Paths are resolved to absolute at startup so the inference subprocess (whose
cwd is the gazetr root) never mis-resolves relative arguments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from common.geometry import gaze_to_dict, pitch_yaw_to_gaze

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GAZETR_ROOT = REPO_ROOT / "gazetr"
DEFAULT_MODEL = DEFAULT_GAZETR_ROOT / "model" / "GazeTR-H-ETH.pt"
DEFAULT_IMAGE = DEFAULT_GAZETR_ROOT / "images" / "inference" / "test.jpg"


def run_python(args: list[str], cwd: Path) -> None:
    """Run ``sys.executable`` with ``args`` from ``cwd``; exit on failure."""
    print(f"\n[GazeTR→FLAME] 运行: python {' '.join(args)}  (cwd: {cwd})")
    completed = subprocess.run([sys.executable, *args], cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_gaze_result(result_file: Path) -> dict:
    """Read the first result (image path + pitch/yaw) from the JSON output."""
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析推理结果文件: {result_file}") from exc

    results = payload.get("results")
    if not results:
        raise ValueError(f"推理结果为空: {result_file}")
    first = results[0]
    return {
        "image": first.get("image"),
        "pitch": float(first["pitch"]),
        "yaw": float(first["yaw"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GazeTR inference and show gaze + FLAME head in one window."
    )
    parser.add_argument("--gazetr-root", default=str(DEFAULT_GAZETR_ROOT),
                        help="GazeTR (gazetr/) project root")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="GazeTR model path")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument("--image_folder", default=None,
                        help="Input image folder (alternative to --image)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (auto-falls back to cpu)")
    parser.add_argument("--skip-viewer", action="store_true",
                        help="Only run GazeTR and print pitch/yaw/gaze; "
                             "do not show the unified window")
    parser.add_argument("--no-align", action="store_true",
                        help="Disable face detection/alignment (ETH-XGaze "
                             "normalization) in the GazeTR stage")
    parser.add_argument("--align-out", default=None,
                        help="Save aligned 224x224 face crops (with landmarks) "
                             "to this directory for debugging")
    parser.add_argument("--head-size", type=int, default=512,
                        help="FLAME panel render edge length (default: 512)")
    parser.add_argument("--display-width", type=int, default=1280,
                        help="Max width of the composed window/panel scaling")
    parser.add_argument("--save-image", default=None,
                        help="Write the composed single-window image to this PNG")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open an interactive window; render to a PNG "
                             "(requires --save-image)")
    return parser


def show_or_save(
    annotated: "np.ndarray",
    head_img: "np.ndarray",
    save_image: str | None,
    headless: bool,
    window_title: str = "GazeFlame",
) -> None:
    """Compose the two panels and either save them or show a single window."""
    from common.flame_mirror import compose_with_head

    composed = compose_with_head(annotated, head_img)
    if save_image:
        cv2.imwrite(str(save_image), composed)
        print(f"已保存合成图: {save_image}")
    if not headless:
        cv2.imshow(window_title, composed)
        print("按任意键关闭窗口")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()

    gazetr_root = Path(args.gazetr_root).resolve()
    # 绝对化解析: 子进程的 cwd 是 gazetr_root, 相对路径必须基于调用方 cwd 解析。
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    image_folder = Path(args.image_folder).resolve() if args.image_folder else None

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if args.headless and not args.save_image:
        raise SystemExit("--headless 需要配合 --save-image <path>")
    save_image = Path(args.save_image).resolve() if args.save_image else None

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
        if args.no_align:
            inference_args += ["--no-align"]
        if args.align_out:
            inference_args += ["--align-out", str(Path(args.align_out).resolve())]
        run_python(inference_args, cwd=gazetr_root)

        result = read_gaze_result(result_file)
        pitch, yaw = result["pitch"], result["yaw"]
        gaze = pitch_yaw_to_gaze(pitch, yaw)
        gaze_dict = gaze_to_dict(gaze)

        print(f"\nPitch: {pitch:.6f} rad, Yaw: {yaw:.6f} rad")
        print(
            f"Gaze vector: x={gaze_dict['x']:.6f}, "
            f"y={gaze_dict['y']:.6f}, z={gaze_dict['z']:.6f}"
        )

        if args.skip_viewer:
            return

        # 统一窗口：左侧标注输入图 + 右侧 FLAME 头部，二者合成到同一画面。
        from common.flame_mirror import FlameMirror
        from common.viz import draw_overlay

        device = resolve_device(args.device)
        source_image = Path(result["image"]).resolve() if result["image"] else image_path
        bgr = cv2.imread(str(source_image))
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {source_image}")
        panel = scale_to_width(bgr, args.display_width // 2)
        annotated = draw_overlay(panel, pitch, yaw, info=None, show_face=False)

        mirror = FlameMirror(str(device), args.head_size, args.head_size)
        head_img = mirror.render(gaze)
        show_or_save(annotated, head_img, save_image, args.headless)


def resolve_device(device: str) -> str:
    """Map ``cuda``→``cpu`` when no GPU is present; return a torch device str."""
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("警告: CUDA不可用，切换到CPU模式")
                return "cpu"
        except Exception:  # pragma: no cover - torch always present here
            return "cpu"
    return device


def scale_to_width(img, max_w: int):
    """Downscale so the width is at most ``max_w`` (keeps large photos sane)."""
    h, w = img.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        return cv2.resize(img, (max_w, max(1, int(h * scale))))
    return img


if __name__ == "__main__":
    main()
