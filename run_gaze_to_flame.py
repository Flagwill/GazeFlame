"""Run GazeTR inference and forward the resulting gaze vector to FLAME.

Orchestrates the two stages of the GazeFlame pipeline:

1. Load GazeTR and run it IN-PROCESS on a single image (or folder) — same
   sys.path pattern as ``run_webcam.py`` (``gazetr/`` on ``sys.path``), which
   saves a second interpreter startup + a second torch/dlib import (~2s).
   ``--subprocess`` keeps the legacy path (``gazetr/inference.py`` child
   process + JSON temp file) as a fallback.
2. Convert pitch/yaw to a normalized 3D gaze vector (``common.geometry``).
3. Render the FLAME head toward that vector (``common.flame_mirror``) and show
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
import time
from pathlib import Path

from common.gpu import flame_render_device, prepare_amd_gpu_env, resolve_device

# 必须早于任何 import torch (ROCm: HSA 覆盖 + 抢注系统 libdrm)。
prepare_amd_gpu_env()

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


def run_gaze_inprocess(
    gazetr_root: Path,
    model_path: Path,
    image_path: Path,
    image_folder: Path | None,
    align: bool,
    align_out: str | None,
    device: str,
    result_file: Path,
) -> None:
    """Load GazeTR in this interpreter and write the JSON result file.

    Mirrors the subprocess call (same ``inference()`` code path) but avoids a
    second interpreter startup + torch/dlib import (~2 s on this machine).
    """
    t0 = time.perf_counter()
    sys.path.insert(0, str(gazetr_root))
    import torch

    import model as gazetr_model  # gazetr/model.py (before flame/ joins sys.path)
    from inference import collect_image_paths, inference as gaze_inference

    print(f"\n[GazeTR→FLAME] 进程内推理 (device: {device})")
    net = gazetr_model.Model()
    net.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    net.to(device)
    image_paths = collect_image_paths(
        str(image_path) if not image_folder else None,
        str(image_folder) if image_folder else None,
    )
    gaze_inference(
        net, image_paths, device=device, output_file=str(result_file), align=align,
        align_out=align_out,
    )
    print(f"[GazeTR→FLAME] GazeTR 阶段耗时 {time.perf_counter() - t0:.2f}s")


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
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "gpu", "rocm"],
                        help="Inference device; auto = GPU (ROCm/CUDA) if usable "
                             "(falls back to cpu). gpu/rocm are cuda aliases")
    parser.add_argument("--subprocess", action="store_true",
                        help="Legacy mode: run gazetr/inference.py in a child "
                             "process instead of in-process")
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

    t_pipeline = time.perf_counter()
    device = resolve_device(args.device)
    if device == "cuda":
        import torch

        hip = getattr(torch.version, "hip", None)
        backend = f"ROCm {hip}" if hip else "CUDA"
        print(f"推理设备: cuda ({backend}, {torch.cuda.get_device_name(0)})")
    else:
        print("推理设备: cpu")

    with tempfile.TemporaryDirectory(prefix="gazetr_") as temp_dir:
        result_file = Path(temp_dir) / "results.json"

        if args.subprocess:
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
            inference_args += ["--output", str(result_file), "--device", device]
            if args.no_align:
                inference_args += ["--no-align"]
            if args.align_out:
                inference_args += ["--align-out", str(Path(args.align_out).resolve())]
            run_python(inference_args, cwd=gazetr_root)
        else:
            if not image_folder and not image_path.exists():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            run_gaze_inprocess(
                gazetr_root, model_path, image_path, image_folder,
                align=not args.no_align, align_out=args.align_out,
                device=device, result_file=result_file,
            )

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
            print(f"[GazeTR→FLAME] 总耗时 {time.perf_counter() - t_pipeline:.2f}s")
            return

        # 统一窗口：左侧标注输入图 + 右侧 FLAME 头部，二者合成到同一画面。
        t_flame = time.perf_counter()
        from common.flame_mirror import FlameMirror
        from common.viz import draw_overlay

        source_image = Path(result["image"]).resolve() if result["image"] else image_path
        bgr = cv2.imread(str(source_image))
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {source_image}")
        panel = scale_to_width(bgr, args.display_width // 2)
        annotated = draw_overlay(panel, pitch, yaw, info=None, show_face=False)

        mirror = FlameMirror(flame_render_device(device), args.head_size, args.head_size)
        head_img = mirror.render(gaze)
        print(f"[GazeTR→FLAME] FLAME 渲染阶段耗时 {time.perf_counter() - t_flame:.2f}s")
        show_or_save(annotated, head_img, save_image, args.headless)
        print(f"[GazeTR→FLAME] 总耗时 {time.perf_counter() - t_pipeline:.2f}s")


def scale_to_width(img, max_w: int):
    """Downscale so the width is at most ``max_w`` (keeps large photos sane)."""
    h, w = img.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        return cv2.resize(img, (max_w, max(1, int(h * scale))))
    return img


if __name__ == "__main__":
    main()
