"""Run GazeTR inference and forward the resulting gaze vector to FLAME.

Orchestrates the two stages of the GazeFlame pipeline:

1. Load GazeTR and run it IN-PROCESS on a single image (or folder) — same
   sys.path pattern as ``run_webcam.py`` (``gazetr/`` on ``sys.path``), which
   saves a second interpreter startup + a second torch/dlib import (~2s).
   ``--subprocess`` keeps the legacy path (``gazetr/inference.py`` child
   process + JSON temp file) as a fallback.
2. Convert the two GazeTR angles to a normalized camera gaze vector
   (``common.geometry``).
3. Render the FLAME head toward that vector (``common.flame_mirror``) and show
   it side-by-side with the annotated input image in ONE cv2 window.
4. Optionally add a software-only assistive-control concept panel that maps
   the gaze point to abstract cockpit regions.  It stops at intent display;
   no robot or flight controller is simulated.

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

from common.gpu import (
    flame_render_device,
    fp16_autocast_enabled,
    prepare_amd_gpu_env,
    resolve_device,
)

# 必须早于任何 import torch (ROCm: HSA 覆盖 + 抢注系统 libdrm)。
prepare_amd_gpu_env()

import cv2
import numpy as np

from common.calibration import camera_gaze_to_screen_point
from common.face import DEFAULT_FOV_DEG
from common.geometry import (
    camera_gaze_to_pitch_yaw,
    camera_to_flame,
    gaze_to_camera,
    gaze_to_dict,
    gazetr_angles_to_gaze,
)

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
    fov_deg: float,
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
        align_out=align_out, fov_deg=fov_deg,
    )
    print(f"[GazeTR→FLAME] GazeTR 阶段耗时 {time.perf_counter() - t0:.2f}s")


def read_gaze_result(result_file: Path) -> dict:
    """Read the first result (image path + raw GazeTR angles) from JSON."""
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
                        help="Only run GazeTR and print angles/gaze; "
                             "do not show the unified window")
    parser.add_argument("--no-align", action="store_true",
                        help="Disable face detection/alignment (ETH-XGaze "
                             "normalization) in the GazeTR stage")
    parser.add_argument("--camera-fov", type=float, default=DEFAULT_FOV_DEG,
                        help="source camera horizontal FOV in degrees for solvePnP "
                             f"(default: {DEFAULT_FOV_DEG:g})")
    parser.add_argument("--align-out", default=None,
                        help="Save aligned 224x224 face crops (with landmarks) "
                             "to this directory for debugging")
    parser.add_argument("--head-size", type=int, default=512,
                        help="FLAME panel render edge length (default: 512)")
    parser.add_argument("--hide-gaze-arrows", action="store_true",
                        help="hide debug arrows in the FLAME eye panel")
    parser.add_argument("--assistive", action="store_true",
                        help="add the software-only assistive-control concept panel")
    parser.add_argument("--display-width", type=int, default=1280,
                        help="Max width of the composed window/panel scaling")
    parser.add_argument("--save-image", default=None,
                        help="Write the composed single-window image to this PNG")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open an interactive window; render to a PNG "
                             "(requires --save-image)")
    parser.add_argument("--windowed", action="store_true",
                        help="keep the visualization window resizable instead of fullscreen")
    return parser


def show_or_save(
    annotated: "np.ndarray",
    head_img: "np.ndarray",
    save_image: str | None,
    headless: bool,
    window_title: str = "GazeFlame",
    fullscreen: bool = True,
    screen_point: "np.ndarray | tuple[float, float] | None" = None,
    assistive_panel: "np.ndarray | None" = None,
) -> None:
    """Compose the visualization panels and either save or show one window."""
    from common.flame_mirror import compose_with_head
    from common.assistive import compose_with_assistive

    from common.viz import draw_screen_point, fit_to_display

    if assistive_panel is not None:
        annotated = draw_screen_point(annotated, screen_point)
        composed = compose_with_assistive(annotated, head_img, assistive_panel)
    else:
        composed = compose_with_head(annotated, head_img)
        composed = draw_screen_point(composed, screen_point)
    if save_image:
        cv2.imwrite(str(save_image), composed)
        print(f"已保存合成图: {save_image}")
    if not headless:
        from common.viz import create_display_window

        display_size = create_display_window(window_title, fullscreen=fullscreen)
        cv2.imshow(window_title, fit_to_display(composed, display_size))
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
    if fp16_autocast_enabled(device):
        print("提示: 已启用 fp16 autocast 推理 (约 1.3-1.7x; GAZE_ROCM_FP16=0 关闭)")

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
            inference_args += ["--camera-fov", str(args.camera_fov)]
            run_python(inference_args, cwd=gazetr_root)
        else:
            if not image_folder and not image_path.exists():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            run_gaze_inprocess(
                gazetr_root, model_path, image_path, image_folder,
                align=not args.no_align, align_out=args.align_out,
                device=device, result_file=result_file,
                fov_deg=args.camera_fov,
            )

        result = read_gaze_result(result_file)
        pitch, yaw = result["pitch"], result["yaw"]
        gaze = gazetr_angles_to_gaze(pitch, yaw)
        gaze_dict = gaze_to_dict(gaze)

        print(f"\nGazeTR outputs: angle0={pitch:.6f} rad, angle1={yaw:.6f} rad")
        print(
            f"Normalized camera gaze: x={gaze_dict['x']:.6f}, "
            f"y={gaze_dict['y']:.6f}, z={gaze_dict['z']:.6f}"
        )
        # Recover the same pose metadata for both the viewer and --skip-viewer
        # so the reported FLAME direction is in the original camera frame.
        source_image = Path(result["image"]).resolve() if result["image"] else image_path
        bgr = cv2.imread(str(source_image))
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {source_image}")
        face_info = None
        if not args.no_align:
            # The batch inference result intentionally stays JSON-friendly;
            # recover the alignment metadata here for pose/blink-aware FLAME
            # rendering and the richer overlay.
            from common.face import align_face_to_gaze_crop

            try:
                _, face_info = align_face_to_gaze_crop(
                    bgr, fov_deg=args.camera_fov
                )
                if face_info.get("face_count", 0) == 0:
                    face_info = None
            except Exception as exc:
                print(f"提示: 无法恢复头姿态信息 ({exc})")
        normalization_rotation = (
            face_info.get("normalization_rotation") if face_info else None
        )
        camera_gaze = gaze_to_camera(gaze, normalization_rotation)
        camera_pitch, camera_yaw = camera_gaze_to_pitch_yaw(camera_gaze)
        print(
            f"Camera gaze: pitch={np.degrees(camera_pitch):+.2f} deg, "
            f"yaw={np.degrees(camera_yaw):+.2f} deg"
        )
        flame_gaze = camera_to_flame(camera_gaze)
        flame_dict = gaze_to_dict(flame_gaze)
        print(
            f"FLAME vector: x={flame_dict['x']:.6f}, "
            f"y={flame_dict['y']:.6f}, z={flame_dict['z']:.6f}"
        )
        if args.skip_viewer:
            print(f"[GazeTR→FLAME] 总耗时 {time.perf_counter() - t_pipeline:.2f}s")
            return

        # 统一窗口：左侧标注输入图 + 右侧 FLAME 头部，二者合成到同一画面。
        t_flame = time.perf_counter()
        from common.flame_mirror import FlameMirror
        from common.assistive import AssistiveIntentState, draw_assistive_panel
        from common.viz import draw_overlay

        screen_pitch, screen_yaw = camera_gaze_to_pitch_yaw(camera_gaze)
        screen_point = camera_gaze_to_screen_point(
            screen_pitch, screen_yaw, mirror=False
        )
        panel = scale_to_width(bgr, args.display_width // 2)
        annotated = draw_overlay(
            panel,
            pitch,
            yaw,
            info=face_info,
            show_face=face_info is not None,
            screen_point=screen_point,
            camera_gaze=camera_gaze,
        )

        assistive_panel = None
        if args.assistive:
            # A still image proves the perception/visualization path, but it
            # cannot prove a temporal fixation.  Keep the state honest and
            # label it as current attention rather than inferred intent.
            reprojection_error = float(face_info.get("reprojection_error", 0.0)) if face_info else 0.0
            confidence = float(np.exp(-max(reprojection_error, 0.0) / 8.0)) if face_info else 0.0
            assistive_estimate = AssistiveIntentState().update(
                screen_point,
                valid=face_info is not None,
                confidence=confidence,
                fixation_duration=0.0,
                velocity_deg_s=0.0,
                blinking=False,
            )
            assistive_panel = draw_assistive_panel(
                annotated.shape[0], assistive_estimate
            )

        mirror = FlameMirror(
            flame_render_device(device), args.head_size, args.head_size,
            show_arrows=not args.hide_gaze_arrows,
        )
        try:
            head_img = mirror.render(
                flame_gaze,
                head_pose=face_info.get("head_pose") if face_info else None,
                blink=0.0 if face_info is None else float(
                    np.clip((0.28 - face_info.get("eye_aspect_ratio", 0.28)) / 0.10, 0.0, 1.0)
                ),
            )
            print(f"[GazeTR→FLAME] FLAME 渲染阶段耗时 {time.perf_counter() - t_flame:.2f}s")
            show_or_save(
                annotated,
                head_img,
                save_image,
                args.headless,
                fullscreen=not args.windowed,
                screen_point=screen_point,
                assistive_panel=assistive_panel,
            )
        finally:
            mirror.close()
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
