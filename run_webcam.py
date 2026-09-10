"""Real-time gaze tracking from a webcam or video file, optionally driving FLAME.

Interactive mode opens a single window showing the live camera feed with face
box / landmarks / gaze arrow, and (with ``--flame``) a side-by-side FLAME head
whose eyeballs follow the gaze in real time.  The FLAME panel is rendered
offscreen, so it also works without a GLFW-capable viewer.

Controls: ``q`` / ``ESC`` quit, ``s`` save a snapshot of the camera feed.

Examples:
    python run_webcam.py --camera 0
    python run_webcam.py --camera 0 --flame
    python run_webcam.py --video clip.mp4 --flame --no-align
    python run_webcam.py --camera 0 --headless --save-frames out/  # automation

Notes:
    - dlib face detection is run on a downscaled frame internally (common.face),
      so a 640x480 webcam costs ~30ms/frame instead of ~450ms.
    - The FLAME head is rendered in the main loop at a capped rate
      (``--head-fps``); pyrender's OpenGL context is bound to the creating
      thread, so it cannot live on a separate thread.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GAZETR_ROOT = REPO_ROOT / "gazetr"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GAZETR_ROOT))

# 必须早于 import torch (ROCm: HSA 覆盖 + 抢注系统 libdrm; CPU/CUDA build 无副作用)。
from common.gpu import (  # noqa: E402
    flame_realtime_device,
    flame_render_device,
    prepare_amd_gpu_env,
    rocm_igpu_guard,
)

prepare_amd_gpu_env()

import cv2
import numpy as np
import torch

# GazeTR (gazetr/model.py)
import model  # noqa: E402

# inference.predict_gaze (gazetr/inference.py, pulls in common.face)
from inference import predict_gaze  # noqa: E402

from common.flame_mirror import FlameMirror, compose_with_head  # noqa: E402
from common.geometry import pitch_yaw_to_gaze  # noqa: E402
from common.viz import draw_overlay  # noqa: E402

DEFAULT_MODEL = str(GAZETR_ROOT / "model" / "GazeTR-H-ETH.pt")


def load_gazetr_model(model_path: str, device, warmup: bool = True):
    """Load the pretrained GazeTR model and put it in eval mode."""
    net = model.Model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    if warmup:
        # 预热: 吸收 GPU 首次 kernel/内存池 与 CPU 首次调度的固定开销,
        # 避免实时流水线第一帧明显变慢。
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(3):
                net({"face": dummy})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time webcam/video gaze tracking (single unified window)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None,
                     help="webcam device index (default: 0)")
    src.add_argument("--video", type=str, default=None,
                     help="play a video file instead of the webcam")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GazeTR model path")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "gpu", "rocm"],
                        help="inference device (auto = GPU if usable, else cpu; "
                             "ROCm/AMD GPUs also use 'cuda')")
    parser.add_argument("--no-align", dest="align", action="store_false",
                        default=True, help="disable face detection/alignment")
    parser.add_argument("--mirror", dest="mirror", default=None, action="store_true",
                       help="mirror the camera feed (selfie view)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                       help="do not mirror the camera feed")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--flame", action="store_true",
                        help="show a FLAME head panel in the window tracking the gaze")
    parser.add_argument("--head-size", type=int, default=480,
                       help="FLAME panel edge length (default: 480 = camera "
                          "height, so compose() shows it 1:1; GL readback "
                          "cost grows with area)")
    parser.add_argument("--head-fps", type=float, default=8.0,
                        help="max FLAME head refresh rate (default: 8)")
    parser.add_argument("--headless", action="store_true",
                        help="no window; process frames and (optionally) save them")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="save annotated camera frames (and FLAME head renders) "
                           "to this directory")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = unlimited)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # FLAME 面板开启时，ROCm 核显会因 GL+HIP 共载挂死 → 自动回退 cpu（见 gpu.py）。
    device = torch.device(flame_realtime_device(args.device, args.flame))
    print("Using device:", device)

    if args.video:
        source_desc = f"video {args.video}"
        args.camera = None
    else:
        args.camera = 0 if args.camera is None else args.camera
        source_desc = f"camera {args.camera}"
    if args.mirror is None:
        args.mirror = args.camera is not None  # mirror webcam by default

    net = load_gazetr_model(args.model, device)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开{source_desc}", file=sys.stderr)
        return 1
    if args.video is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        # V4L2 默认缓存多帧; 流水线耗时>采集周期时这会让我们一直处理/显示
        # 过期画面, 压到 1 以降低延迟 (不支持的驱动上是无害的空操作)。
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mirror: FlameMirror | None = None
    if args.flame:
        if args.headless and not args.save_frames:
            print("提示: --headless 且未指定 --save-frames 时跳过 FLAME 渲染")
        else:
            mirror = FlameMirror(flame_render_device(str(device)), args.head_size, args.head_size)

    print(f"处理 {source_desc} (按 q/ESC 退出, s 保存快照)")
    print("-" * 60)

    frame_idx = 0
    fps_frames, fps_t0 = 0, time.time()
    last_head_time = 0.0
    last_fwd_end = time.perf_counter()
    last_gaze: np.ndarray | None = None
    head_img: np.ndarray | None = None
    read_failures = 0
    head_dir = Path(args.save_frames) / "head" if args.save_frames and mirror else None
    if head_dir:
        head_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("视频播放完毕")
                    break
                # webcam: transient read failures are common; retry a few times
                read_failures += 1
                if read_failures >= 10:
                    print("摄像头读取持续失败，退出")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            pitch, yaw, face_count, _prep, info = predict_gaze(
                net, frame, str(device), align=args.align
            )
            # gfx1035 核显占空比护栏: 距上一次 forward 结束的间隔补足 100ms。
            # 真实摄像头 (~10fps 天然满足) 无操作; --video 快速回放会被限速,
            # 避免背靠背满载触发 GPU Hang (见 common/gpu.py)。
            rocm_igpu_guard(device, time.perf_counter() - last_fwd_end)
            last_fwd_end = time.perf_counter()
            frame_idx += 1

            annotated = draw_overlay(frame, pitch, yaw, info, mirror=args.mirror)

            if face_count > 0:
                last_gaze = pitch_yaw_to_gaze(pitch, yaw)

            if mirror is not None and last_gaze is not None:
                now = time.time()
                if now - last_head_time >= 1.0 / max(args.head_fps, 0.1):
                    head_img = mirror.render(last_gaze)
                    last_head_time = now
                    if head_dir:
                        cv2.imwrite(
                            str(head_dir / f"frame_{frame_idx:05d}.png"), head_img
                        )

            if args.save_frames:
                out_dir = Path(args.save_frames)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), annotated)

            if not args.headless:
                composed = compose_with_head(annotated, head_img)
                cv2.imshow("GazeFlame", composed)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    snap = f"gaze_snapshot_{frame_idx:05d}.png"
                    cv2.imwrite(snap, annotated)
                    print(f"已保存快照: {snap}")

            fps_frames += 1
            if time.time() - fps_t0 >= 2.0:
                elapsed = time.time() - fps_t0
                print(
                    f"frame {frame_idx:5d} | {fps_frames / elapsed:5.1f} fps | "
                    f"pitch {np.degrees(pitch):+6.1f} deg | "
                    f"yaw {np.degrees(yaw):+6.1f} deg | face={face_count}"
                )
                fps_frames, fps_t0 = 0, time.time()

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
