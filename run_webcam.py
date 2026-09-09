"""Real-time gaze tracking from a webcam or video file, optionally driving FLAME.

Interactive mode opens two windows:
  - ``GazeTR``: the live camera feed with face box, 6 landmarks and a gaze arrow
  - ``FLAME`` : a 3D FLAME head whose eyeballs follow the gaze in real time
    (rendered offscreen, so it also works without a GLFW-capable viewer)

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
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
GAZETR_ROOT = REPO_ROOT / "gazetr"
FLAME_ROOT = REPO_ROOT / "flame"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GAZETR_ROOT))
sys.path.insert(0, str(FLAME_ROOT))

# GazeTR (gazetr/model.py)
import model  # noqa: E402

# inference.predict_gaze (gazetr/inference.py, pulls in common.face)
from inference import predict_gaze  # noqa: E402

from common.geometry import pitch_yaw_to_gaze  # noqa: E402

DEFAULT_MODEL = str(GAZETR_ROOT / "model" / "GazeTR-H-ETH.pt")


def load_gazetr_model(model_path: str, device):
    """Load the pretrained GazeTR model and put it in eval mode."""
    net = model.Model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Camera-feed overlay
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    pitch: float,
    yaw: float,
    info: dict | None,
    mirror: bool = False,
) -> np.ndarray:
    """Draw face box, landmarks and a gaze arrow on a copy of ``frame``."""
    out = frame.copy()

    if info:
        l, t, r, b = info["rect"]
        cv2.rectangle(out, (l, t), (r, b), (0, 255, 0), 2)
        for (x, y) in info["landmarks6"].astype(int):
            cv2.circle(out, (x, y), 3, (0, 220, 0), -1)

    h, w = out.shape[:2]
    length = min(w, h) * 0.35
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    if mirror:  # selfie view: mirror the horizontal component too
        dx = -dx
    center = (w // 2, h // 2)
    end = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(out, center, end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        out,
        f"pitch {np.degrees(pitch):+6.1f} deg  yaw {np.degrees(yaw):+6.1f} deg",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return out


# ---------------------------------------------------------------------------
# FLAME mirror (offscreen renderer, created + used on the main thread)
# ---------------------------------------------------------------------------
class FlameMirror:
    """Reusable FLAME head whose eyeballs track a live gaze direction."""

    def __init__(self, device="cpu", width: int = 320, height: int = 320):
        # Imported lazily so --flame off costs nothing extra.
        import pyrender

        import gaze_viewer
        from config import FLAMEConfig
        from FLAME import FLAME

        self._gv = gaze_viewer
        self.width = width
        self.height = height

        config = FLAMEConfig()
        config.batch_size = 1
        # FLAMEConfig 默认路径相对 flame/, 转成绝对路径以免依赖 cwd。
        config.flame_model_path = str(FLAME_ROOT / config.flame_model_path)
        config.static_landmark_embedding_path = str(
            FLAME_ROOT / config.static_landmark_embedding_path
        )
        config.dynamic_landmark_embedding_path = str(
            FLAME_ROOT / config.dynamic_landmark_embedding_path
        )

        self.device = torch.device(device)
        self.flame = FLAME(config).to(self.device)
        self.shape = torch.zeros(
            1, config.shape_params, dtype=torch.float32, device=self.device
        )
        self.expression = torch.zeros(
            1, config.expression_params, dtype=torch.float32, device=self.device
        )
        self.pose = torch.zeros(
            1, config.pose_params, dtype=torch.float32, device=self.device
        )
        self.neck_pose = (
            torch.zeros(1, 3, dtype=torch.float32, device=self.device)
            if config.optimize_neckpose
            else None
        )
        self._camera_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, gaze: np.ndarray) -> np.ndarray:
        """Render the head looking along ``gaze``; returns an RGB image."""
        gv = self._gv
        target_gaze = gv.normalize(np.asarray(gaze, dtype=np.float32))

        eye_axis_angle = gv.forward_to_axis_angle(target_gaze)
        eye_pose_np = np.concatenate(
            [eye_axis_angle, eye_axis_angle], axis=0
        ).reshape(1, 6)
        eye_pose = torch.tensor(eye_pose_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            vertices, landmarks, joints = self.flame(
                self.shape, self.expression, self.pose, self.neck_pose, eye_pose
            )

        vertices_np = vertices[0].detach().cpu().numpy()
        landmarks_np = landmarks[0].detach().cpu().numpy()
        joints_np = joints[0].detach().cpu().numpy()

        left_center, right_center, face_scale = gv.estimate_eye_centers(
            landmarks_np, joints_np
        )
        if np.dot(target_gaze, (self._camera_pos - left_center)) < 0:
            target_gaze = -target_gaze

        scene = gv.build_scene(
            vertices_np, self.flame.faces, left_center, right_center,
            target_gaze, face_scale,
        )
        color, _ = self._renderer.render(scene)
        return color


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time webcam/video gaze tracking (+ optional FLAME mirror)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None,
                     help="webcam device index (default: 0)")
    src.add_argument("--video", type=str, default=None,
                     help="play a video file instead of the webcam")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GazeTR model path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="inference device (auto = cuda if available)")
    parser.add_argument("--no-align", dest="align", action="store_false",
                        default=True, help="disable face detection/alignment")
    parser.add_argument("--mirror", dest="mirror", default=None, action="store_true",
                        help="mirror the camera feed (selfie view)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="do not mirror the camera feed")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--flame", action="store_true",
                        help="open a second window with a FLAME head tracking the gaze")
    parser.add_argument("--head-size", type=int, default=640,
                        help="FLAME window edge length (default: 640; render "
                             "cost barely grows with size, only the mesh/scene "
                             "build dominates)")
    parser.add_argument("--head-fps", type=float, default=8.0,
                        help="max FLAME head refresh rate (default: 8)")
    parser.add_argument("--headless", action="store_true",
                        help="no windows; process frames and (optionally) save them")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="save annotated camera frames (and FLAME head renders) "
                             "to this directory")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = unlimited)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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

    mirror: FlameMirror | None = None
    if args.flame:
        if args.headless and not args.save_frames:
            print("提示: --headless 且未指定 --save-frames 时跳过 FLAME 渲染")
        else:
            mirror = FlameMirror(str(device), args.head_size, args.head_size)

    print(f"处理 {source_desc} (按 q/ESC 退出, s 保存快照)")
    print("-" * 60)

    frame_idx = 0
    fps_frames, fps_t0 = 0, time.time()
    last_head_time = 0.0
    last_gaze: np.ndarray | None = None
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
                    elif not args.headless:
                        cv2.imshow("FLAME Gaze", head_img)

            if args.save_frames:
                out_dir = Path(args.save_frames)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), annotated)

            if not args.headless:
                cv2.imshow("GazeTR Webcam", annotated)
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
