"""Report presentation: real image inference, labelled replay, or full live chain.

./run_demo.sh --camera 0       RGB -> dlib -> GazeTR -> temporal -> FLAME -> intent
./run_demo.sh                  real single-image intro + synthetic rule replay
./run_demo.sh --video clip.mp4 real inference on every decoded frame
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from common.gpu import prepare_amd_gpu_env

prepare_amd_gpu_env()  # before any torch import, including delegated webcam mode

import cv2
import numpy as np

from common.assistive import AssistiveIntentState
from common.calibration import camera_gaze_to_screen_point
from common.demo import DemoDashboard, ReportReplay
from common.geometry import camera_gaze_to_pitch_yaw, camera_to_flame
from common.temporal import GazeTemporalState
from common.viz import create_display_window, draw_overlay, fit_to_display

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="GazeFlame report demo: complete webcam/video chain or guided replay. "
                    "Replay is explicitly synthetic, never a real control command.",
        epilog="Live mode also accepts run_webcam.py options (calibration, model, tracking, etc.).",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, help="full real-time webcam demo (e.g. 0)")
    src.add_argument("--video", help="full-chain inference on a video file")
    parser.add_argument("--image", default=str(ROOT / "gazetr/images/inference/test.jpg"),
                        help="real single-image intro for replay mode")
    parser.add_argument("--skip-image", action="store_true",
                        help="skip actual GazeTR intro; synthetic replay only")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "gpu", "rocm"])
    parser.add_argument("--no-flame", action="store_true", help="2D-only demo; no OpenGL required")
    parser.add_argument("--headless", action="store_true", help="export without opening a window")
    parser.add_argument("--windowed", action="store_true", help="resizable instead of fullscreen")
    parser.add_argument("--output", default=str(ROOT / "demo_outputs"),
                        help="events, stage screenshots and summary directory")
    parser.add_argument("--save-video", help="optional MP4 path (e.g. demo_outputs/demo.mp4)")
    parser.add_argument("--fps", type=float, default=15.0, help="replay/output FPS (default: 15)")
    parser.add_argument("--max-frames", type=int, default=0, help="limit frames (0: complete replay/unlimited live)")
    return parser


def save_png(path: Path, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"无法保存演示图片: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if not np.isfinite(args.fps) or not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.camera is not None or args.video:
        from run_webcam import main as webcam_main

        forwarded = ["--demo", "--device", args.device, "--events-out", str(out / "events.json"),
                     "--output-fps", str(args.fps), "--max-frames", str(args.max_frames)]
        forwarded += ["--video", args.video] if args.video else ["--camera", str(args.camera)]
        if not args.no_flame:
            forwarded.append("--flame")
        if args.headless:
            forwarded += ["--headless", "--save-frames", str(out / "frames")]
        if args.windowed:
            forwarded.append("--windowed")
        if args.save_video:
            forwarded += ["--save-video", args.save_video]
        print("完整链路: RGB -> dlib/ETH-XGaze -> GazeTR -> 3D/滤波 -> FLAME -> 辅助意图")
        print("辅助层只输出软件准备状态，不发送机械臂或飞控命令。")
        return webcam_main(forwarded + extra)

    if extra:
        parser.error("replay mode does not accept: " + " ".join(extra))
    return run_replay(args, out)


def run_replay(args, out: Path) -> int:
    dashboard = DemoDashboard()
    replay = ReportReplay()
    temporal = GazeTemporalState()
    assistive_state = AssistiveIntentState()
    mirror = None
    writer = None
    records = []
    real_result = None
    snapshot = None
    annotated = crop = None
    head = None
    intro_seconds = 0.0 if args.skip_image else 3.0
    total_seconds = intro_seconds + replay.duration
    summary = {
        "mode": "report_replay",
        "boundary": "Software concept only; no robot, flight controller or actuator commands.",
        "replay_source": "synthetic angles/EAR/reprojection error; not GazeTR predictions",
        "fps": args.fps,
        "planned_duration": total_seconds,
    }
    try:
        if not args.skip_image:
            # Import the existing real inference loader before FLAME adds flame/
            # to sys.path. Never duplicate GazeTR or alignment implementations.
            from run_webcam import load_gazetr_model, predict_gaze
            from common.gpu import flame_realtime_device

            device = flame_realtime_device(args.device, not args.no_flame)
            image_path = Path(args.image).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取演示图片: {image_path}")
            net = load_gazetr_model(str(ROOT / "gazetr/model/GazeTR-H-ETH.pt"), device)
            p, y, faces, crop, info = predict_gaze(net, image, device, align=True)
            snapshot = GazeTemporalState().update(p, y, faces, info, timestamp=0.0)
            camera_gaze = snapshot.camera_gaze
            pitch, yaw = camera_gaze_to_pitch_yaw(camera_gaze)
            snapshot.screen_point = camera_gaze_to_screen_point(pitch, yaw)
            annotated = draw_overlay(image, p, y, info, camera_gaze=camera_gaze,
                                     screen_point=snapshot.screen_point)
            real_result = {"image": str(image_path), "angle0": p, "angle1": y,
                           "face_count": faces, "camera_gaze": camera_gaze.tolist()}
            summary["real_image_inference"] = real_result
            del net

        if not args.no_flame:
            from common.flame_mirror import FlameMirror

            try:
                mirror = FlameMirror("cpu", 480, 480)
            except Exception as exc:
                raise RuntimeError("FLAME 需要可用的 OpenGL/X11 上下文；本机可用 DISPLAY=:1。"
                                   "纯2D验证请加 --no-flame。") from exc

        if args.save_video:
            path = Path(args.save_video).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (1440, 900))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建演示视频: {path}")

        display_size = None
        if not args.headless:
            display_size = create_display_window("GazeFlame Demo", fullscreen=not args.windowed)
        print("报告演示: " + ("纯合成规则回放（不运行神经网络）。" if args.skip_image else
                            "真实单图推理 + 明确标注的合成眼动回放。"))
        print("按 SPACE 暂停/继续，s 保存截图，q/ESC 退出；webcam: ./run_demo.sh --camera 0")
        last_stage = None
        last_state = None
        previous_mode = None
        frame_idx = 0
        frame = None
        while frame_idx / args.fps < total_seconds:
            tick_start = time.perf_counter()
            t = frame_idx / args.fps
            is_intro = snapshot is not None and t < intro_seconds
            mode = "real_image" if is_intro else "synthetic_replay"
            if mode != previous_mode:
                assistive_state = AssistiveIntentState()
                previous_mode = mode
            if is_intro:
                estimate = copy.deepcopy(snapshot)
                estimate.timestamp = t
                estimate.events = estimate.events if frame_idx == 0 else []
                estimate.fixation_duration = 0.0  # a still image has NO temporal evidence
                camera_gaze = np.asarray(real_result["camera_gaze"])
                driver_input, network_input = annotated, crop if estimate.valid else None
                source = "REAL IMAGE / GazeTR INFERENCE / NO TEMPORAL EVIDENCE"
                stage = "Single-image chain: input -> face alignment -> GazeTR -> vector -> FLAME"
            else:
                step, p, y, faces, info = replay.sample(t - intro_seconds)
                estimate = temporal.update(p, y, faces, info, timestamp=t)
                camera_gaze = estimate.camera_gaze
                pitch, yaw = camera_gaze_to_pitch_yaw(camera_gaze)
                estimate.screen_point = camera_gaze_to_screen_point(pitch, yaw)
                driver_input = network_input = None
                source = "SYNTHETIC REPLAY / RULE DEMONSTRATION / NOT MODEL ACCURACY"
                stage = step.label

            assistance = assistive_state.update(
                estimate.screen_point, estimate.valid, estimate.confidence,
                estimate.fixation_duration, estimate.velocity_deg_s,
                estimate.blinking, timestamp=t,
            )
            for event in estimate.events:
                records.append({**event, "frame": frame_idx, "source": mode})
            if assistance.event:
                records.append({"type": "assistive_" + assistance.event, "timestamp": t,
                                "frame": frame_idx, "source": mode, "status": assistance.status,
                                "target": assistance.target.key if assistance.target else None,
                                "local_dwell": assistance.fixation_duration})
            if mirror is not None:
                # Same thread, same scene, only poses change. No HIP/GL co-load.
                head = mirror.render(camera_to_flame(camera_gaze),
                                     head_pose=estimate.head_pose, blink=estimate.blink)
            frame = dashboard.render(
                driver_input, head, estimate, assistance, camera_gaze,
                source=source, stage=stage, crop_bgr=network_input,
                device=str(device) if is_intro else "cpu / replay",
                fps=args.fps, progress=t / total_seconds,
            )
            if stage != last_stage:
                save_png(out / f"stage_{frame_idx:05d}.png", frame)
                print(f"{t:5.1f}s | {stage}")
                last_stage = stage
            state_key = (assistance.status, assistance.target.key if assistance.target else None)
            if state_key != last_state:
                save_png(out / f"state_{frame_idx:05d}_{assistance.status.lower()}.png", frame)
                last_state = state_key
            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("GazeFlame Demo", fit_to_display(frame, display_size))
                delay = max(1, round((1 / args.fps - (time.perf_counter() - tick_start)) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key == ord(" "):
                    # Freeze replay/media time: paused wall time is not dwell evidence.
                    while True:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord(" "), ord("q"), 27):
                            break
                        if key == ord("s"):
                            save_png(out / f"snapshot_{frame_idx:05d}.png", frame)
                if key in (ord("q"), 27):
                    frame_idx += 1
                    break
                if key == ord("s"):
                    save_png(out / f"snapshot_{frame_idx:05d}.png", frame)
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
        if frame is not None:
            save_png(out / "last_frame.png", frame)
        summary["frames"] = frame_idx
        summary["statuses"] = sorted({e["status"] for e in records if "status" in e})
        return 0
    finally:
        if writer is not None:
            writer.release()
        if mirror is not None:
            mirror.close()
        if not args.headless:
            cv2.destroyAllWindows()
        (out / "events.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"演示输出: {out}")


if __name__ == "__main__":
    sys.exit(main())
