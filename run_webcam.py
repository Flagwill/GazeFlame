"""Real-time gaze tracking from a webcam or video file, optionally driving FLAME.

Interactive mode opens a single window showing the live camera feed with face
box / landmarks / gaze arrow, an optional side-by-side FLAME head, and an
optional software-only assistive-control concept panel.  The FLAME panel is
rendered offscreen, so it also works without a GLFW-capable viewer.

Controls: ``q`` / ``ESC`` quit, ``s`` save a snapshot, ``c`` calibrate live.

Examples:
    python run_webcam.py --camera 0
    python run_webcam.py --camera 0 --flame --assistive
    python run_webcam.py --video clip.mp4 --flame --no-align
    python run_webcam.py --video clip.mp4 --flame --assistive --save-frames out/
    python run_webcam.py --camera 0 --headless --save-frames out/  # automation
    python run_webcam.py --camera 0 --calibrate profiles/me.json
    python run_webcam.py --camera 0 --calibration-file profiles/me.json \
        --events-out events.json

Notes:
    - dlib face detection is run on a downscaled frame internally (common.face),
      so a 640x480 webcam costs ~30ms/frame instead of ~450ms.
    - The FLAME head is rendered in the main loop at a capped rate
      (``--head-fps``); pyrender's OpenGL context is bound to the creating
      thread, so it cannot live on a separate thread.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GAZETR_ROOT = REPO_ROOT / "gazetr"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GAZETR_ROOT))

# 必须早于 import torch (ROCm: HSA 覆盖 + 抢注系统 libdrm; CPU/CUDA build 无副作用)。
from common.gpu import (  # noqa: E402
    autocast_gpu,
    flame_realtime_device,
    flame_render_device,
    fp16_autocast_enabled,
    is_gpu_runtime_error,
    prepare_amd_gpu_env,
    prepare_rocm_display_env,
    rocm_igpu_guard,
    synchronize_device,
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
from common.assistive import (  # noqa: E402
    AssistiveIntentState,
    compose_with_assistive,
    draw_assistive_panel,
)
from common.calibration import (  # noqa: E402
    CalibrationProfile,
    CalibrationSession,
    camera_gaze_to_screen_point,
)
from common.face import DEFAULT_FOV_DEG, FaceTracker, mirrored_face_info  # noqa: E402
from common.geometry import (  # noqa: E402
    camera_gaze_to_pitch_yaw,
    camera_to_flame,
    gaze_to_camera,
    gazetr_angles_to_gaze,
)
from common.temporal import GazeTemporalState  # noqa: E402
from common.demo import DemoDashboard  # noqa: E402
from common.viz import (  # noqa: E402
    WINDOW_TITLE,
    create_display_window,
    draw_calibration_target,
    draw_overlay,
    draw_screen_point,
    fit_to_display,
)

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
        # 避免实时流水线第一帧明显变慢。用与 predict_gaze 相同的精度,
        # 让 fp16 的 kernel 选择也在首帧前完成。
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.inference_mode(), autocast_gpu(device):
            for _ in range(3):
                net({"face": dummy})
        synchronize_device(device)
    return net


def measure_forward_ms(net, device, repeats: int = 3) -> float:
    """Time a fixed 1x3x224x224 forward (GPU health probe).

    A ROCm gfx1035 GPU can silently degrade after a VM fault (observed: the
    same batch-16 forward going 105ms -> 705ms) instead of failing loudly.
    The live loop uses this to compare against the freshly-loaded baseline and
    drop to CPU when the accelerator is no longer worth using.
    """
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.inference_mode(), autocast_gpu(device):
        net({"face": dummy})
        synchronize_device(device)
        start = time.perf_counter()
        for _ in range(repeats):
            net({"face": dummy})
        synchronize_device(device)
    return (time.perf_counter() - start) / repeats * 1000.0


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
    parser.add_argument("--camera-fov", type=float, default=DEFAULT_FOV_DEG,
                        help="source camera horizontal FOV in degrees for solvePnP "
                             f"(default: {DEFAULT_FOV_DEG:g})")
    parser.add_argument("--face-detect-interval", type=int, default=6,
                        help="full-frame face detection interval in frames "
                             "(default: 6; tracking handles frames between scans)")
    parser.add_argument("--face-track-frames", type=int, default=30,
                        help="maximum frames to bridge after a detection miss "
                             "(default: 30)")
    parser.add_argument("--mirror", dest="mirror", default=None, action="store_true",
                       help="mirror the camera feed (selfie view)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                       help="do not mirror the camera feed")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--flame", action="store_true",
                        help="show a FLAME head panel in the window tracking the gaze")
    parser.add_argument("--assistive", action="store_true",
                        help="show a software-only assistive-control concept panel; "
                             "combine with --flame for the full three-panel view")
    parser.add_argument("--demo", action="store_true",
                        help="report-demo dashboard with full-chain evidence and "
                             "assistive states (implies --assistive)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="export the demo dashboard to an MP4 (requires --demo)")
    parser.add_argument("--output-fps", type=float, default=15.0,
                        help="exported webcam video FPS (default: 15)")
    parser.add_argument("--hide-gaze-arrows", action="store_true",
                        help="hide debug arrows in the FLAME eye panel")
    parser.add_argument("--head-size", type=int, default=480,
                       help="FLAME panel edge length (default: 480 = camera "
                          "height, so compose() shows it 1:1; GL readback "
                          "cost grows with area)")
    parser.add_argument("--head-fps", type=float, default=8.0,
                        help="max FLAME head refresh rate (default: 8)")
    parser.add_argument("--headless", action="store_true",
                        help="no window; process frames and (optionally) save them")
    parser.add_argument("--windowed", action="store_true",
                        help="keep the visualization window resizable instead of fullscreen")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="save annotated camera frames (and FLAME head renders) "
                           "to this directory")
    parser.add_argument("--calibration-file", type=str, default=None,
                        help="load an affine personal gaze calibration JSON")
    parser.add_argument("--calibrate", type=str, default=None, metavar="PATH",
                        help="run timed 9-point calibration and save the profile")
    parser.add_argument("--calibration-settle", type=float, default=0.7,
                        help="seconds to wait before collecting each calibration point")
    parser.add_argument("--calibration-hold", type=float, default=1.5,
                        help="seconds to collect each calibration point")
    parser.add_argument("--gaze-min-cutoff", type=float, default=0.5,
                        help="stationary gaze smoothing cutoff in Hz (default: 0.5)")
    parser.add_argument("--gaze-beta", type=float, default=1.2,
                        help="adaptive response to gaze motion in radians (default: 1.2)")
    parser.add_argument("--gaze-spike-deg", type=float, default=12.0,
                        help="confirm larger one-frame gaze jumps (default: 12 degrees)")
    parser.add_argument("--events-out", type=str, default=None,
                        help="write detected gaze events as a JSON array")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = unlimited)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo:
        args.assistive = True
    if args.save_video and not args.demo:
        parser.error("--save-video requires --demo")
    if not np.isfinite(args.output_fps) or args.output_fps <= 0:
        parser.error("--output-fps must be positive and finite")
    for option, value in (("--gaze-min-cutoff", args.gaze_min_cutoff),
                          ("--gaze-spike-deg", args.gaze_spike_deg),
                          ("--calibration-hold", args.calibration_hold)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{option} must be positive and finite")
    for option, value in (("--gaze-beta", args.gaze_beta),
                          ("--calibration-settle", args.calibration_settle)):
        if not np.isfinite(value) or value < 0:
            parser.error(f"{option} must be nonnegative and finite")

    if args.calibrate and args.headless:
        raise SystemExit("--calibrate requires a visible window")
    if args.calibrate and args.calibration_file:
        raise SystemExit("use either --calibrate or --calibration-file, not both")

    profile = None
    if args.calibration_file:
        loaded_profile = CalibrationProfile.load(args.calibration_file)
        if loaded_profile.is_usable():
            profile = loaded_profile
            print(
                f"Loaded calibration: {args.calibration_file} "
                f"(RMSE={profile.rmse:.4f})"
            )
        else:
            print(
                f"警告: 校准文件质量不合格 (RMSE={loaded_profile.rmse:.4f}), "
                "忽略该文件；请重新运行 --calibrate。",
                file=sys.stderr,
            )
    calibration = (
        CalibrationSession(
            settle_seconds=args.calibration_settle,
            hold_seconds=args.calibration_hold,
        )
        if args.calibrate
        else None
    )

    # 无 --flame 时 ROCm auto 保持 GPU；--flame 的核显 GL+HIP 共载由 gpu.py 保护。
    device = torch.device(flame_realtime_device(args.device, args.flame))
    software_display = prepare_rocm_display_env(
        str(device), args.headless, args.flame
    )
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    if fp16_autocast_enabled(device):
        print("提示: 已启用 fp16 autocast 推理 (约 1.3-1.7x; GAZE_ROCM_FP16=0 关闭)")
    if software_display:
        print("提示: ROCm GUI 使用软件显示光栅化，GazeTR 仍使用 GPU")

    if args.video:
        source_desc = f"video {args.video}"
        args.camera = None
    else:
        args.camera = 0 if args.camera is None else args.camera
        source_desc = f"camera {args.camera}"
    if args.mirror is None:
        args.mirror = args.camera is not None  # mirror webcam by default

    net = load_gazetr_model(args.model, device)
    gpu_baseline_ms = 0.0
    if device.type == "cuda":
        try:
            gpu_baseline_ms = measure_forward_ms(net, device)
            print(f"GPU 前向基线: {gpu_baseline_ms:.1f} ms/帧")
        except Exception as exc:  # noqa: BLE001 - probe must not block startup
            print(f"提示: GPU 健康基线测量失败: {exc}", file=sys.stderr)

    face_tracker = (
        FaceTracker(
            detect_interval=args.face_detect_interval,
            max_tracking_frames=args.face_track_frames,
        )
        if args.align
        else None
    )

    def fallback_to_cpu(reason: str) -> None:
        """Move inference to CPU after GPU faults/degradation; survives in-process."""
        nonlocal device, net
        print(f"警告: {reason}，切换到 CPU 推理以保持会话", file=sys.stderr)
        device = torch.device("cpu")
        net = load_gazetr_model(args.model, device)
        if face_tracker is not None:
            face_tracker.reset()

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

    video_fps = float(cap.get(cv2.CAP_PROP_FPS)) if args.video else args.output_fps
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    dashboard = DemoDashboard() if args.demo else None
    video_writer = None

    mirror: FlameMirror | None = None
    if args.flame:
        if args.headless and not args.save_frames and not args.save_video:
            print("提示: --headless 且未指定保存路径时跳过 FLAME 渲染")
        else:
            mirror = FlameMirror(
                flame_render_device(str(device)), args.head_size, args.head_size,
                show_arrows=not args.hide_gaze_arrows,
            )

    print(f"处理 {source_desc} (按 q/ESC 退出, s 保存快照, c 在当前窗口校准)")
    print("-" * 60)

    if not args.headless:
        display_size = create_display_window(
            WINDOW_TITLE, fullscreen=not args.windowed
        )
    else:
        display_size = None

    frame_idx = 0
    fps_frames, fps_t0 = 0, time.time()
    last_head_time = 0.0
    last_fwd_end = time.perf_counter()
    last_gaze: np.ndarray | None = None
    last_head_pose: np.ndarray | None = None
    head_img: np.ndarray | None = None
    read_failures = 0
    temporal = GazeTemporalState(
        min_cutoff=args.gaze_min_cutoff, beta=args.gaze_beta,
        spike_threshold_deg=args.gaze_spike_deg,
    )
    assistive_state = AssistiveIntentState() if args.assistive else None
    detected_events: list[dict] = []
    last_raw_pitch, last_raw_yaw = 0.0, 0.0
    inference_errors = 0
    render_errors = 0
    consecutive_inference_errors = 0
    last_inference_error_time = 0.0
    last_render_error_time = 0.0
    last_health_time = time.perf_counter()
    degraded_probes = 0
    head_dir = Path(args.save_frames) / "head" if args.save_frames and mirror else None
    if head_dir:
        head_dir.mkdir(parents=True, exist_ok=True)

    playback_t0 = time.perf_counter()
    fps_t0 = time.time()
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

            # Selfie mirroring is a display transform.  Keep model input in
            # the camera's native orientation to avoid a left/right bias.
            camera_frame = frame
            display_frame = (
                cv2.flip(camera_frame, 1) if args.mirror else camera_frame
            )

            try:
                raw_pitch, raw_yaw, face_count, _prep, info = predict_gaze(
                    net,
                    camera_frame,
                    str(device),
                    align=args.align,
                    face_tracker=face_tracker,
                    fov_deg=args.camera_fov,
                )
                last_raw_pitch, last_raw_yaw = raw_pitch, raw_yaw
                consecutive_inference_errors = 0
            except Exception as exc:  # noqa: BLE001 - a bad frame must not end a live stream
                inference_errors += 1
                consecutive_inference_errors += 1
                raw_pitch, raw_yaw = last_raw_pitch, last_raw_yaw
                face_count, info = 0, None
                now = time.time()
                if now - last_inference_error_time >= 2.0:
                    print(f"警告: 当前帧推理失败，按丢脸处理: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    last_inference_error_time = now
                # A dead/erroring HIP context does not recover in-process, so a
                # long session would otherwise keep reusing the last good angle
                # forever.  Reload on CPU and keep producing gaze results.
                if (consecutive_inference_errors >= 3 and device.type == "cuda"
                        and is_gpu_runtime_error(exc)):
                    fallback_to_cpu("GPU 推理连续失败")
                    consecutive_inference_errors = 0
            # gfx1035 核显占空比护栏: 距上一次 forward 结束的间隔补足 100ms。
            # 真实摄像头 (~6fps 以上时可能限速) 仍使用 GPU；--video 快速回放会被限速,
            # 避免背靠背满载触发 GPU Hang (见 common/gpu.py)。
            rocm_igpu_guard(device, time.perf_counter() - last_fwd_end)
            last_fwd_end = time.perf_counter()

            # 周期健康探针: ROCm 核显在 VM fault 后可能进入"无异常但极慢"的
            # 劣化态 (实测 b16 105ms→705ms)。对比刚加载时的基线, 连续两次远
            # 超阈值就切到 CPU, 避免表面正常却慢几倍地空转。
            if device.type == "cuda" and gpu_baseline_ms > 0:
                if time.perf_counter() - last_health_time >= 10.0:
                    last_health_time = time.perf_counter()
                    try:
                        probe_ms = measure_forward_ms(net, device)
                    except Exception:  # noqa: BLE001 - a fault here just triggers fallback
                        probe_ms = None
                    slow_limit = max(gpu_baseline_ms * 6.0, gpu_baseline_ms + 100.0)
                    if probe_ms is not None and probe_ms > slow_limit:
                        degraded_probes += 1
                        print(
                            f"警告: GPU 前向 {probe_ms:.0f}ms 远高于基线 "
                            f"{gpu_baseline_ms:.0f}ms, 疑似劣化 "
                            f"({degraded_probes}/2)",
                            file=sys.stderr,
                        )
                        if degraded_probes >= 2:
                            fallback_to_cpu("GPU 持续劣化")
                    else:
                        degraded_probes = 0
            frame_idx += 1

            # File playback uses the media clock, not CPU processing speed.
            # Otherwise headless/fast decoding invents or loses dwell evidence.
            now = ((frame_idx - 1) / video_fps if args.video else time.perf_counter())
            estimate = temporal.update(
                raw_pitch,
                raw_yaw,
                face_count=face_count,
                info=info,
                timestamp=now,
            )
            if calibration is not None and not calibration.finished:
                calibration_pitch, calibration_yaw = 0.0, 0.0
                if estimate.valid and estimate.measurement_valid:
                    raw_camera_gaze = gaze_to_camera(
                        gazetr_angles_to_gaze(raw_pitch, raw_yaw),
                        info.get("normalization_rotation"),
                    )
                    calibration_pitch, calibration_yaw = camera_gaze_to_pitch_yaw(raw_camera_gaze)
                new_profile = calibration.update(
                    calibration_pitch,
                    calibration_yaw,
                    estimate.valid and estimate.measurement_valid,
                    timestamp=now,
                    confidence=estimate.confidence,
                    blinking=estimate.blinking,
                    velocity_deg_s=estimate.velocity_deg_s,
                )
                if new_profile is not None:
                    if new_profile.is_usable():
                        new_profile.save(args.calibrate)
                        profile = new_profile
                        print(f"Calibration saved: {args.calibrate} "
                              f"(RMSE={new_profile.rmse:.4f}, "
                              f"held-out RMSE={new_profile.validation_rmse:.4f})")
                    else:
                        print("警告: 校准未通过质量检查，保留原校准文件；按 c 重试。", file=sys.stderr)

            # Temporal state already restores and filters the complete camera
            # direction. Applying a fresh R.T here would reintroduce pose jitter.
            filtered_camera_gaze = estimate.camera_gaze
            screen_pitch, screen_yaw = camera_gaze_to_pitch_yaw(
                filtered_camera_gaze
            )
            if profile is not None:
                # Calibration maps camera-frame angles to display coordinates;
                # it does not alter the physical vector sent to FLAME.
                screen_point = profile.screen_point(screen_pitch, screen_yaw)
            else:
                screen_point = camera_gaze_to_screen_point(
                    screen_pitch, screen_yaw, mirror=args.mirror
                )
            estimate.screen_point = screen_point
            assistive_estimate = None
            if assistive_state is not None:
                assistive_estimate = assistive_state.update(
                    screen_point,
                    valid=estimate.valid,
                    confidence=estimate.confidence,
                    fixation_duration=estimate.fixation_duration,
                    velocity_deg_s=estimate.velocity_deg_s,
                    blinking=estimate.blinking,
                    timestamp=now,
                )
                if assistive_estimate.event:
                    event = {
                        "type": f"assistive_{assistive_estimate.event}",
                        "frame": frame_idx,
                        "timestamp": estimate.timestamp,
                        "target": (
                            assistive_estimate.target.key
                            if assistive_estimate.target is not None else None
                        ),
                        "status": assistive_estimate.status,
                    }
                    detected_events.append(event)
                    print(f"event: {event['type']}")
            if estimate.events:
                for event in estimate.events:
                    event["frame"] = frame_idx
                    detected_events.append(event)
                    print(f"event: {event['type']}")

            overlay_info = (
                mirrored_face_info(info, camera_frame.shape[1])
                if args.mirror and info is not None
                else info
            )
            annotated = draw_overlay(
                display_frame,
                estimate.pitch,
                estimate.yaw,
                overlay_info,
                mirror=args.mirror,
                temporal=estimate,
                screen_point=screen_point,
                camera_gaze=filtered_camera_gaze,
            )
            if estimate.valid:
                last_gaze = camera_to_flame(filtered_camera_gaze)
                # ``head_pose`` in ``info`` is the current solvePnP rotation.
                # Do not average Rodrigues-vector components for the render:
                # rotations live on SO(3), not in a Euclidean coordinate box.
                last_head_pose = info.get("head_pose") if info is not None else None

            if mirror is not None and last_gaze is not None:
                now = time.time()
                if now - last_head_time >= 1.0 / max(args.head_fps, 0.1):
                    try:
                        head_img = mirror.render(
                            last_gaze,
                            head_pose=last_head_pose,
                            blink=estimate.blink,
                        )
                        last_head_time = now
                        if head_dir:
                            cv2.imwrite(
                                str(head_dir / f"frame_{frame_idx:05d}.png"), head_img
                            )
                    except Exception as exc:  # noqa: BLE001 - keep camera loop alive
                        render_errors += 1
                        last_head_time = now
                        if now - last_render_error_time >= 2.0:
                            print(f"警告: FLAME 当前帧渲染失败，保留上一帧: {exc}",
                                  file=sys.stderr)
                            traceback.print_exc()
                            last_render_error_time = now

            composed = None
            if dashboard is not None:
                measured_fps = frame_idx / max(time.perf_counter() - playback_t0, 1e-3)
                composed = dashboard.render(
                    annotated, head_img, estimate, assistive_estimate,
                    filtered_camera_gaze,
                    source="VIDEO / REAL GazeTR INFERENCE" if args.video else
                           "WEBCAM / LIVE GazeTR INFERENCE",
                    stage=(calibration.status if calibration is not None else
                           "Driver perception -> cockpit attention -> intent -> assistance readiness"),
                    device=str(device), crop_bgr=_prep if estimate.valid else None,
                    fps=measured_fps, calibrated=profile is not None,
                )
                if args.save_video:
                    if video_writer is None:
                        video_path = Path(args.save_video)
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        video_writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps, (composed.shape[1], composed.shape[0]),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"无法创建演示视频: {video_path}")
                    video_writer.write(composed)
            elif args.assistive:
                assistive_panel = draw_assistive_panel(
                    annotated.shape[0], assistive_estimate
                )
                annotated = draw_screen_point(
                    annotated,
                    estimate.screen_point if estimate.valid else None,
                )
                composed = compose_with_assistive(
                    annotated, head_img, assistive_panel
                )

            if args.save_frames:
                out_dir = Path(args.save_frames)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), annotated)
                if composed is not None:
                    assistive_dir = out_dir / "assistive"
                    assistive_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(assistive_dir / f"frame_{frame_idx:05d}.png"), composed
                    )

            if not args.headless:
                if composed is None:
                    composed = compose_with_head(annotated, head_img)
                if not args.assistive:
                    composed = draw_screen_point(
                        composed, estimate.screen_point if estimate.valid else None
                    )
                if calibration is not None and not calibration.finished:
                    composed = draw_calibration_target(
                        composed,
                        calibration.target,
                        calibration.progress,
                        label=f"CALIBRATION {calibration.index + 1}/{len(calibration.points)}",
                        status=calibration.status,
                    )
                cv2.imshow(WINDOW_TITLE, fit_to_display(composed, display_size))
                delay = 1
                if args.demo and args.video:
                    deadline = playback_t0 + frame_idx / video_fps
                    delay = max(1, round((deadline - time.perf_counter()) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    snap = f"gaze_snapshot_{frame_idx:05d}.png"
                    cv2.imwrite(snap, composed if args.demo else annotated)
                    print(f"已保存快照: {snap}")
                if key == ord("c") and not args.video:
                    args.calibrate = (args.calibrate or args.calibration_file or
                                      str(REPO_ROOT / "profiles/demo_user.json"))
                    calibration = CalibrationSession(
                        settle_seconds=args.calibration_settle,
                        hold_seconds=args.calibration_hold,
                    )
                    print(f"开始当前窗口九点校准，合格后保存至 {args.calibrate}")

            fps_frames += 1
            if time.time() - fps_t0 >= 2.0:
                elapsed = time.time() - fps_t0
                print(
                    f"frame {frame_idx:5d} | {fps_frames / elapsed:5.1f} fps | "
                    f"pitch {np.degrees(screen_pitch):+6.1f} deg | "
                    f"yaw {np.degrees(screen_yaw):+6.1f} deg | face={face_count}"
                )
                fps_frames, fps_t0 = 0, time.time()

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if mirror is not None:
            try:
                mirror.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must stay best effort
                print(f"警告: 释放 FLAME renderer 失败: {exc}", file=sys.stderr)
        if inference_errors or render_errors:
            print(
                f"运行期间跳过异常帧: inference={inference_errors}, "
                f"flame_render={render_errors}",
                file=sys.stderr,
            )
        if args.events_out:
            events_path = Path(args.events_out)
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.write_text(
                json.dumps(detected_events, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Events saved: {args.events_out}")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
