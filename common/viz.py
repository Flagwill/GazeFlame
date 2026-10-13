"""Shared 2D overlay drawing for the GazeFlame single-window visualization.

``draw_overlay`` annotates a BGR frame/image with the detected face box, the
6 ETH-XGaze landmarks, a gaze arrow, the estimated screen point and the
pitch/yaw readout.  It is used by both the live webcam loop
(``run_webcam.py``) and the single-image pipeline (``run_gaze_to_flame.py``)
so the 2D overlay stays identical across the two.

The composed single window (input panel + FLAME head) is assembled by
``common.flame_mirror.compose_with_head``.
"""

from __future__ import annotations

import re
import subprocess

import cv2
import numpy as np

from common.geometry import camera_gaze_to_pitch_yaw


WINDOW_TITLE = "GazeFlame"


def create_display_window(
    window_title: str = WINDOW_TITLE,
    fullscreen: bool = True,
) -> tuple[int, int] | None:
    """Create the shared HighGUI window and return its fullscreen size.

    ``WINDOW_FULLSCREEN`` changes the window bounds, but HighGUI does not
    necessarily scale an image that is smaller than those bounds.  Callers
    use the returned size with :func:`fit_to_display` before ``imshow``.
    """
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    if not fullscreen:
        return None

    cv2.setWindowProperty(
        window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    # Let the backend apply the fullscreen state before reading its geometry.
    cv2.waitKey(1)
    display_size = _query_display_size(window_title)
    if display_size is None:
        # Some backends only create a measurable image area after an imshow.
        cv2.imshow(window_title, np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        display_size = _query_display_size(window_title)
    return display_size


def _query_display_size(window_title: str) -> tuple[int, int] | None:
    """Find the active display size without making it a runtime dependency."""
    # xrandr reports the monitor containing the primary fullscreen desktop and
    # is available on the X11 systems supported by the existing GUI path.
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
        lines = result.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        lines = []

    connected_sizes: list[tuple[bool, int, int]] = []
    for line in lines:
        match = re.match(
            r"^\S+\s+connected(?:\s+(primary))?\s+(\d+)x(\d+)\+",
            line,
        )
        if match:
            connected_sizes.append(
                (
                    match.group(1) is not None,
                    int(match.group(2)),
                    int(match.group(3)),
                )
            )
    if connected_sizes:
        primary = next(
            (
                (width, height)
                for is_primary, width, height in connected_sizes
                if is_primary
            ),
            None,
        )
        if primary is not None:
            return primary
        return connected_sizes[0][1], connected_sizes[0][2]

    current = re.search(
        r"^Screen \d+:.*current (\d+) x (\d+),",
        "\n".join(lines),
        re.MULTILINE,
    )
    if current:
        return int(current.group(1)), int(current.group(2))

    # This fallback also supports non-X11 HighGUI backends that expose the
    # fullscreen image rectangle after the first event-loop pass.
    try:
        _, _, width, height = cv2.getWindowImageRect(window_title)
    except cv2.error:
        return None
    if width > 1 and height > 1:
        return int(width), int(height)
    return None


def fit_to_display(
    image: np.ndarray,
    display_size: tuple[int, int] | None,
) -> np.ndarray:
    """Scale an image to fill the display bounds while preserving its aspect.

    The result always has the display dimensions when they are known.  Black
    bars are used only where the source and monitor aspect ratios differ, so
    faces and the FLAME head are not distorted by fullscreen presentation.
    """
    if display_size is None or image.ndim < 2:
        return image
    display_w, display_h = (int(display_size[0]), int(display_size[1]))
    image_h, image_w = image.shape[:2]
    if min(display_w, display_h, image_w, image_h) <= 0:
        return image

    scale = min(display_w / float(image_w), display_h / float(image_h))
    target_w = max(1, min(display_w, int(round(image_w * scale))))
    target_h = max(1, min(display_h, int(round(image_h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    if target_w == display_w and target_h == display_h:
        return resized
    canvas_shape = (display_h, display_w, *image.shape[2:])
    canvas = np.zeros(canvas_shape, dtype=image.dtype)
    x0 = (display_w - target_w) // 2
    y0 = (display_h - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = resized
    return canvas


def draw_screen_point(
    frame: np.ndarray,
    point: np.ndarray | tuple[float, float] | None,
) -> np.ndarray:
    """Draw a normalized gaze point as a high-contrast crosshair."""
    if point is None:
        return frame
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size < 2 or not np.all(np.isfinite(value[:2])):
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    center = (
        int(np.clip(value[0], 0.0, 1.0) * max(w - 1, 0)),
        int(np.clip(value[1], 0.0, 1.0) * max(h - 1, 0)),
    )
    radius = max(10, int(min(w, h) * 0.025))
    thickness = max(2, int(radius / 6))

    # Black edging keeps the marker readable over both the face and bright
    # backgrounds; the red center matches the existing gaze arrow.
    cv2.circle(out, center, radius + 4, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.circle(out, center, radius, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.drawMarker(
        out, center, (255, 255, 255), cv2.MARKER_CROSS,
        radius * 2 + 12, thickness, cv2.LINE_AA,
    )
    return out


def draw_overlay(
    frame: np.ndarray,
    pitch: float,
    yaw: float,
    info: dict | None = None,
    mirror: bool = False,
    show_face: bool = True,
    temporal=None,
    screen_point: np.ndarray | tuple[float, float] | None = None,
    camera_gaze: np.ndarray | None = None,
) -> np.ndarray:
    """Draw face box, landmarks and a gaze arrow on a copy of ``frame``.

    ``info`` is the alignment dict from ``common.face`` (``rect`` /
    ``landmarks6``); pass ``None`` (or ``show_face=False``) to only draw the
    arrow + text readout.  ``mirror`` flips the arrow's horizontal component to
    match a selfie mirror feed. ``camera_gaze`` optionally supplies the
    head-compensated direction in the original camera frame. ``screen_point``
    is used for the normalized coordinate readout; the corresponding
    full-window marker is drawn by ``draw_screen_point`` after panel
    composition.
    """
    out = frame.copy()

    if show_face and info:
        l, t, r, b = info["rect"]
        cv2.rectangle(out, (l, t), (r, b), (0, 255, 0), 2)
        for (x, y) in np.asarray(info["landmarks6"]).astype(int):
            cv2.circle(out, (x, y), 3, (0, 220, 0), -1)

    h, w = out.shape[:2]
    length = min(w, h) * 0.35
    readout_pitch, readout_yaw = float(pitch), float(yaw)
    if camera_gaze is not None:
        arrow_pitch, arrow_yaw = camera_gaze_to_pitch_yaw(camera_gaze)
        readout_pitch, readout_yaw = arrow_pitch, arrow_yaw
        dx = length * np.sin(arrow_yaw) * np.cos(arrow_pitch)
        dy = -length * np.sin(arrow_pitch)
    else:
        dx = -length * np.sin(yaw) * np.cos(pitch)
        dy = -length * np.sin(pitch)
    if mirror:  # selfie view: mirror the horizontal component too
        dx = -dx
    center = (w // 2, h // 2)
    end = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(out, center, end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)

    display_point = screen_point
    if temporal is not None:
        if not temporal.valid:
            display_point = None
        elif display_point is None:
            display_point = temporal.screen_point

    scale = max(0.6, min(w, h) / 720.0)
    readout = (f"pitch {np.degrees(readout_pitch):+6.1f} deg  "
               f"yaw {np.degrees(readout_yaw):+6.1f} deg")
    text_width = cv2.getTextSize(readout, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
    readout_scale = scale * min(1.0, max(1, w - 20) / max(text_width, 1))
    cv2.putText(
        out, readout, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, readout_scale,
        (255, 255, 255), 1 if readout_scale < 0.5 else 2,
    )
    if temporal is not None:
        status = "face" if temporal.valid else "no face"
        if temporal.blinking:
            status += " | blink"
        if temporal.event:
            status += f" | {temporal.event}"
        cv2.putText(
            out,
            f"{status}  conf {temporal.confidence:.2f}  speed {temporal.velocity_deg_s:4.0f} deg/s",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, scale * 0.72),
            (0, 255, 255),
            2,
        )
        if temporal.head_euler is not None:
            pose_deg = np.degrees(np.asarray(temporal.head_euler, dtype=np.float64))
            cv2.putText(
                out,
                f"head x/y/z {pose_deg[0]:+4.0f} {pose_deg[1]:+4.0f} {pose_deg[2]:+4.0f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, scale * 0.68),
                (255, 220, 0),
                2,
            )
    if display_point is not None:
        point = np.asarray(display_point)
        cv2.putText(
            out,
            f"screen {point[0]:.2f}, {point[1]:.2f}",
            (10, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, scale * 0.68),
            (255, 220, 0),
            2,
        )
    return out


def draw_calibration_target(
    frame: np.ndarray,
    point: tuple[float, float],
    progress: float = 0.0,
    label: str = "CALIBRATION",
    status: str | None = None,
) -> np.ndarray:
    """Draw a high-contrast timed calibration target over a camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    center = (int(np.clip(point[0], 0.0, 1.0) * (w - 1)),
              int(np.clip(point[1], 0.0, 1.0) * (h - 1)))
    radius = max(16, int(min(w, h) * 0.035))
    cv2.circle(out, center, radius + 8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, center, radius, (0, 70, 255), -1, cv2.LINE_AA)
    cv2.circle(out, center, max(3, radius // 5), (255, 255, 255), -1, cv2.LINE_AA)
    for text, y in ((label, h - (66 if status else 42)), (status, h - 42)):
        if text is None:
            continue
        scale = max(0.55, min(w, h) / 900.0)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1.0, max(1, w - 24) / max(text_width, 1))
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    bar_w = max(120, int(w * 0.35))
    x0 = (w - bar_w) // 2
    y0 = h - 20
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + 8), (80, 80, 80), -1)
    cv2.rectangle(out, (x0, y0), (x0 + int(bar_w * np.clip(progress, 0.0, 1.0)), y0 + 8),
                  (0, 220, 0), -1)
    return out
