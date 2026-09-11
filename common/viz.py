"""Shared 2D overlay drawing for the GazeFlame single-window visualization.

``draw_overlay`` annotates a BGR frame/image with the detected face box, the
6 ETH-XGaze landmarks, a gaze arrow and the pitch/yaw readout.  It is used by
both the live webcam loop (``run_webcam.py``) and the single-image pipeline
(``run_gaze_to_flame.py``) so the 2D overlay stays identical across the two.

The composed single window (input panel + FLAME head) is assembled by
``common.flame_mirror.compose_with_head``.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_overlay(
    frame: np.ndarray,
    pitch: float,
    yaw: float,
    info: dict | None = None,
    mirror: bool = False,
    show_face: bool = True,
    temporal=None,
) -> np.ndarray:
    """Draw face box, landmarks and a gaze arrow on a copy of ``frame``.

    ``info`` is the alignment dict from ``common.face`` (``rect`` /
    ``landmarks6``); pass ``None`` (or ``show_face=False``) to only draw the
    arrow + text readout.  ``mirror`` flips the arrow's horizontal component to
    match a selfiemirror feed.
    """
    out = frame.copy()

    if show_face and info:
        l, t, r, b = info["rect"]
        cv2.rectangle(out, (l, t), (r, b), (0, 255, 0), 2)
        for (x, y) in np.asarray(info["landmarks6"]).astype(int):
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

    scale = max(0.6, min(w, h) / 720.0)
    cv2.putText(
        out,
        f"pitch {np.degrees(pitch):+6.1f} deg  yaw {np.degrees(yaw):+6.1f} deg",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
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
        if temporal.screen_point is not None:
            point = np.asarray(temporal.screen_point)
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
    cv2.putText(out, label, (12, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.55, min(w, h) / 900.0), (255, 255, 255), 2)
    bar_w = max(120, int(w * 0.35))
    x0 = (w - bar_w) // 2
    y0 = h - 20
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + 8), (80, 80, 80), -1)
    cv2.rectangle(out, (x0, y0), (x0 + int(bar_w * np.clip(progress, 0.0, 1.0)), y0 + 8),
                  (0, 220, 0), -1)
    return out
