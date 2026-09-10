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
    return out
