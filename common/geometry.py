"""Coordinate transforms shared across the GazeFlame pipeline.

GazeTR's own ``gtools.gazeto3d`` defines its two network outputs ``a0, a1``
as a direction in the usual ETH-XGaze/OpenCV camera axes::

    x = -cos(a1) * sin(a0)   # camera right
    y = -sin(a1)              # camera down
    z = -cos(a1) * cos(a0)    # camera away from the camera

The first output is therefore the horizontal angle (with the opposite sign
of a screen-right yaw), and the second output is the vertical pitch.  Keeping
this convention explicit is important: swapping X/Y after this conversion
would exchange horizontal and vertical gaze.
"""

from __future__ import annotations

import math

import numpy as np


def gazetr_angles_to_gaze(angle0: float, angle1: float) -> np.ndarray:
    """Convert the two raw GazeTR outputs into an OpenCV camera vector.

    Args:
        angle0: GazeTR horizontal output (positive points camera-left).
        angle1: GazeTR vertical output (positive points camera-up).

    Returns:
        A normalized ``[x, y, z]`` numpy vector.
    """
    x = -math.cos(angle1) * math.sin(angle0)
    y = -math.sin(angle1)
    z = -math.cos(angle1) * math.cos(angle0)
    return np.array([x, y, z], dtype=np.float64)


def pitch_yaw_to_gaze(pitch: float, yaw: float) -> np.ndarray:
    """Backward-compatible name for the GazeTR angle conversion.

    The historical project name is misleading: these positional arguments
    are the two values emitted by GazeTR, whose first value controls camera X
    and whose second value controls camera Y.  New code should use
    :func:`gazetr_angles_to_gaze` to make that fact visible at call sites.
    """
    return gazetr_angles_to_gaze(pitch, yaw)


def gazetr_to_flame(gaze: np.ndarray) -> np.ndarray:
    """Convert a camera-frame GazeTR vector into FLAME's render frame.

    OpenCV uses ``+X`` right, ``+Y`` down and ``+Z`` away from the camera;
    FLAME's viewer uses ``+X`` right, ``+Y`` up and ``+Z`` toward the camera.
    """
    return camera_to_flame(gaze)


def gaze_to_camera(
    gaze: np.ndarray,
    normalization_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a GazeTR vector to the original camera coordinate frame.

    GazeTR's vector already uses the normalized camera axes.  ETH-XGaze face
    normalization rotates the original camera into a face-centred frame.
    ``normalization_rotation`` is that camera-to-normalized rotation, so its
    transpose restores the direction before frontalization.
    """
    value = np.asarray(gaze, dtype=np.float64).reshape(-1)
    if value.size != 3 or not np.all(np.isfinite(value)):
        raise ValueError("GazeTR vector must contain three finite values.")

    camera = value.copy()
    if normalization_rotation is not None:
        rotation = np.asarray(normalization_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("normalization_rotation must be a finite 3x3 matrix.")
        camera = rotation.T @ camera
    return normalize(camera)


def camera_to_flame(gaze: np.ndarray) -> np.ndarray:
    """Convert an image-camera direction to FLAME's right/up/toward-camera frame."""
    value = normalize(gaze).reshape(3)
    # OpenCV camera coordinates are right/down/away-from-camera; FLAME uses
    # right/up/toward-camera.
    return normalize(np.array([value[0], -value[1], -value[2]], dtype=np.float64))


def camera_gaze_to_pitch_yaw(gaze: np.ndarray) -> tuple[float, float]:
    """Return screen-oriented (pitch-up, yaw-right) angles from camera gaze."""
    value = normalize(gaze).reshape(3)
    # In OpenCV camera coordinates, gaze points toward the screen along -Z.
    # Pitch is recovered from the vertical component against the horizontal
    # plane; yaw is recovered independently in the X/Z plane.  Using -Z as
    # pitch's denominator or hypot(X, Z) as yaw's denominator couples the two
    # angles and is not the inverse of gazetr_angles_to_gaze().
    horizontal_forward = float(np.hypot(value[0], value[2]))
    pitch = math.atan2(float(-value[1]), horizontal_forward)
    yaw = math.atan2(float(value[0]), float(-value[2]))
    return float(pitch), float(yaw)


def gaze_to_dict(gaze: np.ndarray) -> dict[str, float]:
    """Return the gaze vector as ``{"x": ..., "y": ..., "z": ...}`` floats."""
    return {"x": float(gaze[0]), "y": float(gaze[1]), "z": float(gaze[2])}


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ``v`` normalized; raise if its norm is effectively zero."""
    value = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("Vector contains NaN or infinity, cannot normalize.")
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("Vector norm is zero, cannot normalize.")
    return value / norm


def angular_error(gaze1: np.ndarray, gaze2: np.ndarray) -> float:
    """Angular error in degrees between two 3D gaze vectors."""
    v1 = normalize(gaze1)
    v2 = normalize(gaze2)
    cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
