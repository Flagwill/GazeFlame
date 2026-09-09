"""Gaze geometry helpers shared across the GazeFlame pipeline.

The GazeTR convention maps a 2D (pitch, yaw) pair to a normalized 3D
direction vector:

    x = -cos(yaw) * sin(pitch)
    y = -sin(yaw)
    z = -cos(yaw) * cos(pitch)

This module is the single source of truth for that conversion so the
orchestrator and any downstream consumer never duplicate the math.
"""

from __future__ import annotations

import math

import numpy as np


def pitch_yaw_to_gaze(pitch: float, yaw: float) -> np.ndarray:
    """Convert gaze pitch/yaw (radians) into a normalized 3D direction vector.

    Args:
        pitch: Pitch angle in radians (positive = looking up).
        yaw: Yaw angle in radians (positive = looking right).

    Returns:
        A normalized ``[x, y, z]`` numpy vector.
    """
    x = -math.cos(yaw) * math.sin(pitch)
    y = -math.sin(yaw)
    z = -math.cos(yaw) * math.cos(pitch)
    return np.array([x, y, z], dtype=np.float64)


def gaze_to_dict(gaze: np.ndarray) -> dict[str, float]:
    """Return the gaze vector as ``{"x": ..., "y": ..., "z": ...}`` floats."""
    return {"x": float(gaze[0]), "y": float(gaze[1]), "z": float(gaze[2])}


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ``v`` normalized; raise if its norm is effectively zero."""
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("Vector norm is zero, cannot normalize.")
    return v / norm


def angular_error(gaze1: np.ndarray, gaze2: np.ndarray) -> float:
    """Angular error in degrees between two 3D gaze vectors."""
    v1 = normalize(gaze1)
    v2 = normalize(gaze2)
    cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
