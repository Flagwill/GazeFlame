"""Small, persistent personal gaze calibration profile."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CALIBRATION_POINTS = tuple(
    (x, y) for y in (0.15, 0.5, 0.85) for x in (0.15, 0.5, 0.85)
)


@dataclass
class CalibrationProfile:
    """Affine map from model angles to normalized screen coordinates."""

    matrix: np.ndarray
    max_pitch_deg: float = 30.0
    max_yaw_deg: float = 40.0
    rmse: float = 0.0

    def screen_point(self, pitch: float, yaw: float) -> np.ndarray:
        feature = np.array([float(pitch), float(yaw), 1.0], dtype=np.float64)
        point = np.asarray(self.matrix, dtype=np.float64) @ feature
        return np.clip(point, 0.0, 1.0)

    def correct_angles(self, pitch: float, yaw: float) -> tuple[float, float, np.ndarray]:
        point = self.screen_point(pitch, yaw)
        corrected_pitch = (0.5 - float(point[1])) * np.radians(self.max_pitch_deg) * 2.0
        corrected_yaw = (float(point[0]) - 0.5) * np.radians(self.max_yaw_deg) * 2.0
        return float(corrected_pitch), float(corrected_yaw), point

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "matrix": np.asarray(self.matrix, dtype=np.float64).tolist(),
            "max_pitch_deg": self.max_pitch_deg,
            "max_yaw_deg": self.max_yaw_deg,
            "rmse": self.rmse,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        if matrix.shape != (2, 3):
            raise ValueError("calibration matrix must have shape (2, 3)")
        return cls(
            matrix=matrix,
            max_pitch_deg=float(data.get("max_pitch_deg", 30.0)),
            max_yaw_deg=float(data.get("max_yaw_deg", 40.0)),
            rmse=float(data.get("rmse", 0.0)),
        )


class CalibrationSession:
    """Nine-point timed calibration state machine for the webcam loop."""

    def __init__(
        self,
        points=DEFAULT_CALIBRATION_POINTS,
        settle_seconds: float = 0.45,
        hold_seconds: float = 0.9,
        min_samples: int = 8,
    ):
        self.points = tuple((float(x), float(y)) for x, y in points)
        self.settle_seconds = float(settle_seconds)
        self.hold_seconds = float(hold_seconds)
        self.min_samples = int(min_samples)
        self.index = 0
        self.started_at: float | None = None
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._point_samples = 0
        self.finished = False
        self.profile: CalibrationProfile | None = None

    @property
    def target(self) -> tuple[float, float] | None:
        if self.finished or self.index >= len(self.points):
            return None
        return self.points[self.index]

    @property
    def progress(self) -> float:
        if self.finished:
            return 1.0
        return self.index / float(max(len(self.points), 1))

    def update(
        self,
        pitch: float,
        yaw: float,
        valid: bool,
        timestamp: float | None = None,
    ) -> CalibrationProfile | None:
        if self.finished:
            return self.profile
        now = time.perf_counter() if timestamp is None else float(timestamp)
        if self.started_at is None:
            self.started_at = now
        elapsed = now - self.started_at
        if valid and elapsed >= self.settle_seconds:
            self.samples.append(
                (np.array([pitch, yaw, 1.0], dtype=np.float64),
                 np.array(self.target, dtype=np.float64))
            )
            self._point_samples += 1
        if elapsed < self.settle_seconds + self.hold_seconds:
            return None
        if self._point_samples < self.min_samples:
            # Keep the same target on screen until a face is available for a
            # sufficient number of frames; a temporary tracking loss should
            # never produce a misleading profile or crash the live loop.
            return None
        self.index += 1
        self.started_at = now
        self._point_samples = 0
        if self.index < len(self.points):
            return None
        features = np.vstack([item[0] for item in self.samples])
        targets = np.vstack([item[1] for item in self.samples])
        matrix = np.linalg.lstsq(features, targets, rcond=None)[0].T
        prediction = features @ matrix.T
        rmse = float(np.sqrt(np.mean((prediction - targets) ** 2)))
        self.profile = CalibrationProfile(matrix=matrix, rmse=rmse)
        self.finished = True
        return self.profile
