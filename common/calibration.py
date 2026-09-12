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
MAX_USABLE_RMSE = 0.15


def angles_to_screen_point(
    pitch: float,
    yaw: float,
    max_pitch_deg: float = 30.0,
    max_yaw_deg: float = 40.0,
    mirror: bool = False,
) -> np.ndarray:
    """Map model angles to a clipped, normalized screen point.

    This is the uncalibrated fallback used to keep the gaze position visible.
    A ``CalibrationProfile`` remains the preferred mapping when one is
    available.
    """
    pitch_limit = max(np.radians(float(max_pitch_deg)), 1e-6)
    yaw_limit = max(np.radians(float(max_yaw_deg)), 1e-6)
    point = np.array(
        [0.5 + (1.0 if mirror else -1.0) * float(yaw) / (2.0 * yaw_limit),
         0.5 - float(pitch) / (2.0 * pitch_limit)],
        dtype=np.float64,
    )
    return np.clip(point, 0.0, 1.0)


def camera_gaze_to_screen_point(
    pitch: float,
    yaw: float,
    max_pitch_deg: float = 30.0,
    max_yaw_deg: float = 40.0,
    mirror: bool = False,
) -> np.ndarray:
    """Map camera-frame angles (pitch-up/yaw-right) to a display point."""
    pitch_limit = max(np.radians(float(max_pitch_deg)), 1e-6)
    yaw_limit = max(np.radians(float(max_yaw_deg)), 1e-6)
    point = np.array(
        [0.5 + (-1.0 if mirror else 1.0) * float(yaw) / (2.0 * yaw_limit),
         0.5 - float(pitch) / (2.0 * pitch_limit)],
        dtype=np.float64,
    )
    return np.clip(point, 0.0, 1.0)


@dataclass
class CalibrationProfile:
    """Affine map from camera-frame angles to normalized screen coordinates."""

    matrix: np.ndarray
    max_pitch_deg: float = 30.0
    max_yaw_deg: float = 40.0
    rmse: float = 0.0
    feature_frame: str = "camera_angles"

    def is_usable(self, max_rmse: float = MAX_USABLE_RMSE) -> bool:
        """Return whether this profile is finite and fit well enough to use."""
        return bool(
            np.all(np.isfinite(self.matrix))
            and np.isfinite(self.rmse)
            and 0.0 <= self.rmse <= float(max_rmse)
            and self.feature_frame == "camera_angles"
        )

    def screen_point(self, pitch: float, yaw: float) -> np.ndarray:
        feature = np.array([float(pitch), float(yaw), 1.0], dtype=np.float64)
        point = np.asarray(self.matrix, dtype=np.float64) @ feature
        if not np.all(np.isfinite(point)):
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.clip(point, 0.0, 1.0)

    def correct_angles(self, pitch: float, yaw: float) -> tuple[float, float, np.ndarray]:
        point = self.screen_point(pitch, yaw)
        corrected_pitch = (0.5 - float(point[1])) * np.radians(self.max_pitch_deg) * 2.0
        corrected_yaw = (float(point[0]) - 0.5) * np.radians(self.max_yaw_deg) * 2.0
        return float(corrected_pitch), float(corrected_yaw), point

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "matrix": np.asarray(self.matrix, dtype=np.float64).tolist(),
            "max_pitch_deg": self.max_pitch_deg,
            "max_yaw_deg": self.max_yaw_deg,
            "rmse": self.rmse,
            "feature_frame": self.feature_frame,
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
            # Version 1 profiles were fitted against the model's signed yaw,
            # before head compensation.  Do not silently apply them to the
            # new camera-frame feature convention.
            feature_frame=data.get(
                "feature_frame",
                "model_angles" if int(data.get("version", 1)) < 2 else "camera_angles",
            ),
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
        self._point_features: list[np.ndarray] = []
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
            self._point_features.append(
                np.array([pitch, yaw, 1.0], dtype=np.float64)
            )
            self._point_samples += 1
        if elapsed < self.settle_seconds + self.hold_seconds:
            return None
        if self._point_samples < self.min_samples:
            # Keep the same target on screen until a face is available for a
            # sufficient number of frames; a temporary tracking loss should
            # never produce a misleading profile or crash the live loop.
            return None
        # A short median per target suppresses blink/tracker/model outliers;
        # fitting every correlated frame made one noisy hold dominate a whole
        # calibration profile.
        feature = np.median(np.vstack(self._point_features), axis=0)
        self.samples.append((feature, np.array(self.target, dtype=np.float64)))
        self.index += 1
        self.started_at = now
        self._point_samples = 0
        self._point_features = []
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
