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
MAX_USABLE_GAIN = 8.0  # normalized screen units / radian


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
    validation_rmse: float | None = None
    sample_counts: tuple[int, ...] = ()
    point_spread_deg: tuple[float, ...] = ()

    def is_usable(self, max_rmse: float = MAX_USABLE_RMSE) -> bool:
        """Return whether this profile is finite and fit well enough to use."""
        return bool(
            np.shape(self.matrix) == (2, 3)
            and np.all(np.isfinite(self.matrix))
            and np.isfinite(self.rmse)
            and 0.0 <= self.rmse <= float(max_rmse)
            and self.feature_frame == "camera_angles"
            and np.linalg.norm(np.asarray(self.matrix)[:, :2], ord=2) <= MAX_USABLE_GAIN
            and (self.validation_rmse is None or (
                np.isfinite(self.validation_rmse) and 0 <= self.validation_rmse <= max_rmse
            ))
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
            "version": 3,
            "matrix": np.asarray(self.matrix, dtype=np.float64).tolist(),
            "max_pitch_deg": self.max_pitch_deg,
            "max_yaw_deg": self.max_yaw_deg,
            "rmse": self.rmse,
            "feature_frame": self.feature_frame,
            "validation_rmse": self.validation_rmse,
            "sample_counts": list(self.sample_counts),
            "point_spread_deg": list(self.point_spread_deg),
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
            validation_rmse=(None if data.get("validation_rmse") is None
                             else float(data["validation_rmse"])),
            sample_counts=tuple(data.get("sample_counts", ())),
            point_spread_deg=tuple(data.get("point_spread_deg", ())),
        )


class CalibrationSession:
    """Nine-point timed calibration state machine for the webcam loop."""

    def __init__(
        self,
        points=DEFAULT_CALIBRATION_POINTS,
        settle_seconds: float = 0.7,
        hold_seconds: float = 1.5,
        min_samples: int = 10,
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
        self._sample_started: float | None = None
        self._last_update: float | None = None
        self.sample_counts: list[int] = []
        self.point_spread_deg: list[float] = []
        self.status = "LOOK AT TARGET"

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
        confidence: float = 1.0,
        blinking: bool = False,
        velocity_deg_s: float = 0.0,
    ) -> CalibrationProfile | None:
        if self.finished:
            return self.profile
        now = time.perf_counter() if timestamp is None else float(timestamp)
        if self._last_update is not None:
            if now <= self._last_update:
                return None
            if now - self._last_update > 0.35:
                self._point_features.clear()
                self._point_samples = 0
                self._sample_started = None
        self._last_update = now
        if self.started_at is None:
            self.started_at = now
        elapsed = now - self.started_at
        usable = (
            valid and not blinking
            and np.all(np.isfinite([pitch, yaw, confidence, velocity_deg_s]))
            and confidence >= 0.5 and velocity_deg_s <= 35.0
        )
        if not usable:
            # Require a fresh steady hold after loss, blink or saccade; stale
            # samples from a previous pose must not complete the target.
            self._point_features.clear()
            self._point_samples = 0
            self._sample_started = None
            self.status = "WAIT FOR OPEN EYES / STEADY TRACKING"
            return None
        self.status = "LOOK AT TARGET" if elapsed < self.settle_seconds else "HOLD STEADY"
        if elapsed >= self.settle_seconds:
            if self._sample_started is None:
                self._sample_started = now
            self._point_features.append(
                np.array([pitch, yaw, 1.0], dtype=np.float64)
            )
            self._point_samples += 1
        if self._sample_started is None or now - self._sample_started < self.hold_seconds:
            return None
        if self._point_samples < self.min_samples:
            # Keep the same target on screen until a face is available for a
            # sufficient number of frames; a temporary tracking loss should
            # never produce a misleading profile or crash the live loop.
            return None
        features = np.vstack(self._point_features)
        center = np.median(features, axis=0)
        deviation = np.abs(features[:, :2] - center[:2])
        sigma = 1.4826 * np.median(deviation, axis=0)
        inliers = np.all(deviation <= np.maximum(3.0 * sigma, np.radians(0.75)), axis=1)
        kept = features[inliers]
        spread = (float(np.degrees(np.percentile(np.linalg.norm(kept[:, :2] - center[:2], axis=1), 90)))
                  if len(kept) else float("inf"))
        if len(kept) < self.min_samples or spread > 3.0:
            self.status = "POINT TOO NOISY / KEEP LOOKING"
            self._point_features.clear()
            self._point_samples = 0
            self._sample_started = None
            return None
        feature = np.median(kept, axis=0)
        self.samples.append((feature, np.array(self.target, dtype=np.float64)))
        self.sample_counts.append(len(kept))
        self.point_spread_deg.append(spread)
        self.index += 1
        self.started_at = now
        self._point_samples = 0
        self._point_features = []
        self._sample_started = None
        if self.index < len(self.points):
            return None
        features = np.vstack([item[0] for item in self.samples])
        targets = np.vstack([item[1] for item in self.samples])
        centered = features[:, :2] - features[:, :2].mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        # A narrow/collinear range can fit the nine medians but amplify tiny
        # angle noise or extrapolate wildly on the next frame.
        well_spread = (
            len(singular) == 2 and singular[-1] > np.radians(1.0)
            and singular[-1] / max(singular[0], 1e-8) >= 0.08
            and np.linalg.matrix_rank(features) == 3
        )
        matrix = np.linalg.lstsq(features, targets, rcond=None)[0].T
        prediction = features @ matrix.T
        rmse = float(np.sqrt(np.mean((prediction - targets) ** 2)))
        # Predict each target without using it for the fit. Training RMSE alone
        # rewards over-sensitive maps and conceals a badly sampled corner.
        held_out = []
        for i in range(len(features)):
            mask = np.arange(len(features)) != i
            fit = np.linalg.lstsq(features[mask], targets[mask], rcond=None)[0]
            held_out.append(features[i] @ fit - targets[i])
        validation_rmse = float(np.sqrt(np.mean(np.asarray(held_out) ** 2)))
        self.profile = CalibrationProfile(
            matrix=matrix, rmse=rmse,
            validation_rmse=validation_rmse if well_spread else float("inf"),
            sample_counts=tuple(self.sample_counts),
            point_spread_deg=tuple(self.point_spread_deg),
        )
        self.finished = True
        self.status = (f"CALIBRATION READY / RMSE {validation_rmse:.3f}" if self.profile.is_usable()
                       else "CALIBRATION REJECTED / PRESS C TO RETRY")
        return self.profile
