"""Temporal gaze filtering and event detection.

The model predicts an angle for each frame.  This module turns those
independent samples into a realtime signal with bounded jitter, a confidence
value, and coarse eye-movement events that are useful to applications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from common.geometry import pitch_yaw_to_gaze


class OneEuroFilter:
    """Low-latency adaptive low-pass filter from the One Euro Filter paper."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.035,
        derivative_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return float(1.0 / (1.0 + tau / max(dt, 1e-6)))

    def update(self, value, timestamp: float | None = None) -> np.ndarray:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        sample = np.asarray(value, dtype=np.float64)
        if self._value is None:
            self._value = sample.copy()
            self._derivative = np.zeros_like(sample)
            self._timestamp = now
            return self._value.copy()

        dt = max(now - float(self._timestamp), 1e-3)
        raw_derivative = (sample - self._value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self._derivative = (
            derivative_alpha * raw_derivative
            + (1.0 - derivative_alpha) * self._derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = np.asarray([self._alpha(c, dt) for c in np.ravel(cutoff)]).reshape(
            cutoff.shape
        )
        self._value = alpha * sample + (1.0 - alpha) * self._value
        self._timestamp = now
        return self._value.copy()


@dataclass
class TemporalEstimate:
    """The filtered state returned for one input frame."""

    timestamp: float
    raw_pitch: float
    raw_yaw: float
    pitch: float
    yaw: float
    gaze: np.ndarray
    valid: bool
    confidence: float
    velocity_deg_s: float
    fixation_duration: float
    blink: float
    blinking: bool
    head_pose: np.ndarray | None = None
    head_euler: np.ndarray | None = None
    screen_point: np.ndarray | None = None
    events: list[dict] = field(default_factory=list)

    @property
    def event(self) -> str | None:
        """Return the first event name for convenient overlay code."""
        return self.events[0]["type"] if self.events else None


class GazeTemporalState:
    """Filter gaze/head pose and emit blink, fixation and saccade events."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.035,
        fixation_velocity_deg_s: float = 35.0,
        saccade_velocity_deg_s: float = 100.0,
        fixation_min_duration: float = 0.18,
        max_missing_seconds: float = 0.35,
        blink_closed_ear: float = 0.18,
        blink_open_ear: float = 0.28,
    ):
        self._gaze_filter = OneEuroFilter(min_cutoff, beta)
        self._head_filter = OneEuroFilter(min_cutoff, beta)
        self.fixation_velocity_deg_s = float(fixation_velocity_deg_s)
        self.saccade_velocity_deg_s = float(saccade_velocity_deg_s)
        self.fixation_min_duration = float(fixation_min_duration)
        self.max_missing_seconds = float(max_missing_seconds)
        self.blink_closed_ear = float(blink_closed_ear)
        self.blink_open_ear = float(blink_open_ear)

        self._last_timestamp: float | None = None
        self._last_gaze: np.ndarray | None = None
        self._last_pitch_yaw = np.zeros(2, dtype=np.float64)
        self._last_head_pose: np.ndarray | None = None
        self._last_head_euler: np.ndarray | None = None
        self._last_valid_timestamp: float | None = None
        self._was_valid = False
        self._low_velocity_since: float | None = None
        self._fixation_active = False
        self._fixation_started: float | None = None
        self._saccade_active = False
        self._blink_value = 0.0
        self._blinking = False
        self._blink_started: float | None = None

    @staticmethod
    def _event(event_type: str, timestamp: float, **values) -> dict:
        return {"type": event_type, "timestamp": float(timestamp), **values}

    def _end_fixation(self, timestamp: float, events: list[dict]) -> None:
        if self._fixation_active:
            events.append(
                self._event(
                    "fixation_end",
                    timestamp,
                    duration=max(0.0, timestamp - float(self._fixation_started)),
                )
            )
        self._fixation_active = False
        self._fixation_started = None
        self._low_velocity_since = None

    def _update_motion(
        self,
        gaze: np.ndarray,
        timestamp: float,
        events: list[dict],
    ) -> float:
        if self._last_gaze is None or self._last_timestamp is None:
            self._last_gaze = gaze.copy()
            return 0.0
        dt = max(timestamp - self._last_timestamp, 1e-3)
        cosine = float(np.clip(np.dot(self._last_gaze, gaze), -1.0, 1.0))
        speed = float(np.degrees(np.arccos(cosine)) / dt)

        if speed >= self.saccade_velocity_deg_s:
            self._low_velocity_since = None
            self._end_fixation(timestamp, events)
            if not self._saccade_active:
                events.append(self._event("saccade_start", timestamp, velocity=speed))
                self._saccade_active = True
        elif self._saccade_active and speed < self.saccade_velocity_deg_s * 0.65:
            events.append(self._event("saccade_end", timestamp, velocity=speed))
            self._saccade_active = False

        if speed <= self.fixation_velocity_deg_s and not self._saccade_active:
            if self._low_velocity_since is None:
                self._low_velocity_since = timestamp
            if (
                not self._fixation_active
                and timestamp - self._low_velocity_since >= self.fixation_min_duration
            ):
                self._fixation_active = True
                self._fixation_started = self._low_velocity_since
                events.append(
                    self._event(
                        "fixation_start",
                        timestamp,
                        pitch=float(self._last_pitch_yaw[0]),
                        yaw=float(self._last_pitch_yaw[1]),
                    )
                )
        elif not self._saccade_active:
            self._end_fixation(timestamp, events)

        self._last_gaze = gaze.copy()
        return speed

    def _update_blink(
        self,
        eye_aspect_ratio: float | None,
        timestamp: float,
        events: list[dict],
    ) -> float:
        if eye_aspect_ratio is None:
            return self._blink_value
        span = max(self.blink_open_ear - self.blink_closed_ear, 1e-4)
        target = float(np.clip((self.blink_open_ear - eye_aspect_ratio) / span, 0.0, 1.0))
        self._blink_value = 0.65 * self._blink_value + 0.35 * target
        if self._blink_value >= 0.55 and not self._blinking:
            self._blinking = True
            self._blink_started = timestamp
            events.append(self._event("blink_start", timestamp))
        elif self._blink_value <= 0.25 and self._blinking:
            events.append(
                self._event(
                    "blink_end",
                    timestamp,
                    duration=max(0.0, timestamp - float(self._blink_started)),
                )
            )
            self._blinking = False
            self._blink_started = None
        return self._blink_value

    def update(
        self,
        pitch: float,
        yaw: float,
        face_count: int = 1,
        info: dict | None = None,
        timestamp: float | None = None,
        screen_point: np.ndarray | None = None,
    ) -> TemporalEstimate:
        """Consume one model result and return the current temporal estimate."""
        now = time.perf_counter() if timestamp is None else float(timestamp)
        info = info or {}
        has_face = (
            face_count > 0
            and info.get("landmarks6") is not None
            and np.all(np.isfinite([pitch, yaw]))
        )
        events: list[dict] = []
        if has_face:
            filtered = self._gaze_filter.update([pitch, yaw], now)
            filtered_pitch, filtered_yaw = map(float, filtered)
            gaze = pitch_yaw_to_gaze(filtered_pitch, filtered_yaw).astype(np.float64)
            self._last_pitch_yaw[:] = [filtered_pitch, filtered_yaw]
            self._last_valid_timestamp = now
            if not self._was_valid:
                events.append(self._event("face_found", now))
            self._was_valid = True
            velocity = self._update_motion(gaze, now, events)
            blink = self._update_blink(info.get("eye_aspect_ratio"), now, events)
            error = float(info.get("reprojection_error", 0.0))
            confidence = float(np.exp(-max(error, 0.0) / 8.0))
            if info.get("head_pose") is not None:
                self._last_head_pose = self._head_filter.update(info["head_pose"], now)
            if info.get("head_euler") is not None:
                self._last_head_euler = np.asarray(info["head_euler"], dtype=np.float64).copy()
        else:
            last_seen = self._last_valid_timestamp
            still_recent = last_seen is not None and now - last_seen <= self.max_missing_seconds
            filtered_pitch, filtered_yaw = map(float, self._last_pitch_yaw)
            gaze = (
                self._last_gaze.copy()
                if self._last_gaze is not None
                else pitch_yaw_to_gaze(filtered_pitch, filtered_yaw)
            )
            velocity = 0.0
            blink = self._blink_value
            confidence = 0.15 if still_recent else 0.0
            if self._was_valid:
                events.append(self._event("face_lost", now))
                self._end_fixation(now, events)
            if self._saccade_active:
                events.append(self._event("saccade_end", now, velocity=0.0))
                self._saccade_active = False
            self._was_valid = False

        if self._last_timestamp is not None and now < self._last_timestamp:
            now = self._last_timestamp
        self._last_timestamp = now
        fixation_duration = (
            max(0.0, now - float(self._fixation_started))
            if self._fixation_active and self._fixation_started is not None
            else 0.0
        )
        return TemporalEstimate(
            timestamp=now,
            raw_pitch=float(pitch),
            raw_yaw=float(yaw),
            pitch=filtered_pitch,
            yaw=filtered_yaw,
            gaze=gaze,
            valid=has_face,
            confidence=confidence,
            velocity_deg_s=velocity,
            fixation_duration=fixation_duration,
            blink=blink,
            blinking=self._blinking,
            head_pose=None if self._last_head_pose is None else self._last_head_pose.copy(),
            head_euler=None if self._last_head_euler is None else self._last_head_euler.copy(),
            screen_point=None if screen_point is None else np.asarray(screen_point).copy(),
            events=events,
        )
