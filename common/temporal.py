"""Temporal gaze filtering and event detection.

The model predicts an angle for each frame.  This module turns those
independent samples into a realtime signal with bounded jitter, a confidence
value, and coarse eye-movement events that are useful to applications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from common.geometry import (
    angular_error,
    camera_gaze_to_pitch_yaw,
    camera_pitch_yaw_to_gaze,
    gaze_to_camera,
    gazetr_angles_to_gaze,
)


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
        self._raw_value: np.ndarray | None = None

    def reset(self) -> None:
        """Forget samples after a tracking gap or a change of coordinate frame."""
        self._value = self._derivative = self._raw_value = None
        self._timestamp = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return float(1.0 / (1.0 + tau / max(dt, 1e-6)))

    def update(self, value, timestamp: float | None = None) -> np.ndarray:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        sample = np.asarray(value, dtype=np.float64)
        if self._value is None:
            self._value = sample.copy()
            self._raw_value = sample.copy()
            self._derivative = np.zeros_like(sample)
            self._timestamp = now
            return self._value.copy()

        if now <= float(self._timestamp):
            return self._value.copy()
        dt = max(now - float(self._timestamp), 1e-3)
        # Derivatives use consecutive measurements, not the lagging filtered
        # value. Otherwise residual filter error masquerades as eye motion.
        raw_derivative = (sample - self._raw_value) / dt
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
        self._raw_value = sample.copy()
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
    camera_gaze: np.ndarray | None = None
    measurement_valid: bool = True

    @property
    def event(self) -> str | None:
        """Return the first event name for convenient overlay code."""
        return self.events[0]["type"] if self.events else None


class GazeTemporalState:
    """Filter gaze/head pose and emit blink, fixation and saccade events."""

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 1.2,
        fixation_velocity_deg_s: float = 35.0,
        saccade_velocity_deg_s: float = 100.0,
        fixation_min_duration: float = 0.18,
        max_missing_seconds: float = 0.35,
        blink_closed_ear: float = 0.18,
        blink_open_ear: float = 0.28,
        spike_threshold_deg: float = 12.0,
        min_confidence: float = 0.25,
    ):
        self._gaze_filter = OneEuroFilter(min_cutoff, beta)
        self._head_filter = OneEuroFilter(min_cutoff, beta)
        self.fixation_velocity_deg_s = float(fixation_velocity_deg_s)
        self.saccade_velocity_deg_s = float(saccade_velocity_deg_s)
        self.fixation_min_duration = float(fixation_min_duration)
        self.max_missing_seconds = float(max_missing_seconds)
        self.blink_closed_ear = float(blink_closed_ear)
        self.blink_open_ear = float(blink_open_ear)
        self.spike_threshold_deg = float(spike_threshold_deg)
        self.min_confidence = float(min_confidence)

        self._last_timestamp: float | None = None
        self._last_gaze: np.ndarray | None = None
        self._last_pitch_yaw = np.zeros(2, dtype=np.float64)
        self._last_head_pose: np.ndarray | None = None
        self._last_head_euler: np.ndarray | None = None
        self._last_valid_timestamp: float | None = None
        self._last_measurement_timestamp: float | None = None
        self._last_raw_gaze: np.ndarray | None = None
        self._pending_gaze: np.ndarray | None = None
        self._pending_timestamp: float | None = None
        self._motion_timestamp: float | None = None
        self._motion_gaze: np.ndarray | None = None
        self._motion_raw_gaze: np.ndarray | None = None
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
        raw_gaze: np.ndarray,
        timestamp: float,
        events: list[dict],
        raw_timestamp: float | None = None,
    ) -> float:
        dt = timestamp - self._motion_timestamp if self._motion_timestamp is not None else 0.0
        speed = angular_error(self._motion_gaze, gaze) / dt if dt > 0 else 0.0
        raw_dt = (
            (timestamp if raw_timestamp is None else raw_timestamp) - self._motion_timestamp
            if self._motion_timestamp is not None else 0.0
        )
        raw_speed = angular_error(self._motion_raw_gaze, raw_gaze) / raw_dt if raw_dt > 0 else 0.0
        # Retain real saccades even though the displayed vector is smoothed;
        # ordinary frame noise uses the filtered speed for fixation evidence.
        if raw_speed >= self.saccade_velocity_deg_s:
            speed = max(speed, raw_speed)

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

        self._motion_gaze = gaze.copy()
        self._motion_raw_gaze = raw_gaze.copy()
        self._motion_timestamp = timestamp
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
        if (eye_aspect_ratio <= self.blink_closed_ear or self._blink_value >= 0.55) and not self._blinking:
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

    @staticmethod
    def tracking_confidence(info: dict) -> float:
        """Geometry/tracker quality proxy; not a learned gaze probability."""
        error = float(info.get("reprojection_error", 0.0))
        if not np.isfinite(error):
            return 0.0
        confidence = float(np.exp(-max(error, 0.0) / 8.0))
        quality = info.get("tracking_quality")
        if info.get("source") == "tracked" and quality is not None:
            if not np.isfinite(quality):
                return 0.0
            confidence *= float(np.clip((quality - 3.0) / 4.0, 0.0, 1.0))
        return confidence

    def _reset_motion(self, now: float, events: list[dict]) -> None:
        self._end_fixation(now, events)
        if self._saccade_active:
            events.append(self._event("saccade_end", now, velocity=0.0))
        self._saccade_active = False
        self._motion_gaze = self._motion_raw_gaze = None
        self._motion_timestamp = None

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
        if self._last_timestamp is not None:
            now = max(now, self._last_timestamp)
        info = info or {}
        confidence = self.tracking_confidence(info)
        has_face = (
            face_count > 0
            and info.get("landmarks6") is not None
            and np.all(np.isfinite([pitch, yaw]))
            and confidence >= self.min_confidence
            and (info.get("eye_aspect_ratio") is None or np.isfinite(info["eye_aspect_ratio"]))
        )
        events: list[dict] = []
        measurement_valid = False
        rotation = info.get("normalization_rotation")
        raw_camera_gaze = None
        if has_face:
            try:
                raw_camera_gaze = gaze_to_camera(gazetr_angles_to_gaze(pitch, yaw), rotation)
            except ValueError:
                has_face = False
        gap = (
            self._last_measurement_timestamp is not None
            and now - self._last_measurement_timestamp > self.max_missing_seconds
        )
        if gap:
            self._gaze_filter.reset()
            self._head_filter.reset()
            self._last_raw_gaze = self._pending_gaze = None
            self._pending_timestamp = None
            self._reset_motion(now, events)
        if has_face:
            self._last_valid_timestamp = now
            if not self._was_valid:
                events.append(self._event("face_found", now))
            self._was_valid = True
            blink = self._update_blink(info.get("eye_aspect_ratio"), now, events)
            measurement_valid = not self._blinking
            raw_timestamp = now
            if measurement_valid and self._last_raw_gaze is not None:
                jump = angular_error(self._last_raw_gaze, raw_camera_gaze)
                if jump > self.spike_threshold_deg:
                    # One frame cannot distinguish a bad crop from a saccade.
                    # Confirm large jumps on the next nearby measurement.
                    measurement_valid = (
                        self._pending_gaze is not None
                        and angular_error(self._pending_gaze, raw_camera_gaze) <= self.spike_threshold_deg * 0.65
                    )
                    if measurement_valid:
                        # Confirmation delays display by one frame, but the
                        # movement occurred when the candidate first arrived.
                        raw_timestamp = self._pending_timestamp
                        self._pending_gaze = None
                        self._pending_timestamp = None
                    else:
                        self._pending_gaze = raw_camera_gaze.copy()
                        self._pending_timestamp = now
                else:
                    self._pending_gaze = None
                    self._pending_timestamp = None
            if measurement_valid:
                camera_angles = camera_gaze_to_pitch_yaw(raw_camera_gaze)
                filtered = self._gaze_filter.update(camera_angles, now)
                camera_gaze = camera_pitch_yaw_to_gaze(*filtered)
                self._last_gaze = camera_gaze.copy()
                self._last_raw_gaze = raw_camera_gaze.copy()
                self._last_measurement_timestamp = now
                velocity = self._update_motion(camera_gaze, raw_camera_gaze, now, events, raw_timestamp)
            else:
                camera_gaze = self._last_gaze.copy() if self._last_gaze is not None else raw_camera_gaze.copy()
                velocity = 0.0
                if not self._blinking:
                    confidence = min(confidence, 0.15)
                self._end_fixation(now, events)
                if self._blinking:
                    self._pending_gaze = None
                    self._pending_timestamp = None
                    self._reset_motion(now, events)
            # Keep the legacy normalized-frame output for callers that still
            # need it, but consumers must use camera_gaze without another R.T.
            gaze = camera_gaze.copy() if rotation is None else np.asarray(rotation) @ camera_gaze
            normalized_pitch, normalized_yaw = camera_gaze_to_pitch_yaw(gaze)
            filtered_pitch, filtered_yaw = -normalized_yaw, normalized_pitch
            self._last_pitch_yaw[:] = [filtered_pitch, filtered_yaw]
            if info.get("head_pose") is not None and np.all(np.isfinite(info["head_pose"])):
                self._last_head_pose = self._head_filter.update(info["head_pose"], now)
            if info.get("head_euler") is not None and np.all(np.isfinite(info["head_euler"])):
                self._last_head_euler = np.asarray(info["head_euler"], dtype=np.float64).copy()
        else:
            last_seen = self._last_valid_timestamp
            still_recent = last_seen is not None and now - last_seen <= self.max_missing_seconds
            filtered_pitch, filtered_yaw = map(float, self._last_pitch_yaw)
            camera_gaze = (
                self._last_gaze.copy()
                if self._last_gaze is not None
                else gazetr_angles_to_gaze(filtered_pitch, filtered_yaw)
            )
            gaze = gazetr_angles_to_gaze(filtered_pitch, filtered_yaw)
            velocity = 0.0
            blink = self._blink_value
            confidence = 0.15 if still_recent else 0.0
            if self._was_valid:
                events.append(self._event("face_lost", now))
            self._reset_motion(now, events)
            self._pending_gaze = None
            self._pending_timestamp = None
            self._was_valid = False

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
            camera_gaze=camera_gaze,
            measurement_valid=measurement_valid,
        )
