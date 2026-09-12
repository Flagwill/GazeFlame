"""Software-only assistive-control concept layer.

This module deliberately stops at the intent/assistance boundary.  It maps a
normalized gaze point to a few conceptual cockpit regions and exposes a small
state machine for displaying how a future actuator could assist the pilot.
No robot, flight controller, or physical actuator is simulated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AssistiveTarget:
    """A conceptual cockpit attention region in normalized coordinates."""

    key: str
    label: str
    rect: tuple[float, float, float, float]
    color: tuple[int, int, int]

    def contains(self, point: np.ndarray) -> bool:
        x0, y0, x1, y1 = self.rect
        return bool(x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1)


@dataclass(frozen=True)
class AssistiveEstimate:
    """Frame-level output of :class:`AssistiveIntentState`."""

    target: AssistiveTarget | None
    status: str
    status_label: str
    intent_label: str
    point: np.ndarray | None
    confidence: float
    fixation_duration: float
    valid: bool
    event: str | None = None


# The regions are intentionally broad.  They are conceptual areas, not a
# calibrated cockpit or an assertion about a particular aircraft layout.
ASSISTIVE_TARGETS = (
    AssistiveTarget(
        "system", "SYSTEM STATUS", (0.04, 0.16, 0.30, 0.58), (90, 160, 235)
    ),
    AssistiveTarget(
        "attitude", "ATTITUDE", (0.34, 0.16, 0.66, 0.58), (90, 210, 120)
    ),
    AssistiveTarget(
        "navigation", "NAVIGATION", (0.70, 0.16, 0.96, 0.58), (220, 160, 80)
    ),
    AssistiveTarget(
        "control", "CONTROL AREA", (0.30, 0.66, 0.70, 0.92), (190, 110, 220)
    ),
)


class AssistiveIntentState:
    """Turn gaze/fixation evidence into an explainable assistance state.

    The state names describe a future human-machine interface.  They are not
    control commands and do not claim that an actuator is connected.
    """

    def __init__(
        self,
        intent_seconds: float = 0.35,
        ready_seconds: float = 0.80,
        min_confidence: float = 0.25,
    ):
        self.intent_seconds = float(intent_seconds)
        self.ready_seconds = float(ready_seconds)
        self.min_confidence = float(min_confidence)
        self._last_target_key: str | None = None
        self._last_status: str | None = None

    @staticmethod
    def _find_target(point: np.ndarray | None) -> AssistiveTarget | None:
        if point is None:
            return None
        value = np.asarray(point, dtype=np.float64).reshape(-1)
        if value.size < 2 or not np.all(np.isfinite(value[:2])):
            return None
        if not np.all((value[:2] >= 0.0) & (value[:2] <= 1.0)):
            return None
        return next(
            (target for target in ASSISTIVE_TARGETS if target.contains(value)),
            None,
        )

    def update(
        self,
        point: np.ndarray | tuple[float, float] | None,
        valid: bool,
        confidence: float,
        fixation_duration: float = 0.0,
        velocity_deg_s: float = 0.0,
        blinking: bool = False,
        timestamp: float | None = None,
    ) -> AssistiveEstimate:
        """Update the conceptual assistance state for one frame."""
        del timestamp  # Reserved for future event logging; state is frame-based.
        value = (
            None
            if point is None
            else np.asarray(point, dtype=np.float64).reshape(-1)[:2]
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        valid = bool(valid and value is not None and confidence >= self.min_confidence)
        target = self._find_target(value) if valid else None
        fixation_duration = max(0.0, float(fixation_duration))
        velocity_deg_s = max(0.0, float(velocity_deg_s))

        if not valid:
            status = "NO_TRACKING"
            status_label = "NO TRACKING"
            intent_label = "WAITING FOR DRIVER"
        elif blinking:
            status = "PAUSED"
            status_label = "PAUSED"
            intent_label = "BLINK / TEMPORARY LOSS"
        elif target is None:
            status = "MONITORING"
            status_label = "MONITORING"
            intent_label = "NO COCKPIT TARGET"
        elif velocity_deg_s >= 100.0 or fixation_duration < self.intent_seconds:
            status = "MONITORING"
            status_label = "MONITORING"
            intent_label = f"OBSERVING {target.label}"
        elif fixation_duration < self.ready_seconds:
            status = "INTENT_DETECTED"
            status_label = "INTENT DETECTED"
            intent_label = f"FOCUS ON {target.label}"
        else:
            status = "ASSIST_READY"
            status_label = "ASSIST READY"
            intent_label = f"ASSIST {target.label}"

        target_key = target.key if target else None
        previous_target_key = self._last_target_key
        if target_key != previous_target_key and target is not None:
            # A new cockpit region must earn its own dwell time.  This keeps
            # the concept layer from carrying fixation evidence across a
            # saccade or a target boundary.
            status = "MONITORING"
            status_label = "MONITORING"
            intent_label = f"OBSERVING {target.label}"
        event = None
        if target_key != previous_target_key and target_key is not None:
            event = "target_changed"
        elif status != self._last_status:
            event = status.lower()
        self._last_target_key = target_key
        self._last_status = status

        return AssistiveEstimate(
            target=target,
            status=status,
            status_label=status_label,
            intent_label=intent_label,
            point=None if value is None else value.copy(),
            confidence=confidence,
            fixation_duration=fixation_duration,
            valid=valid,
            event=event,
        )


def draw_assistive_panel(
    height: int,
    estimate: AssistiveEstimate | None,
    width: int | None = None,
) -> np.ndarray:
    """Render the conceptual cockpit and assistive-state panel."""
    height = max(int(height), 300)
    width = max(int(width or round(height * 0.78)), 300)
    panel = np.full((height, width, 3), (26, 30, 35), dtype=np.uint8)
    pad = max(16, int(min(width, height) * 0.045))
    scale = max(0.48, min(width, height) / 720.0)

    def text(value: str, origin: tuple[int, int], size: float = 0.65,
             color: tuple[int, int, int] = (225, 230, 235), thickness: int = 1):
        cv2.putText(panel, value, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, size * scale), color, thickness, cv2.LINE_AA)

    text("ASSISTIVE CONTROL CONCEPT", (pad, int(30 * scale) + 10), 0.68,
         (240, 245, 250), 2)
    text("Virtual intent layer", (pad, int(54 * scale) + 10), 0.50, (145, 155, 165), 1)

    cockpit_top = max(76, int(height * 0.16))
    cockpit_bottom = max(cockpit_top + 100, int(height * 0.60))
    cockpit_left, cockpit_right = pad, width - pad
    cv2.rectangle(panel, (cockpit_left, cockpit_top), (cockpit_right, cockpit_bottom),
                  (80, 88, 96), 1, cv2.LINE_AA)
    text("CONCEPTUAL COCKPIT", (cockpit_left + 8, cockpit_top - 10), 0.48,
         (150, 165, 175), 1)

    for target in ASSISTIVE_TARGETS:
        x0, y0, x1, y1 = target.rect
        box = (
            cockpit_left + int(x0 * (cockpit_right - cockpit_left)),
            cockpit_top + int(y0 * (cockpit_bottom - cockpit_top)),
            cockpit_left + int(x1 * (cockpit_right - cockpit_left)),
            cockpit_top + int(y1 * (cockpit_bottom - cockpit_top)),
        )
        active = (
            estimate is not None
            and estimate.target is not None
            and estimate.target.key == target.key
        )
        color = target.color if active else (75, 82, 90)
        thickness = 3 if active else 1
        cv2.rectangle(panel, box[:2], box[2:], color, thickness, cv2.LINE_AA)
        text(target.label, (box[0] + 7, box[1] + 22), 0.43,
             color if active else (145, 150, 155), 1)

    if estimate is not None and estimate.point is not None:
        point = np.asarray(estimate.point, dtype=np.float64)
        marker = (
            cockpit_left + int(np.clip(point[0], 0.0, 1.0) * (cockpit_right - cockpit_left)),
            cockpit_top + int(np.clip(point[1], 0.0, 1.0) * (cockpit_bottom - cockpit_top)),
        )
        cv2.circle(panel, marker, max(7, int(10 * scale)), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(panel, marker, (0, 80, 255), cv2.MARKER_CROSS,
                       max(18, int(28 * scale)), 2, cv2.LINE_AA)

    status_top = int(height * 0.66)
    cv2.line(panel, (pad, status_top - 14), (width - pad, status_top - 14),
             (70, 78, 86), 1, cv2.LINE_AA)
    status = estimate.status_label if estimate is not None else "WAITING"
    status_color = {
        "NO TRACKING": (120, 130, 140),
        "MONITORING": (210, 180, 80),
        "INTENT DETECTED": (80, 210, 240),
        "ASSIST READY": (80, 220, 120),
        "PAUSED": (150, 150, 230),
    }.get(status, (220, 220, 220))
    text(f"STATE  {status}", (pad, status_top + 16), 0.62, status_color, 2)
    target_label = estimate.target.label if estimate and estimate.target else "NONE"
    intent_label = estimate.intent_label if estimate else "WAITING FOR DRIVER"
    confidence = estimate.confidence if estimate else 0.0
    fixation = estimate.fixation_duration if estimate else 0.0
    text(f"TARGET  {target_label}", (pad, status_top + 45), 0.48)
    text(f"INTENT  {intent_label}", (pad, status_top + 71), 0.45)
    text(f"CONFIDENCE  {confidence:.2f}   FIXATION  {fixation:.2f}s",
         (pad, status_top + 97), 0.43, (175, 185, 195))

    footer_y = height - max(20, int(28 * scale))
    cv2.line(panel, (pad, footer_y - 18), (width - pad, footer_y - 18),
             (70, 78, 86), 1, cv2.LINE_AA)
    text("GAZE  ->  INTENT  ->  ASSIST", (pad, footer_y), 0.48, (180, 205, 220), 1)
    return panel


def compose_with_assistive(
    left_bgr: np.ndarray,
    head_bgr: np.ndarray | None,
    assistive_panel: np.ndarray,
    sep_w: int = 4,
) -> np.ndarray:
    """Build a stable input + optional FLAME + assistive concept window."""
    from common.flame_mirror import compose_with_head

    base = compose_with_head(left_bgr, head_bgr) if head_bgr is not None else left_bgr
    h = base.shape[0]
    panel = assistive_panel
    if panel.shape[0] != h:
        panel = cv2.resize(panel, (panel.shape[1], h), interpolation=cv2.INTER_AREA)
    sep = np.zeros((h, max(1, int(sep_w)), 3), dtype=np.uint8)
    return np.hstack([base, sep, panel])
