"""Report-demo presentation and explicitly synthetic scenario inputs.

No model, GL context or actuator is created here. Live demo frames use the
existing webcam pipeline; replay samples only exercise the shared temporal and
assistive state machines (report sections 5.2/5.3), not model accuracy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from common.assistive import AssistiveEstimate, draw_assistive_panel
from common.temporal import TemporalEstimate


@dataclass(frozen=True)
class ReplayStep:
    label: str
    seconds: float
    point: tuple[float, float]
    valid: bool = True
    ear: float = 0.30
    reprojection_error: float = 0.8


REPORT_STEPS = (
    ReplayStep("ATTITUDE: stable gaze builds intent", 3.0, (0.50, 0.34)),
    ReplayStep("SYSTEM: new region resets dwell", 3.0, (0.17, 0.34)),
    ReplayStep("NAVIGATION: observe a new target", 3.0, (0.83, 0.34)),
    ReplayStep("CONTROL: assistance readiness only", 3.0, (0.50, 0.79)),
    ReplayStep("BLINK: pause, never execute", 0.6, (0.50, 0.79), ear=0.12),
    ReplayStep("RECOVERY: rebuild stable evidence", 2.0, (0.50, 0.79)),
    ReplayStep("LOW CONFIDENCE: reject assistance", 1.2, (0.50, 0.79),
               reprojection_error=24.0),
    ReplayStep("RECOVERY: confidence restored", 2.0, (0.50, 0.79)),
    ReplayStep("FACE LOST: clear driver intent", 1.2, (0.50, 0.79), valid=False),
    ReplayStep("FACE FOUND: reacquire ATTITUDE", 3.0, (0.50, 0.34)),
    ReplayStep("OUTSIDE: no cockpit target", 2.0, (0.02, 0.98)),
)


class ReportReplay:
    """Deterministic sensor-level inputs; never presented as GazeTR outputs."""

    def __init__(self, seed: int = 49):
        self.rng = np.random.default_rng(seed)
        self.duration = round(sum(step.seconds for step in REPORT_STEPS), 6)

    def sample(self, timestamp: float) -> tuple[ReplayStep, float, float, int, dict]:
        elapsed = max(0.0, float(timestamp))
        step = REPORT_STEPS[-1]
        for candidate in REPORT_STEPS:
            if elapsed < candidate.seconds - 1e-9:
                step = candidate
                break
            elapsed -= candidate.seconds
        x, y = step.point
        # Invert the demo's *uncalibrated screen mapping*. The actual vector
        # conversion still happens only in common.geometry / temporal.
        angle0 = np.radians(-(x - 0.5) * 80.0)
        angle1 = np.radians((0.5 - y) * 60.0)
        angle0, angle1 = self.rng.normal([angle0, angle1], np.radians(0.45))
        info = {
            "landmarks6": np.zeros((6, 2)),  # synthetic validity marker, not a detection
            "eye_aspect_ratio": step.ear,
            "reprojection_error": step.reprojection_error,
            "head_pose": np.array([0.04 * np.sin(timestamp),
                                   0.12 * np.sin(timestamp * 0.5), 0.0]),
        }
        return step, float(angle0), float(angle1), int(step.valid), info


def _text(canvas, value, x, y, size=0.55, color=(220, 230, 240), bold=False):
    cv2.putText(canvas, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 2 if bold else 1, cv2.LINE_AA)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox without stretching a face or a FLAME render."""
    result = np.full((height, width, 3), (19, 25, 34), np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    x, y = (width - w) // 2, (height - h) // 2
    result[y:y + h, x:x + w] = resized
    return result


class DemoDashboard:
    """Fixed 1440x900 presentation, shared by replay and real webcam/video."""

    def __init__(self):
        self.history = deque(maxlen=180)
        self.events = deque(maxlen=7)
        self.first_timestamp: float | None = None

    def render(
        self, input_bgr: np.ndarray | None, head_bgr: np.ndarray | None,
        estimate: TemporalEstimate, assistive: AssistiveEstimate,
        camera_gaze: np.ndarray, *, source: str, stage: str,
        device: str = "cpu", crop_bgr: np.ndarray | None = None,
        fps: float = 0.0, progress: float | None = None,
        calibrated: bool = False,
    ) -> np.ndarray:
        if self.first_timestamp is None:
            self.first_timestamp = estimate.timestamp
        t = estimate.timestamp - self.first_timestamp
        self.history.append((t, estimate.raw_pitch, estimate.pitch, estimate.valid))
        for event in estimate.events:
            self.events.append((t, event["type"]))
        if assistive.event:
            self.events.append((t, "assistive_" + assistive.event))

        canvas = np.full((900, 1440, 3), (12, 18, 27), np.uint8)
        accent = (235, 195, 75)
        _text(canvas, "GazeFlame", 24, 40, 1.0, accent, True)
        _text(canvas, "REPORT DEMO / DRIVER ASSISTANCE", 245, 38, 0.65, bold=True)
        _text(canvas, source, 24, 70, 0.53, (110, 200, 250))
        keys = "c: calibrate | s: save | q: quit" if source.startswith("WEBCAM") else "s: snapshot | q/ESC: quit"
        _text(canvas, keys, 1050, 70, 0.44)
        _text(canvas, f"{device.upper()}  |  {fps:4.1f} fps  |  t={t:5.1f}s",
              1050, 38, 0.5)
        _text(canvas, stage[:102], 24, 100, 0.57)

        flow = ("RGB INPUT", "FACE + ALIGN", "GazeTR", "3D + FILTER",
                "FLAME", "INTENT / READY")
        for i, label in enumerate(flow):
            x = 24 + i * 233
            cv2.rectangle(canvas, (x, 117), (x + 214, 157), (40, 53, 68), -1)
            _text(canvas, label, x + 13, 144, 0.51, accent)
            if i < len(flow) - 1:
                _text(canvas, ">", x + 219, 144, 0.55)

        # Input and normalized network input, not a second alignment implementation.
        _text(canvas, "01 / DRIVER INPUT", 24, 188, 0.6, bold=True)
        if input_bgr is not None:
            canvas[204:500, 24:422] = _fit(input_bgr, 398, 296)
        else:
            _text(canvas, "SYNTHETIC GAZE REPLAY", 48, 286, 0.65, accent, True)
            _text(canvas, "No camera / no neural inference", 48, 320, 0.52)
            _text(canvas, "Scripted angles + EAR + confidence", 48, 350, 0.48)
            _text(canvas, "Exercises temporal + assistive logic", 48, 380, 0.47)
        if crop_bgr is not None:
            canvas[518:630, 24:136] = _fit(crop_bgr, 112, 112)
            _text(canvas, "NETWORK INPUT", 152, 543, 0.5, accent)
            _text(canvas, "ETH-XGaze normalization", 152, 572, 0.44)
            _text(canvas, "224 x 224 / same frame", 152, 600, 0.44)
        elif input_bgr is None:
            _text(canvas, "Evidence: angles / blink / tracking", 24, 542, 0.49)
            _text(canvas, "Script != real driver measurements", 24, 573, 0.49)
            _text(canvas, "Use --camera 0 for the full live chain", 24, 604, 0.46)
        else:
            _text(canvas, "NO VALID ALIGNED FACE", 24, 542, 0.54, (90, 140, 245))
            _text(canvas, "Face the camera in adequate light", 24, 573, 0.49)
            _text(canvas, "No intent without reliable tracking", 24, 604, 0.49)

        _text(canvas, "02 / FLAME HEAD + EYES", 442, 188, 0.6, bold=True)
        if head_bgr is not None:
            canvas[204:596, 442:834] = _fit(head_bgr, 392, 392)
        else:
            _text(canvas, "FLAME OFF / WAITING", 468, 372, 0.62, accent)
            _text(canvas, "Enable GL for the 3D panel", 468, 409, 0.49)
        _text(canvas, "TRACKING" if estimate.valid else "STALE / NO TRACKING",
              452, 625, 0.52, (100, 215, 125) if estimate.valid else (90, 140, 245))

        _text(canvas, "03 / ASSISTIVE CONTROL CONCEPT", 854, 188, 0.6, bold=True)
        canvas[204:644, 854:1416] = draw_assistive_panel(440, assistive, width=562)
        mapping = "PERSONAL CALIBRATION" if calibrated else "UNCALIBRATED / CONCEPT REGIONS"
        _text(canvas, mapping, 863, 655, 0.43, (135, 165, 195))

        # Live numerical evidence, not historical performance claims.
        _text(canvas, "GAZE VECTOR / CAMERA", 24, 674, 0.48, accent)
        _text(canvas, "[%+.3f, %+.3f, %+.3f]" % tuple(camera_gaze), 24, 702, 0.54)
        _text(canvas, f"Confidence {estimate.confidence:.2f}   Blink {estimate.blink:.2f}",
              24, 732, 0.51)
        _text(canvas, f"Speed {estimate.velocity_deg_s:5.1f} deg/s", 24, 762, 0.51)
        _text(canvas, f"Local dwell {assistive.fixation_duration:.2f}s", 24, 792, 0.51)
        _text(canvas, "Intent >=0.35s / Ready >=0.80s", 24, 822, 0.48, accent)

        # Trace uses GazeTR angle0 (horizontal), not the misleading legacy pitch name.
        _text(canvas, "ANGLE0 / ONE EURO FILTER", 442, 681, 0.48, accent)
        cv2.rectangle(canvas, (442, 696), (834, 815), (26, 36, 48), -1)
        cv2.line(canvas, (450, 755), (826, 755), (55, 65, 78), 1)
        rows = list(self.history)
        for col, color in ((1, (100, 115, 140)), (2, (235, 195, 75))):
            for i in range(1, len(rows)):
                if not rows[i - 1][3] or not rows[i][3]:
                    continue
                def point(index):
                    y = int(755 - np.clip(np.degrees(rows[index][col]), -45, 45) * 1.2)
                    return (450 + int(index * 376 / max(len(rows) - 1, 1)), y)
                cv2.line(canvas, point(i - 1), point(i), color, 1, cv2.LINE_AA)
        _text(canvas, "gray: raw     blue: filtered     +/-45 deg", 442, 837, 0.44)

        _text(canvas, "RECENT EVENTS", 854, 685, 0.48, accent)
        for i, (event_t, label) in enumerate(self.events):
            _text(canvas, f"{event_t:5.1f}s  {label}", 854, 710 + i * 19, 0.43)
        if progress is not None:
            cv2.rectangle(canvas, (24, 849), (1416, 853), (40, 53, 68), -1)
            end = 24 + int(np.clip(progress, 0, 1) * 1392)
            cv2.rectangle(canvas, (24, 849), (end, 853), accent, -1)
        _text(canvas, "SOFTWARE CONCEPT ONLY: no robot, flight controller or actuator commands.",
              24, 883, 0.51, (110, 160, 235), True)
        return canvas
