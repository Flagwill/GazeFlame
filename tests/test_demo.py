"""Report replay and dashboard checks; no torch weights, camera or GL needed."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from common.assistive import AssistiveIntentState
from common.calibration import camera_gaze_to_screen_point
from common.demo import DemoDashboard, ReportReplay
from common.geometry import camera_gaze_to_pitch_yaw
from common.temporal import GazeTemporalState


class ReportDemoTest(unittest.TestCase):
    def test_replay_covers_report_states_and_all_regions(self):
        replay = ReportReplay()
        temporal = GazeTemporalState()
        intent = AssistiveIntentState()
        statuses, ready_targets, events = set(), set(), set()
        for t in np.arange(0, replay.duration, 1 / 15):
            _step, p, y, faces, info = replay.sample(float(t))
            estimate = temporal.update(p, y, faces, info, timestamp=float(t))
            pitch, yaw = camera_gaze_to_pitch_yaw(estimate.gaze)
            estimate.screen_point = camera_gaze_to_screen_point(pitch, yaw)
            assistance = intent.update(
                estimate.screen_point, estimate.valid, estimate.confidence,
                estimate.fixation_duration, estimate.velocity_deg_s,
                estimate.blinking, timestamp=float(t),
            )
            statuses.add(assistance.status)
            if assistance.status == "ASSIST_READY":
                ready_targets.add(assistance.target.key)
                self.assertGreaterEqual(assistance.fixation_duration, 0.8)
                self.assertTrue(estimate.valid)
                self.assertFalse(estimate.blinking)
            events.update(event["type"] for event in estimate.events)
        self.assertTrue({"MONITORING", "INTENT_DETECTED", "ASSIST_READY", "PAUSED",
                         "NO_TRACKING"}.issubset(statuses))
        self.assertEqual(ready_targets, {"attitude", "system", "navigation", "control"})
        self.assertTrue({"face_found", "face_lost", "blink_start", "blink_end",
                         "saccade_start", "fixation_start"}.issubset(events))

    def test_dashboard_dimensions_and_bounded_history(self):
        dashboard = DemoDashboard()
        temporal = GazeTemporalState()
        intent = AssistiveIntentState()
        estimate = temporal.update(0.0, 0.0, 0, timestamp=0.0)
        assistance = intent.update(None, False, 0.0)
        image = np.zeros((224, 224, 3), np.uint8)
        for source, input_frame, head, crop in (
            ("SYNTHETIC REPLAY", None, None, None),
            ("WEBCAM / REAL INFERENCE", image, image, image),
        ):
            result = dashboard.render(input_frame, head, estimate, assistance, estimate.gaze,
                                      source=source, stage="Test", crop_bgr=crop)
            self.assertEqual(result.shape, (900, 1440, 3))
            self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(dashboard.history.maxlen, 180)
        self.assertEqual(dashboard.events.maxlen, 7)

    def test_webcam_entry_delegates_to_full_real_pipeline(self):
        import run_demo

        with tempfile.TemporaryDirectory() as output:
            with patch.dict("sys.modules", {"run_webcam": SimpleNamespace(main=lambda args: args)}):
                args = run_demo.main(["--camera", "0", "--output", output,
                                      "--calibration-file", "profiles/test.json"])
        self.assertIn("--demo", args)
        self.assertIn("--flame", args)
        self.assertEqual(args[args.index("--camera") + 1], "0")
        self.assertEqual(args[args.index("--device") + 1], "cpu")
        self.assertIn("--calibration-file", args)
        self.assertNotIn("--skip-image", args)

    def test_replay_is_deterministic(self):
        a, b = ReportReplay(), ReportReplay()
        for t in (0.0, 1.5, 10.0, 18.0):
            result_a, result_b = a.sample(t), b.sample(t)
            self.assertEqual(result_a[:4], result_b[:4])


if __name__ == "__main__":
    unittest.main()
