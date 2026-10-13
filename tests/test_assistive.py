import unittest

import numpy as np

from common.assistive import AssistiveIntentState, draw_assistive_panel


class AssistiveIntentStateTest(unittest.TestCase):
    def test_fixation_progression_is_explainable(self):
        state = AssistiveIntentState()

        monitoring = state.update(
            [0.5, 0.3], valid=True, confidence=0.9, fixation_duration=0.1
        )
        self.assertEqual(monitoring.target.key, "attitude")
        self.assertEqual(monitoring.status, "MONITORING")

        intent = state.update(
            [0.5, 0.3], valid=True, confidence=0.9, fixation_duration=0.5
        )
        self.assertEqual(intent.status, "INTENT_DETECTED")

        ready = state.update(
            [0.5, 0.3], valid=True, confidence=0.9, fixation_duration=1.0
        )
        self.assertEqual(ready.status, "ASSIST_READY")

    def test_invalid_tracking_does_not_claim_assistance(self):
        state = AssistiveIntentState()
        estimate = state.update(
            np.array([0.5, 0.5]), valid=False, confidence=0.0, fixation_duration=2.0
        )
        self.assertIsNone(estimate.target)
        self.assertEqual(estimate.status, "NO_TRACKING")

    def test_new_region_cannot_inherit_long_fixation(self):
        state = AssistiveIntentState()
        state.update([0.5, 0.3], True, 0.9, 10.0, timestamp=0.0)
        ready = state.update([0.5, 0.3], True, 0.9, 11.0, timestamp=1.0)
        self.assertEqual(ready.status, "ASSIST_READY")
        switched = state.update([0.83, 0.3], True, 0.9, 11.1, timestamp=1.1)
        next_frame = state.update([0.83, 0.3], True, 0.9, 11.2, timestamp=1.2)
        self.assertEqual(switched.status, "MONITORING")
        self.assertEqual(next_frame.status, "MONITORING")
        self.assertAlmostEqual(next_frame.fixation_duration, 0.1)
        ready = state.update([0.83, 0.3], True, 0.9, 12.0, timestamp=2.0)
        self.assertEqual(ready.status, "ASSIST_READY")

    def test_reset_without_timestamps(self):
        state = AssistiveIntentState()
        state.update([0.5, 0.3], True, 0.9, 10.0)
        state.update([0.5, 0.3], True, 0.9, 11.0)
        state.update([0.83, 0.3], True, 0.9, 11.1)
        next_frame = state.update([0.83, 0.3], True, 0.9, 11.2)
        self.assertEqual(next_frame.status, "MONITORING")

    def test_blink_loss_and_saccade_reset_evidence(self):
        for interruption in ({"blinking": True}, {"valid": False},
                             {"confidence": 0.1}, {"velocity_deg_s": 120.0}):
            with self.subTest(interruption=interruption):
                state = AssistiveIntentState()
                state.update([0.5, 0.3], True, 0.9, 5.0, timestamp=0.0)
                state.update([0.5, 0.3], True, 0.9, 6.0, timestamp=1.0)
                inputs = dict(point=[0.5, 0.3], valid=True, confidence=0.9,
                              fixation_duration=6.1, timestamp=1.1)
                inputs.update(interruption)
                interrupted = state.update(**inputs)
                self.assertNotEqual(interrupted.status, "ASSIST_READY")
                recovered = state.update([0.5, 0.3], True, 0.9, 6.2, timestamp=1.2)
                self.assertEqual(recovered.status, "MONITORING")
                self.assertEqual(recovered.fixation_duration, 0.0)

    def test_nonfinite_or_outside_evidence_is_not_valid(self):
        for point in ([np.nan, 0.3], [0.5], [1.1, 0.3]):
            state = AssistiveIntentState()
            result = state.update(point, True, 0.9, 2.0)
            self.assertEqual(result.status, "NO_TRACKING")
            self.assertFalse(result.valid)
        state = AssistiveIntentState()
        self.assertEqual(state.update([0.5, 0.3], True, np.nan, 2.0).status,
                         "NO_TRACKING")

    def test_blink_on_new_target_is_paused_not_monitoring(self):
        state = AssistiveIntentState()
        self.assertEqual(state.update([0.5, 0.3], True, 0.9, 2.0,
                                      blinking=True).status, "PAUSED")

    def test_panel_has_stable_dimensions(self):
        state = AssistiveIntentState()
        estimate = state.update(
            [0.5, 0.3], valid=True, confidence=0.9, fixation_duration=0.5
        )
        panel = draw_assistive_panel(480, estimate)
        self.assertEqual(panel.shape[0], 480)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreaterEqual(panel.shape[1], 300)


if __name__ == "__main__":
    unittest.main()
