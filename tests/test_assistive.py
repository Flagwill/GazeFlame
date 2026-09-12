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
