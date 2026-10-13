"""Signal-quality regressions; no camera, torch or display is required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.calibration import CalibrationProfile, CalibrationSession
from common.geometry import camera_gaze_to_pitch_yaw, camera_pitch_yaw_to_gaze
from common.temporal import GazeTemporalState, OneEuroFilter


def face_info(**changes):
    return {"landmarks6": np.zeros((6, 2)), "eye_aspect_ratio": 0.30,
            "reprojection_error": 0.8, **changes}


class TemporalAccuracyTest(unittest.TestCase):
    def test_normalization_changes_do_not_move_a_stationary_camera_gaze(self):
        state = GazeTemporalState()
        camera_gaze = camera_pitch_yaw_to_gaze(0.2, -0.3)
        for i in range(60):
            angle = np.radians(4.0 * (-1) ** i)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            pitch, yaw = camera_gaze_to_pitch_yaw(rotation @ camera_gaze)
            estimate = state.update(-yaw, pitch,
                                    info=face_info(normalization_rotation=rotation),
                                    timestamp=i / 15)
            np.testing.assert_allclose(estimate.camera_gaze, camera_gaze, atol=1e-10)
            np.testing.assert_allclose(rotation.T @ estimate.gaze, camera_gaze, atol=1e-10)
        self.assertGreater(estimate.fixation_duration, 2.0)

    def test_stationary_noise_is_reduced_at_camera_frame_rates(self):
        for fps in (8, 15, 30):
            with self.subTest(fps=fps):
                rng = np.random.default_rng(49)
                noise = rng.normal(0, np.radians(1.0), (fps * 10, 2))
                state = GazeTemporalState()
                filtered = []
                for i, (p, y) in enumerate(noise):
                    estimate = state.update(p, y, info=face_info(), timestamp=i / fps)
                    filtered.append(camera_gaze_to_pitch_yaw(estimate.camera_gaze))
                # Ignore initial acquisition. Both axes must improve, rather
                # than hiding one noisy axis in an averaged score.
                ratio = np.std(filtered[fps:], axis=0) / np.std(noise[fps:], axis=0)[::-1]
                self.assertTrue(np.all(ratio < 0.62), ratio)

    def test_sustained_new_target_is_confirmed_and_reached_promptly(self):
        for fps in (8, 15, 30):
            with self.subTest(fps=fps):
                state = GazeTemporalState()
                for i in range(fps):
                    state.update(0, 0, info=face_info(), timestamp=i / fps)
                reached = None
                events = set()
                for i in range(fps):
                    estimate = state.update(-np.radians(20), 0, info=face_info(),
                                            timestamp=1 + i / fps)
                    events.update(event["type"] for event in estimate.events)
                    yaw = camera_gaze_to_pitch_yaw(estimate.camera_gaze)[1]
                    if yaw >= np.radians(18) and reached is None:
                        reached = i / fps
                self.assertIsNotNone(reached)
                self.assertLessEqual(reached, 0.4)
                self.assertIn("saccade_start", events)

    def test_single_spike_cannot_contaminate_subsequent_frames(self):
        state = GazeTemporalState()
        baseline = state.update(0.1, 0.05, info=face_info(), timestamp=0)
        spike = state.update(-0.7, 0.5, info=face_info(), timestamp=0.1)
        self.assertFalse(spike.measurement_valid)
        self.assertLess(spike.confidence, 0.25)
        np.testing.assert_allclose(spike.camera_gaze, baseline.camera_gaze)
        recovered = state.update(0.1, 0.05, info=face_info(), timestamp=0.2)
        np.testing.assert_allclose(recovered.camera_gaze, baseline.camera_gaze)

    def test_blink_holds_gaze_from_the_first_closed_frame(self):
        state = GazeTemporalState()
        for i in range(20):
            baseline = state.update(0.1, 0.05, info=face_info(), timestamp=i / 15)
        closed = state.update(-0.7, 0.4, info=face_info(eye_aspect_ratio=0.12), timestamp=20 / 15)
        self.assertTrue(closed.blinking)
        self.assertFalse(closed.measurement_valid)
        self.assertEqual(closed.fixation_duration, 0)
        np.testing.assert_allclose(closed.camera_gaze, baseline.camera_gaze)
        for i in range(21, 30):
            reopened = state.update(0.1, 0.05, info=face_info(), timestamp=i / 15)
        self.assertFalse(reopened.blinking)
        np.testing.assert_allclose(reopened.camera_gaze, baseline.camera_gaze)

    def test_bad_geometry_and_tracker_frames_do_not_enter_filter(self):
        for changes in ({"reprojection_error": 24}, {"reprojection_error": np.nan},
                        {"source": "tracked", "tracking_quality": 3.0},
                        {"eye_aspect_ratio": np.nan},
                        {"normalization_rotation": np.full((3, 3), np.nan)}):
            with self.subTest(changes=changes):
                state = GazeTemporalState()
                good = state.update(0, 0, info=face_info(), timestamp=0)
                bad = state.update(0.5, 0.2, info=face_info(**changes), timestamp=0.1)
                self.assertFalse(bad.valid)
                np.testing.assert_allclose(bad.camera_gaze, good.camera_gaze)

    def test_long_loss_reacquires_without_dragging_old_target(self):
        state = GazeTemporalState()
        state.update(0, 0, info=face_info(), timestamp=0)
        state.update(0, 0, face_count=0, timestamp=1)
        found = state.update(-0.5, 0.2, info=face_info(), timestamp=1.1)
        self.assertTrue(found.measurement_valid)
        np.testing.assert_allclose(found.camera_gaze, camera_pitch_yaw_to_gaze(0.2, 0.5))
        self.assertEqual(found.fixation_duration, 0)

    def test_duplicate_clock_ticks_do_not_add_filter_motion(self):
        filt = OneEuroFilter()
        original = filt.update([0, 0], timestamp=1)
        np.testing.assert_equal(filt.update([1, 1], timestamp=1), original)
        np.testing.assert_equal(filt.update([1, 1], timestamp=0.5), original)


class CalibrationAccuracyTest(unittest.TestCase):
    @staticmethod
    def fit_session(scale=1.0, contaminate=False):
        session = CalibrationSession(settle_seconds=0.1, hold_seconds=0.6, min_samples=8)
        rng = np.random.default_rng(17)
        t = 0.0
        for frame in range(1000):
            x, y = session.target
            pitch = ((0.5 - y) * 0.3 + 0.05) * scale
            yaw = ((x - 0.5) * 0.4 - 0.02) * scale
            if contaminate:
                pitch, yaw = rng.normal([pitch, yaw], 0.001)
                if frame % 13 == 0:
                    pitch += 0.12
                    yaw -= 0.12
            profile = session.update(pitch, yaw, True, timestamp=t)
            if profile is not None:
                return session, profile
            t += 0.05
        raise AssertionError("calibration did not finish")

    def test_robust_calibration_predicts_targets_not_used_as_samples(self):
        session, profile = self.fit_session(contaminate=True)
        self.assertTrue(profile.is_usable())
        self.assertLess(profile.validation_rmse, 0.02)
        for x, y in ((0.27, 0.65), (0.72, 0.26), (0.5, 0.5)):
            predicted = profile.screen_point((0.5 - y) * 0.3 + 0.05,
                                             (x - 0.5) * 0.4 - 0.02)
            np.testing.assert_allclose(predicted, [x, y], atol=0.015)
        self.assertEqual(len(session.sample_counts), 9)

    def test_perfect_fit_with_tiny_angle_range_is_rejected(self):
        _session, profile = self.fit_session(scale=0.03)
        self.assertLess(profile.rmse, 1e-8)
        self.assertFalse(profile.is_usable())

    def test_bad_held_out_error_rejects_an_apparently_good_fit(self):
        profile = CalibrationProfile(np.array([[0, 2, 0.5], [-2, 0, 0.5]]),
                                     rmse=0.05, validation_rmse=0.3)
        self.assertFalse(profile.is_usable())

    def test_blinks_and_low_quality_do_not_supply_calibration_samples(self):
        for changes in ({"blinking": True}, {"confidence": 0.1},
                        {"velocity_deg_s": 120}, {"valid": False}):
            with self.subTest(changes=changes):
                session = CalibrationSession(settle_seconds=0, hold_seconds=0.5, min_samples=3)
                for t in np.arange(0, 0.4, 0.05):
                    session.update(0.1, 0.2, True, timestamp=t)
                kwargs = {"valid": True, **changes}
                session.update(1.0, 1.0, timestamp=0.4, **kwargs)
                for t in np.arange(0.45, 0.9, 0.05):
                    session.update(0.1, 0.2, True, timestamp=t)
                self.assertEqual(session.index, 0)
                for t in np.arange(0.9, 1.1, 0.05):
                    session.update(0.1, 0.2, True, timestamp=t)
                self.assertEqual(session.index, 1)

    def test_a_stalled_or_duplicate_clock_cannot_complete_a_hold(self):
        session = CalibrationSession(settle_seconds=0, hold_seconds=0.5, min_samples=3)
        for _ in range(20):
            session.update(0.1, 0.2, True, timestamp=0)
        session.update(0.1, 0.2, True, timestamp=4)
        self.assertEqual(session.index, 0)

    def test_noisy_target_is_repeated_instead_of_fitted(self):
        session = CalibrationSession(settle_seconds=0, hold_seconds=0.5, min_samples=8)
        for i in range(11):
            session.update(0.1 * (-1) ** i, 0.1 * (-1) ** i, True, timestamp=i / 20)
        self.assertEqual(session.index, 0)
        self.assertEqual(session.status, "POINT TOO NOISY / KEEP LOOKING")

    def test_profile_round_trip_and_old_camera_profiles_still_work(self):
        _, profile = self.fit_session()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            loaded = CalibrationProfile.load(path)
            self.assertTrue(loaded.is_usable())
            self.assertEqual(loaded.sample_counts, profile.sample_counts)
            data = profile.to_dict()
            data["version"] = 2
            for key in ("validation_rmse", "sample_counts", "point_spread_deg"):
                data.pop(key)
            path.write_text(json.dumps(data))
            old = CalibrationProfile.load(path)
            self.assertTrue(old.is_usable())
            np.testing.assert_allclose(old.screen_point(0.05, -0.02), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
