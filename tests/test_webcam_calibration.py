"""Verify profile replacement at the real entry point without a camera/GUI."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from common.calibration import CalibrationProfile


class WebcamCalibrationTest(unittest.TestCase):
    def run_camera(self, profile, destination, *, hotkey=False):
        import run_webcam

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        session = MagicMock()
        session.finished = False
        session.status = "LOOK AT TARGET"
        session.points = ((0.5, 0.5),)
        session.index = 0
        session.target = (0.5, 0.5)
        session.progress = 0.0

        def complete(*args, **kwargs):
            self.assertTrue(args[2])  # accepted measurement, not merely a face box
            self.assertGreaterEqual(kwargs["confidence"], 0.5)
            session.finished = True
            session.status = "COMPLETE"
            return profile

        session.update.side_effect = complete
        info = {"landmarks6": np.zeros((6, 2)), "rect": [100, 100, 300, 300],
                "reprojection_error": 0.8, "eye_aspect_ratio": 0.3,
                "normalization_rotation": np.eye(3)}
        args = ["--camera", "0", "--device", "cpu", "--demo", "--max-frames", "3"]
        args += ["--calibration-file" if hotkey else "--calibrate", str(destination)]
        with contextlib.ExitStack() as stack:
            for name, value in (
                ("flame_realtime_device", "cpu"), ("prepare_rocm_display_env", False),
                ("load_gazetr_model", MagicMock()), ("create_display_window", (1440, 900)),
                ("predict_gaze", (0.0, 0.0, 1, frame, info)),
                ("CalibrationSession", session),
            ):
                stack.enter_context(patch.object(run_webcam, name, return_value=value))
            stack.enter_context(patch.object(run_webcam.cv2, "VideoCapture", return_value=cap))
            stack.enter_context(patch.object(run_webcam.cv2, "imshow"))
            stack.enter_context(patch.object(run_webcam.cv2, "destroyAllWindows"))
            stack.enter_context(patch.object(run_webcam.cv2, "waitKey",
                                             side_effect=[ord("c"), -1, -1] if hotkey else [-1] * 3))
            save = stack.enter_context(patch.object(profile, "save", wraps=profile.save))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertEqual(run_webcam.main(args), 0)
        return save

    def test_accepted_profile_is_saved_once_and_hotkey_uses_loaded_path(self):
        profile = CalibrationProfile(np.array([[0, 2, 0.5], [-2, 0, 0.5]]),
                                     rmse=0.01, validation_rmse=0.02)
        for hotkey in (False, True):
            with self.subTest(hotkey=hotkey), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "personal.json"
                if hotkey:
                    CalibrationProfile(profile.matrix).save(path)
                saved = self.run_camera(profile, path, hotkey=hotkey)
                saved.assert_called_once_with(str(path))
                self.assertEqual(CalibrationProfile.load(path).validation_rmse, 0.02)

    def test_rejected_profile_cannot_overwrite_existing_file(self):
        profile = CalibrationProfile(np.array([[0, 2, 0.5], [-2, 0, 0.5]]),
                                     rmse=0.01, validation_rmse=0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "personal.json"
            original = b"existing personal calibration"
            path.write_bytes(original)
            saved = self.run_camera(profile, path)
            saved.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
