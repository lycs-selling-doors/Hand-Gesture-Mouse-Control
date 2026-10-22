"""Regression checks for the Tasks migration; never open a camera or move a mouse."""

from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app import download_model
from app.hand_tracking import HandTracker, draw_landmarks
from app.model import DEFAULT_MODEL_PATH


class TrackerTests(unittest.TestCase):
    def test_missing_model_explains_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "python -m app.download_model"):
                HandTracker(Path(directory) / "missing.task")

    def test_video_timestamps_increase_when_clock_repeats_or_moves_back(self):
        with tempfile.NamedTemporaryFile() as model:
            with patch("app.hand_tracking.vision.HandLandmarker.create_from_options") as create:
                with HandTracker(model.name) as tracker:
                    with patch("app.hand_tracking.time.monotonic_ns", side_effect=[5_000_000, 5_000_000, 4_000_000]):
                        for _ in range(3):
                            tracker.process(np.zeros((8, 8, 3), dtype=np.uint8))
                timestamps = [call.args[1] for call in create.return_value.detect_for_video.call_args_list]
                self.assertEqual(timestamps, [5, 6, 7])
                create.return_value.close.assert_called_once()

    def test_landmarker_closes_on_processing_failure(self):
        with tempfile.NamedTemporaryFile() as model:
            with patch("app.hand_tracking.vision.HandLandmarker.create_from_options") as create:
                create.return_value.detect_for_video.side_effect = RuntimeError("inference failed")
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    with HandTracker(model.name) as tracker:
                        tracker.process(np.zeros((8, 8, 3), dtype=np.uint8))
                create.return_value.close.assert_called_once()

    @unittest.skipUnless(DEFAULT_MODEL_PATH.is_file(), "run python -m app.download_model for real model inference")
    def test_real_model_accepts_consecutive_video_frames(self):
        with HandTracker(DEFAULT_MODEL_PATH) as tracker:
            for _ in range(3):
                result = tracker.process(np.zeros((240, 320, 3), dtype=np.uint8))
                self.assertEqual(result.hand_landmarks, [])

    def test_tasks_landmarks_draw_with_opencv(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [SimpleNamespace(x=0.1 + i * 0.03, y=0.2 + i * 0.02) for i in range(21)]
        draw_landmarks(frame, landmarks)
        self.assertTrue(np.any(frame))
        self.assertEqual(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).shape, frame.shape)


class DownloadTests(unittest.TestCase):
    def test_cached_verified_model_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.task"
            path.write_bytes(b"verified")
            with patch.object(download_model, "MODEL_SHA256", hashlib.sha256(b"verified").hexdigest()):
                with patch.object(download_model, "urlopen") as request:
                    self.assertEqual(download_model.download_model(path), path)
                    request.assert_not_called()

    def test_checksum_failure_preserves_existing_model_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.task"
            path.write_bytes(b"existing")
            with patch.object(download_model, "urlopen", return_value=io.BytesIO(b"corrupted")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_model.download_model(path)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).glob("*.download")), [])

    def test_verified_download_replaces_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.task"
            with patch.object(download_model, "MODEL_SHA256", hashlib.sha256(b"new model").hexdigest()):
                with patch.object(download_model, "urlopen", return_value=io.BytesIO(b"new model")):
                    download_model.download_model(path)
            self.assertEqual(path.read_bytes(), b"new model")


class CameraLifecycleTests(unittest.TestCase):
    def setUp(self):
        # A mock PyAutoGUI import keeps this test usable without a desktop/display.
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            import importlib.util
            spec = importlib.util.spec_from_file_location("app._gesture_under_test",
                Path(__file__).resolve().parents[1] / "app" / "hand_gesture.py")
            self.gesture = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.gesture)
        self.gesture.pyautogui.size.return_value = (1920, 1080)

    def test_missing_camera_releases_capture_and_tracker(self):
        with ExitStack() as stack:
            tracker = stack.enter_context(patch.object(self.gesture, "HandTracker"))
            capture = stack.enter_context(patch.object(self.gesture.cv2, "VideoCapture")).return_value
            destroy = stack.enter_context(patch.object(self.gesture.cv2, "destroyAllWindows"))
            capture.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
                self.gesture.start_hand_gesture_control()
            capture.release.assert_called_once()
            destroy.assert_called_once()
            tracker.return_value.__exit__.assert_called_once()

    def test_capture_exception_releases_resources(self):
        with ExitStack() as stack:
            tracker = stack.enter_context(patch.object(self.gesture, "HandTracker"))
            capture = stack.enter_context(patch.object(self.gesture.cv2, "VideoCapture")).return_value
            stack.enter_context(patch.object(self.gesture.cv2, "namedWindow"))
            destroy = stack.enter_context(patch.object(self.gesture.cv2, "destroyAllWindows"))
            capture.isOpened.return_value = True
            capture.read.side_effect = RuntimeError("capture failed")
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                self.gesture.start_hand_gesture_control()
            capture.release.assert_called_once()
            destroy.assert_called_once()
            tracker.return_value.__exit__.assert_called_once()

    def test_one_frame_quit_uses_tasks_results_without_mouse_actions(self):
        with ExitStack() as stack:
            tracker = stack.enter_context(patch.object(self.gesture, "HandTracker"))
            tracker.return_value.__enter__.return_value.process.return_value = SimpleNamespace(hand_landmarks=[])
            capture = stack.enter_context(patch.object(self.gesture.cv2, "VideoCapture")).return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
            stack.enter_context(patch.object(self.gesture.cv2, "namedWindow"))
            stack.enter_context(patch.object(self.gesture.cv2, "imshow"))
            stack.enter_context(patch.object(self.gesture.cv2, "waitKey", return_value=ord("q")))
            stack.enter_context(patch.object(self.gesture.cv2, "destroyAllWindows"))
            self.gesture.start_hand_gesture_control()
            capture.release.assert_called_once()
            self.gesture.pyautogui.moveTo.assert_not_called()
            self.gesture.pyautogui.click.assert_not_called()


    def test_tasks_landmark_list_still_triggers_original_left_click(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        for index, x, y in [(4, 0.3, 0.3), (8, 0.3, 0.3), (12, 0.8, 0.8),
                            (16, 0.9, 0.8), (20, 0.8, 0.9)]:
            landmarks[index] = SimpleNamespace(x=x, y=y)
        with ExitStack() as stack:
            tracker = stack.enter_context(patch.object(self.gesture, "HandTracker"))
            tracker.return_value.__enter__.return_value.process.return_value = SimpleNamespace(hand_landmarks=[landmarks])
            capture = stack.enter_context(patch.object(self.gesture.cv2, "VideoCapture")).return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            stack.enter_context(patch.object(self.gesture, "MouseControl"))
            stack.enter_context(patch.object(self.gesture.cv2, "namedWindow"))
            stack.enter_context(patch.object(self.gesture.cv2, "imshow"))
            stack.enter_context(patch.object(self.gesture.cv2, "waitKey", return_value=ord("q")))
            stack.enter_context(patch.object(self.gesture.cv2, "destroyAllWindows"))
            self.gesture.last_gesture_time = 0
            self.gesture.start_hand_gesture_control()
            self.gesture.pyautogui.click.assert_called_once_with(button="left")
            self.gesture.pyautogui.scroll.assert_not_called()
            self.gesture.pyautogui.doubleClick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
