"""Small adapter for the supported MediaPipe Hand Landmarker API."""

from enum import IntEnum
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20


class HandTracker:
    def __init__(self, model_path, num_hands=2):
        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {model_path}. "
                "Run: python -m app.download_model"
            )
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, rgb_frame):
        # VIDEO mode requires strictly increasing timestamps, even for fast frames.
        timestamp_ms = max(time.monotonic_ns() // 1_000_000,
                           self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def draw_landmarks(frame, landmarks):
    """Draw Tasks landmarks without depending on removed mp.solutions helpers."""
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end],
                 (0, 200, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)
