from __future__ import annotations

import numpy as np
import pytest

from src.features.face_analyzer import FaceAnalyzer


class TestFaceAnalyzer:
    def test_analyze_empty_roi(self) -> None:
        fa = FaceAnalyzer(ear_threshold=0.21, mar_threshold=0.60)
        result = fa.analyze(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result["ear_left"] == 0.0
        assert result["ear_right"] == 0.0
        assert result["mar"] == 0.0
        assert result["eye_closed"] is True
        assert result["is_yawning"] is False
        assert result["landmarks_468"].shape == (468, 2)

    def test_analyze_returns_expected_keys(self) -> None:
        fa = FaceAnalyzer(ear_threshold=0.21, mar_threshold=0.60)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        result = fa.analyze(roi)
        expected_keys = {
            "ear_left",
            "ear_right",
            "mar",
            "pitch",
            "yaw",
            "roll",
            "eye_closed",
            "pose_valid",
            "is_yawning",
            "landmarks_468",
        }
        assert set(result.keys()) == expected_keys

    def test_analyze_no_face_in_blank_image(self) -> None:
        fa = FaceAnalyzer(ear_threshold=0.21, mar_threshold=0.60)
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        result = fa.analyze(roi)
        # Blank image has no face; MediaPipe returns zeros / defaults
        assert result["ear_left"] == 0.0
        assert result["ear_right"] == 0.0
        assert result["mar"] == 0.0
        assert result["pitch"] == 0.0
        assert result["yaw"] == 0.0
        assert result["roll"] == 0.0
        assert result["eye_closed"] is True
        assert result["is_yawning"] is False
