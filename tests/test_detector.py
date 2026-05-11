from __future__ import annotations

import numpy as np
import pytest

from src.perception.detector import Detection, YOLODetector, YOLO_CLASS_MAP


class TestYOLODetector:
    def test_class_map_has_four_classes(self) -> None:
        assert len(YOLO_CLASS_MAP) == 4
        assert YOLO_CLASS_MAP[0] == "face"
        assert YOLO_CLASS_MAP[1] == "phone"
        assert YOLO_CLASS_MAP[2] == "cigarette"
        assert YOLO_CLASS_MAP[3] == "seatbelt"

    def test_detect_returns_single_face_placeholder(self) -> None:
        detector = YOLODetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.9)
        x, y, w, h = dets[0].bbox
        assert w == 640 // 2
        assert h == (480 * 2) // 3
        assert x == 640 // 4
        assert y == 480 // 6

    def test_to_grouped_dict_empty(self) -> None:
        grouped = YOLODetector.to_grouped_dict([])
        assert grouped == {"face": [], "phone": [], "cigarette": [], "seatbelt": []}

    def test_to_grouped_dict_multiple_classes(self) -> None:
        dets = [
            Detection(0, 0.9, (10, 10, 50, 50)),
            Detection(1, 0.8, (100, 100, 30, 30)),
            Detection(1, 0.7, (200, 200, 30, 30)),
            Detection(3, 0.6, (0, 0, 10, 10)),
        ]
        grouped = YOLODetector.to_grouped_dict(dets)
        assert len(grouped["face"]) == 1
        assert len(grouped["phone"]) == 2
        assert len(grouped["cigarette"]) == 0
        assert len(grouped["seatbelt"]) == 1
