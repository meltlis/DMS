from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

YOLO_CLASS_MAP = {0: "face", 1: "phone", 2: "cigarette", 3: "seatbelt"}

# 9-class behavior model labels (train16/best.pt)
BEHAVIOR_CLASS_MAP = {
    0: "eyes_closed",
    1: "eyes_closed_head_left",
    2: "eyes_closed_head_right",
    3: "focused",
    4: "head_down",
    5: "head_up",
    6: "seeing_left",
    7: "seeing_right",
    8: "yarning",
}

_EAR_CLOSED = 0.10
_EAR_OPEN = 0.32

# Canonical feature values for each behavior class
BEHAVIOR_FEATURES: Dict[str, dict] = {
    "eyes_closed":            {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None},
    "eyes_closed_head_left":  {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": -30.0, "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None},
    "eyes_closed_head_right": {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": 30.0,  "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None},
    "focused":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None},
    "head_down":              {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": -25.0, "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None},
    "head_up":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 25.0,  "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None},
    "seeing_left":            {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": -30.0, "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None},
    "seeing_right":           {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": 30.0,  "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None},
    "yarning":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.70, "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": True,  "landmarks_468": None},
}


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]


class YOLODetector:
    """YOLOv11 detector with fallback to enhanced synthetic mode.

    When loaded with a 9-class behavior model (train16/best.pt), ``behavior_mode``
    is True and ``get_behavior_features()`` returns canonical feature dicts
    directly — the pipeline skips MediaPipe for those frames.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu", conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.device = device
        self.use_real_model = False
        self.behavior_mode = False
        self.model = None

        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                self.model.to(device)
                self.use_real_model = True
                # Detect whether this is the 9-class behavior model
                try:
                    nc = len(self.model.names)
                    self.behavior_mode = (nc == 9)
                except Exception:
                    pass
                mode = "behavior (9-class)" if self.behavior_mode else "standard (4-class)"
                logger.info(f"YOLO model loaded from {model_path} — {mode} mode")
            except Exception as e:
                logger.warning(f"Failed to load real YOLO model: {e}. Using enhanced synthetic mode.")

        if not self.use_real_model:
            logger.info("Using enhanced synthetic detector mode (fallback for YOLO weights)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.use_real_model and self.model is not None:
            return self._detect_real(frame)
        return self._detect_synthetic(frame)

    def get_behavior_features(self, detections: List[Detection]) -> Optional[dict]:
        """Return feature dict for the highest-confidence behavior detection.

        Returns None when not in behavior mode or when no detection is available.
        """
        if not self.behavior_mode or not detections:
            return None
        best = max(detections, key=lambda d: d.confidence)
        class_name = BEHAVIOR_CLASS_MAP.get(best.class_id)
        if class_name is None:
            return None
        return dict(BEHAVIOR_FEATURES[class_name])

    def _detect_real(self, frame: np.ndarray) -> List[Detection]:
        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold, device=self.device)
            result = results[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    detections.append(Detection(class_id=class_id, confidence=confidence, bbox=(int(x1), int(y1), int(x2), int(y2))))
            return detections
        except Exception as e:
            logger.error(f"Real inference failed: {e}, falling back to synthetic")
            return self._detect_synthetic(frame)

    def _detect_synthetic(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        return [Detection(class_id=0, confidence=0.99, bbox=(0, 0, w, h))]

    @staticmethod
    def to_grouped_dict(
        detections: List[Detection],
        behavior_mode: bool = False,
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Group detections by class name.

        In behavior mode all detections are treated as face bboxes for ByteTracker;
        phone/cigarette/seatbelt groups remain empty.
        """
        grouped: Dict[str, List[Tuple[int, int, int, int]]] = {k: [] for k in YOLO_CLASS_MAP.values()}
        if behavior_mode:
            for det in detections:
                grouped["face"].append(det.bbox)
        else:
            for det in detections:
                if det.class_id in YOLO_CLASS_MAP:
                    grouped[YOLO_CLASS_MAP[det.class_id]].append(det.bbox)
        return grouped
