from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

YOLO_CLASS_MAP = {0: "face", 1: "phone", 2: "cigarette", 3: "seatbelt"}

GROUPED_CLASSES = (
    "face",
    "phone",
    "cigarette",
    "smoke",
    "seatbelt",
    "bottle",
    "cup",
    "drink",
    "water",
)

CLASS_ALIASES: Dict[str, str] = {
    "face": "face",
    "driver_face": "face",
    "head": "face",
    "phone": "phone",
    "cell phone": "phone",
    "cellphone": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "telephone": "phone",
    "cigarette": "cigarette",
    "cigar": "cigarette",
    "smoking": "cigarette",
    "smoke": "smoke",
    "seatbelt": "seatbelt",
    "seat_belt": "seatbelt",
    "safety belt": "seatbelt",
    "bottle": "bottle",
    "water bottle": "bottle",
    "cup": "cup",
    "mug": "cup",
    "drink": "drink",
    "drinking": "drink",
    "water": "water",
}

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
    "eyes_closed":            {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "eyes_closed_head_left":  {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": -30.0, "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "eyes_closed_head_right": {"ear_left": _EAR_CLOSED, "ear_right": _EAR_CLOSED, "mar": 0.0,  "pitch": 0.0,   "yaw": 30.0,  "roll": 0.0, "eye_closed": True,  "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "focused":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "head_down":              {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": -25.0, "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "head_up":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 25.0,  "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "seeing_left":            {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": -30.0, "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "seeing_right":           {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.0,  "pitch": 0.0,   "yaw": 30.0,  "roll": 0.0, "eye_closed": False, "is_yawning": False, "landmarks_468": None, "pose_valid": True},
    "yarning":                {"ear_left": _EAR_OPEN,   "ear_right": _EAR_OPEN,   "mar": 0.70, "pitch": 0.0,   "yaw": 0.0,   "roll": 0.0, "eye_closed": False, "is_yawning": True,  "landmarks_468": None, "pose_valid": True},
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

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        imgsz: int | None = None,
        class_ids: List[int] | None = None,
        class_conf_thresholds: Mapping[str, float] | None = None,
    ):
        self.conf_threshold = conf_threshold
        self.class_conf_thresholds = {str(k): float(v) for k, v in (class_conf_thresholds or {}).items()}
        self.device = device
        self.imgsz = imgsz
        self.class_ids = class_ids
        self.use_real_model = False
        self.behavior_mode = False
        self.model = None
        self.class_map: Dict[int, str] = dict(YOLO_CLASS_MAP)

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
                    self.class_map = self._build_class_map(self.model.names)
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

    @staticmethod
    def _canonical_class_name(name: str) -> str | None:
        key = str(name).strip().lower().replace("-", "_")
        key = " ".join(key.replace("_", " ").split())
        return CLASS_ALIASES.get(key)

    @classmethod
    def _build_class_map(cls, names: Mapping[int, str] | List[str]) -> Dict[int, str]:
        class_map: Dict[int, str] = {}
        items = names.items() if hasattr(names, "items") else enumerate(names)
        for idx, name in items:
            canonical = cls._canonical_class_name(str(name))
            if canonical is not None:
                class_map[int(idx)] = canonical
        return class_map or dict(YOLO_CLASS_MAP)

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
            kwargs = {
                "verbose": False,
                "conf": self.conf_threshold,
                "device": self.device,
            }
            if self.imgsz is not None:
                kwargs["imgsz"] = self.imgsz
            if self.class_ids is not None:
                kwargs["classes"] = self.class_ids
            results = self.model(frame, **kwargs)
            result = results[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = self.class_map.get(class_id)
                    min_conf = self.class_conf_thresholds.get(class_name or "", self.conf_threshold)
                    if confidence < min_conf:
                        continue
                    detections.append(Detection(class_id=class_id, confidence=confidence, bbox=(int(x1), int(y1), int(x2), int(y2))))
            return detections
        except Exception as e:
            logger.error(f"Real inference failed: {e}, falling back to synthetic")
            return self._detect_synthetic(frame)

    def _detect_synthetic(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        return [Detection(class_id=0, confidence=0.9, bbox=(w // 4, h // 6, w // 2, (h * 2) // 3))]

    @staticmethod
    def _xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def group_detections(self, detections: List[Detection]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        return self.to_grouped_dict(detections, self.behavior_mode, self.class_map)

    @staticmethod
    def to_grouped_dict(
        detections: List[Detection],
        behavior_mode: bool = False,
        class_map: Mapping[int, str] | None = None,
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Group detections by class name.

        In behavior mode all detections are treated as face bboxes for ByteTracker;
        phone/cigarette/seatbelt groups remain empty.
        """
        resolved_map = class_map or YOLO_CLASS_MAP
        grouped_classes = GROUPED_CLASSES if class_map is not None else tuple(YOLO_CLASS_MAP.values())
        grouped: Dict[str, List[Tuple[int, int, int, int]]] = {k: [] for k in grouped_classes}
        if behavior_mode:
            for det in detections:
                grouped["face"].append(det.bbox)
        else:
            for det in detections:
                if det.class_id in resolved_map:
                    class_name = resolved_map[det.class_id]
                    if class_name not in grouped:
                        grouped[class_name] = []
                    if class_name == "face":
                        grouped[class_name].append(det.bbox)
                    else:
                        grouped[class_name].append(YOLODetector._xyxy_to_xywh(det.bbox))
        return grouped
