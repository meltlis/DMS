from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np
import yaml

from src.decision.danger import DangerDetector, iou
from src.decision.fsm import PipelineFSM, FatigueStateFSM, TimedStateFSM
from src.decision.rules import state_from_metrics
from src.decision.lstm_classifier import LSTMClassifier
from src.features.face_analyzer import FaceAnalyzer
from src.perception.detector import YOLODetector
from src.perception.tracker import ByteTrackerWrapper
from src.temporal.aggregator import TemporalAggregator


_AUX_OBJECT_CLASSES = ("phone", "bottle", "cup", "drink", "water")
_AUX_COCO_CLASS_IDS = [39, 41, 67]  # bottle, cup, cell phone
_SMOKE_OBJECT_CLASSES = ("cigarette", "smoke")
_BEHAVIOR_DISTRACTIONS = {
    "PHONE_CALL",
    "PHONE_USE",
    "PHONE",
    "SUSPECTED_PHONE_USE",
    "DRINK",
    "SMOKE",
    "HEAD_DOWN",
    "LOOK_AROUND",
    "GAZE_AWAY",
}
_OBJECT_BEHAVIOR_DISTRACTIONS = {"PHONE_CALL", "PHONE_USE", "PHONE", "DRINK", "SMOKE"}
_POSE_DISTRACTIONS = {"HEAD_DOWN", "LOOK_AROUND", "GAZE_AWAY", "SUSPECTED_PHONE_USE"}
_SMOOTHED_OBJECT_CLASSES = ("phone", "bottle", "cup", "drink", "water", "cigarette", "smoke")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: object | None, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path


def resolve_external_project_root(runtime: Dict[str, Any] | None = None) -> Path | None:
    """Find the optional YOLO+sequence-model project next to this repo."""
    runtime = runtime or {}
    candidates: list[Path] = []
    configured = _resolve_path(runtime.get("external_project_dir"), base=_workspace_root())
    env_path = _resolve_path(os.environ.get("DMS_EXTERNAL_PROJECT_DIR"), base=_workspace_root())
    for path in (configured, env_path):
        if path is not None:
            candidates.append(path)

    workspace = _workspace_root()
    candidates.extend(
        [
            workspace / "Drowsiness-Detection-based-on-yolo11-and-LSTM-main",
            workspace / "Drowsiness-Detection-based-on-yolo11-and-LSTM",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_optional_model_path(
    value: str | None,
    *,
    project_root: Path,
    external_root: Path | None,
    external_default: str,
) -> Path | None:
    configured = _resolve_path(value, base=project_root)
    if configured is not None and configured.exists():
        return configured

    if external_root is not None:
        external_path = external_root / external_default
        if external_path.exists():
            return external_path
    return None


def resolve_sequence_model_path(runtime: Dict[str, Any], external_root: Path | None) -> Path | None:
    project_root = _project_root()
    configured = _resolve_path(runtime.get("sequence_model_path"), base=project_root)
    if configured is not None and configured.exists():
        return configured

    if external_root is None:
        return None
    for name in ("lstm_model.pth", "transformer_model.pth"):
        candidate = external_root / name
        if candidate.exists():
            return candidate
    return None


class ObjectBoxSmoother:
    """Tiny class-wise object tracker for short detection dropouts.

    It is intentionally simpler than a full multi-object tracker: the DMS
    behavior rules only need stable phone/cup/bottle boxes around one driver.
    """

    def __init__(self, ttl_frames: int = 5, alpha: float = 0.65) -> None:
        self.ttl_frames = max(0, int(ttl_frames))
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self._tracks: Dict[str, list[dict[str, object]]] = {key: [] for key in _SMOOTHED_OBJECT_CLASSES}

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _center_dist(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay = ObjectBoxSmoother._center(a)
        bx, by = ObjectBoxSmoother._center(b)
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    def reset(self) -> None:
        for tracks in self._tracks.values():
            tracks.clear()

    def _match_score(self, new_box: tuple[int, int, int, int], old_box: tuple[int, int, int, int]) -> float:
        overlap = iou(new_box, old_box)
        if overlap > 0.0:
            return overlap + 1.0
        max_size = max(new_box[2], new_box[3], old_box[2], old_box[3], 1)
        dist = self._center_dist(new_box, old_box)
        if dist <= max_size * 0.90:
            return max(0.0, 1.0 - dist / (max_size * 0.90))
        return 0.0

    def _smooth_box(
        self,
        old_box: tuple[int, int, int, int],
        new_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return tuple(
            int(round((1.0 - self.alpha) * old + self.alpha * new))
            for old, new in zip(old_box, new_box)
        )

    def update(self, grouped: Dict[str, list]) -> Dict[str, list]:
        smoothed = {key: list(value) for key, value in grouped.items()}

        for class_name in _SMOOTHED_OBJECT_CLASSES:
            new_boxes = [tuple(map(int, box)) for box in grouped.get(class_name, [])]
            tracks = self._tracks.setdefault(class_name, [])
            used_tracks: set[int] = set()
            next_tracks: list[dict[str, object]] = []

            for new_box in new_boxes:
                best_idx = -1
                best_score = 0.0
                for idx, track in enumerate(tracks):
                    if idx in used_tracks:
                        continue
                    old_box = track["box"]
                    score = self._match_score(new_box, old_box)  # type: ignore[arg-type]
                    if score > best_score:
                        best_idx = idx
                        best_score = score

                if best_idx >= 0 and best_score >= 0.20:
                    old_box = tracks[best_idx]["box"]  # type: ignore[assignment]
                    box = self._smooth_box(old_box, new_box)  # type: ignore[arg-type]
                    used_tracks.add(best_idx)
                else:
                    box = new_box
                next_tracks.append({"box": box, "missed": 0})

            for idx, track in enumerate(tracks):
                if idx in used_tracks:
                    continue
                missed = int(track.get("missed", 0)) + 1
                if missed <= self.ttl_frames:
                    next_tracks.append({"box": track["box"], "missed": missed})

            self._tracks[class_name] = next_tracks
            smoothed[class_name] = [track["box"] for track in next_tracks]

        return smoothed


class DMSPipeline:
    def __init__(self, thresholds: Dict[str, float], runtime: Dict[str, Any]) -> None:
        self.thresholds = thresholds
        self.runtime = runtime
        project_root = _project_root()
        external_root = resolve_external_project_root(runtime)
        # Prefer the local 4-class detector fine-tuned for DMS objects, including phone.
        dms4class_model = project_root / "runs" / "detect" / "dms4class" / "weights" / "best.pt"
        behavior_model = resolve_optional_model_path(
            runtime.get("behavior_model_path"),
            project_root=project_root,
            external_root=external_root,
            external_default="runs/detect/train16/weights/best.pt",
        )
        standard_model = project_root / "weights" / "yolov11n.pt"
        aux_model = Path(str(thresholds.get("aux_model_path", "")))
        if not aux_model.is_absolute():
            aux_model = project_root / aux_model
        if not aux_model.exists():
            aux_model = standard_model
        if dms4class_model.exists():
            chosen_model = str(dms4class_model)
        elif behavior_model is not None and behavior_model.exists():
            chosen_model = str(behavior_model)
        elif standard_model.exists():
            chosen_model = str(standard_model)
        else:
            chosen_model = None
        self.detector = YOLODetector(
            model_path=chosen_model,
            device=str(runtime.get("device", "cpu")),
            conf_threshold=float(thresholds.get("yolo_confidence", 0.5)),
            class_conf_thresholds={
                "phone": float(thresholds.get("phone_yolo_confidence", thresholds.get("yolo_confidence", 0.5))),
            },
        )
        self.aux_detector: YOLODetector | None = None
        if (
            bool(thresholds.get("aux_coco_detector_enabled", True))
            and aux_model.exists()
            and chosen_model != str(aux_model)
        ):
            self.aux_detector = YOLODetector(
                model_path=str(aux_model),
                device=str(runtime.get("device", "cpu")),
                conf_threshold=float(thresholds.get("aux_yolo_confidence", 0.12)),
                imgsz=int(thresholds.get("aux_yolo_imgsz", 960)),
                class_ids=_AUX_COCO_CLASS_IDS,
                class_conf_thresholds={
                    "phone": float(thresholds.get("aux_phone_yolo_confidence", thresholds.get("aux_yolo_confidence", 0.12))),
                },
            )
        smoke_model = Path(str(thresholds.get("smoke_model_path", "")))
        if not smoke_model.is_absolute():
            smoke_model = project_root / smoke_model
        self.smoke_detector: YOLODetector | None = None
        if bool(thresholds.get("smoke_detector_enabled", True)) and smoke_model.exists():
            self.smoke_detector = YOLODetector(
                model_path=str(smoke_model),
                device=str(runtime.get("device", "cpu")),
                conf_threshold=float(thresholds.get("smoke_yolo_confidence", 0.25)),
                imgsz=int(thresholds.get("smoke_yolo_imgsz", 800)),
            )
        self.tracker = ByteTrackerWrapper(lost_ttl_frames=int(thresholds.get("face_track_ttl_frames", 8)))
        self.object_smoother = ObjectBoxSmoother(
            ttl_frames=int(thresholds.get("object_track_ttl_frames", 5)),
            alpha=float(thresholds.get("object_track_smoothing", 0.65)),
        )
        self.face_analyzer = FaceAnalyzer(
            ear_threshold=float(thresholds["ear_threshold"]),
            mar_threshold=float(thresholds["mar_threshold"]),
        )
        self.temporal = TemporalAggregator(
            fps=int(runtime.get("fps", 30)),
            window_seconds=float(thresholds["window_seconds"]),
            yaw_threshold_deg=float(thresholds["yaw_threshold_deg"]),
            head_down_pitch_deg=float(thresholds.get("phone_head_down_pitch_deg", -18.0)),
            head_down_pitch_mode=str(thresholds.get("head_down_pitch_mode", "negative")),
        )
        self.danger = DangerDetector(
            phone_iou_threshold=float(thresholds["phone_iou_threshold"]),
            phone_duration_seconds=float(thresholds["phone_duration_seconds"]),
            drink_duration_seconds=float(thresholds.get("drink_duration_seconds", 0.6)),
            smoke_duration_seconds=float(thresholds.get("smoke_duration_seconds", 0.8)),
            seatbelt_grace_seconds=10.0,
            seatbelt_enabled=bool(thresholds.get("seatbelt_enabled", False)),
            head_down_pitch_deg=float(thresholds.get("phone_head_down_pitch_deg", -18.0)),
        )
        sequence_model_path = resolve_sequence_model_path(runtime, external_root)
        self.lstm = LSTMClassifier(
            model_path=str(sequence_model_path) if sequence_model_path is not None else "",
            seq_len=30,
            thresholds=thresholds,
        )
        self.fsm = PipelineFSM()
        self.fatigue_fsm = FatigueStateFSM(downgrade_frames=10)
        self.distraction_fsm = TimedStateFSM(
            min_state_seconds=float(thresholds.get("distraction_state_min_seconds", 3.0)),
            candidate_seconds=float(thresholds.get("distraction_candidate_seconds", 0.8)),
        )
        self._last_pose_by_track: dict[int, dict[str, float]] = {}
        self._warm_up()

    @staticmethod
    def _merge_aux_grouped(
        grouped: Dict[str, list],
        aux_grouped: Dict[str, list],
    ) -> Dict[str, list]:
        merged = {key: list(value) for key, value in grouped.items()}
        for key in _AUX_OBJECT_CLASSES:
            merged.setdefault(key, [])
            for box in aux_grouped.get(key, []):
                if not any(iou(box, old) > 0.65 for old in merged[key]):
                    merged[key].append(box)
        return merged

    @staticmethod
    def _merge_smoke_grouped(
        grouped: Dict[str, list],
        smoke_grouped: Dict[str, list],
    ) -> Dict[str, list]:
        merged = {key: list(value) for key, value in grouped.items()}
        for key in _SMOKE_OBJECT_CLASSES:
            merged.setdefault(key, [])
            for box in smoke_grouped.get(key, []):
                if not any(iou(box, old) > 0.45 for old in merged[key]):
                    merged[key].append(box)
        return merged

    @staticmethod
    def _xyxy_to_xywh(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1)))

    @staticmethod
    def _box_center_xywh(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _center_distance_xywh(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax, ay = DMSPipeline._box_center_xywh(a)
        bx, by = DMSPipeline._box_center_xywh(b)
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    @staticmethod
    def _point_in_xywh(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
        px, py = point
        x, y, w, h = box
        return x <= px <= x + w and y <= py <= y + h

    @staticmethod
    def _intersects_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh)

    @classmethod
    def _is_plausible_aux_object_box(
        cls,
        class_name: str,
        box: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> bool:
        _, _, face_w, face_h = cls._xyxy_to_xywh(face_bbox)
        _, _, box_w, box_h = box
        if face_w <= 0 or face_h <= 0 or box_w <= 0 or box_h <= 0:
            return False

        face_area = max(1, face_w * face_h)
        box_area = box_w * box_h
        aspect = box_w / max(1.0, float(box_h))

        if class_name == "phone":
            return (
                box_w >= face_w * 0.07
                and box_h >= face_h * 0.09
                and box_area >= face_area * 0.005
                and 0.25 <= aspect <= 2.90
            )

        if class_name in {"bottle", "water"}:
            return (
                box_w >= face_w * 0.08
                and box_h >= face_h * 0.14
                and box_area >= face_area * 0.010
                and 0.16 <= aspect <= 1.35
            )

        if class_name in {"cup", "drink"}:
            return (
                box_w >= face_w * 0.09
                and box_h >= face_h * 0.13
                and box_area >= face_area * 0.010
                and 0.30 <= aspect <= 1.85
            )

        return False

    @classmethod
    def _driver_interaction_regions(
        cls,
        face_bbox: tuple[int, int, int, int],
    ) -> Dict[str, tuple[int, int, int, int]]:
        fx, fy, fw, fh = cls._xyxy_to_xywh(face_bbox)
        return {
            "driver_object_zone": (
                int(fx - fw * 1.05),
                int(fy - fh * 0.25),
                max(1, int(fw * 3.10)),
                max(1, int(fh * 3.05)),
            ),
            "ear_mouth_zone": (
                int(fx - fw * 0.60),
                int(fy - fh * 0.10),
                max(1, int(fw * 2.20)),
                max(1, int(fh * 1.15)),
            ),
            "handheld_zone": (
                int(fx - fw * 0.95),
                int(fy + fh * 0.45),
                max(1, int(fw * 2.90)),
                max(1, int(fh * 2.20)),
            ),
        }

    @classmethod
    def _mouth_object_zone(
        cls,
        face_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = cls._xyxy_to_xywh(face_bbox)
        return (
            int(fx - fw * 0.45),
            int(fy + fh * 0.25),
            max(1, int(fw * 1.90)),
            max(1, int(fh * 0.95)),
        )

    @classmethod
    def _phone_box_looks_like_cigarette(
        cls,
        box: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> bool:
        _, _, face_w, face_h = cls._xyxy_to_xywh(face_bbox)
        _, _, box_w, box_h = box
        if face_w <= 0 or face_h <= 0 or box_w <= 0 or box_h <= 0:
            return False

        center = cls._box_center_xywh(box)
        if not cls._point_in_xywh(center, cls._mouth_object_zone(face_bbox)):
            return False

        face_area = max(1, face_w * face_h)
        box_area = box_w * box_h
        aspect = box_w / max(1.0, float(box_h))
        small_near_mouth = (
            box_w <= face_w * 0.55
            and box_h <= face_h * 0.28
            and box_area <= face_area * 0.055
        )
        elongated = aspect >= 3.0 or aspect <= 0.33
        return small_near_mouth and elongated

    @classmethod
    def _remap_phone_cigarette_confusions(
        cls,
        grouped: Dict[str, list],
        face_bbox: tuple[int, int, int, int],
    ) -> Dict[str, list]:
        remapped = {key: list(value) for key, value in grouped.items()}
        phones: list[tuple[int, int, int, int]] = []
        cigarettes = remapped.setdefault("cigarette", [])
        for box in remapped.get("phone", []):
            if cls._phone_box_looks_like_cigarette(box, face_bbox):
                cigarettes.append(box)
            else:
                phones.append(box)
        remapped["phone"] = phones
        return remapped

    @classmethod
    def _phone_box_looks_like_drink(
        cls,
        box: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> bool:
        _, _, face_w, face_h = cls._xyxy_to_xywh(face_bbox)
        _, _, box_w, box_h = box
        if face_w <= 0 or face_h <= 0 or box_w <= 0 or box_h <= 0:
            return False

        mouth_zone = cls._mouth_object_zone(face_bbox)
        center = cls._box_center_xywh(box)
        if not (cls._point_in_xywh(center, mouth_zone) or cls._intersects_xywh(box, mouth_zone)):
            return False

        face_area = max(1, face_w * face_h)
        box_area = box_w * box_h
        aspect = box_w / max(1.0, float(box_h))
        return (
            box_area >= face_area * 0.10
            and box_w >= face_w * 0.28
            and box_h >= face_h * 0.22
            and 0.45 <= aspect <= 2.60
        )

    @classmethod
    def _remap_phone_drink_confusions(
        cls,
        grouped: Dict[str, list],
        face_bbox: tuple[int, int, int, int],
    ) -> Dict[str, list]:
        remapped = {key: list(value) for key, value in grouped.items()}
        phones: list[tuple[int, int, int, int]] = []
        cups = remapped.setdefault("cup", [])
        for box in remapped.get("phone", []):
            if cls._phone_box_looks_like_drink(box, face_bbox):
                cups.append(box)
            else:
                phones.append(box)
        remapped["phone"] = phones
        return remapped

    @classmethod
    def _is_plausible_smoke_box(
        cls,
        class_name: str,
        box: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> bool:
        _, _, face_w, face_h = cls._xyxy_to_xywh(face_bbox)
        _, _, box_w, box_h = box
        if face_w <= 0 or face_h <= 0 or box_w <= 0 or box_h <= 0:
            return False

        center = cls._box_center_xywh(box)
        mouth_zone = cls._mouth_object_zone(face_bbox)
        if not (cls._point_in_xywh(center, mouth_zone) or cls._intersects_xywh(box, mouth_zone)):
            return False

        face_area = max(1, face_w * face_h)
        box_area = box_w * box_h
        aspect = box_w / max(1.0, float(box_h))
        elongation = max(aspect, 1.0 / max(aspect, 1e-6))
        if class_name == "cigarette":
            return (
                box_w >= face_w * 0.035
                and box_h >= face_h * 0.025
                and box_area >= face_area * 0.0015
                and box_area <= face_area * 0.060
                and min(box_w, box_h) <= max(face_w, face_h) * 0.20
                and max(box_w, box_h) <= max(face_w, face_h) * 0.70
                and elongation >= 1.8
            )
        if class_name == "smoke":
            return (
                box_area >= face_area * 0.004
                and box_area <= face_area * 0.35
                and 0.35 <= aspect <= 3.5
            )
        return False

    def _smoke_grouped_for_driver(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> Dict[str, list]:
        if self.smoke_detector is None:
            return {}
        crop_info = self._driver_crop(frame, face_bbox)
        if crop_info is None:
            return {}
        crop, (offset_x, offset_y) = crop_info
        smoke_grouped = self.smoke_detector.group_detections(self.smoke_detector.detect(crop))
        remapped = {key: [] for key in _SMOKE_OBJECT_CLASSES}
        for key in _SMOKE_OBJECT_CLASSES:
            for x, y, w, h in smoke_grouped.get(key, []):
                box = (int(x + offset_x), int(y + offset_y), int(w), int(h))
                if self._is_plausible_smoke_box(key, box, face_bbox):
                    remapped[key].append(box)
        return remapped

    @classmethod
    def _is_driver_object_box(
        cls,
        class_name: str,
        box: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int],
    ) -> bool:
        if not cls._is_plausible_aux_object_box(class_name, box, face_bbox):
            return False

        regions = cls._driver_interaction_regions(face_bbox)
        center = cls._box_center_xywh(box)
        if not cls._point_in_xywh(center, regions["driver_object_zone"]):
            return False

        if class_name == "phone":
            return (
                cls._point_in_xywh(center, regions["ear_mouth_zone"])
                or cls._point_in_xywh(center, regions["handheld_zone"])
            )

        if class_name in {"bottle", "cup", "drink", "water"}:
            mouth_zone = cls._mouth_object_zone(face_bbox)
            return (
                cls._point_in_xywh(center, mouth_zone)
                or cls._intersects_xywh(box, mouth_zone)
            )

        return False

    @classmethod
    def _looks_like_same_small_object(
        cls,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> bool:
        if iou(a, b) > 0.02:
            return True
        max_size = max(a[2], a[3], b[2], b[3], 1)
        return cls._center_distance_xywh(a, b) <= max_size * 0.80

    @classmethod
    def _suppress_aux_conflicts(
        cls,
        grouped: Dict[str, list],
        aux_grouped: Dict[str, list],
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> Dict[str, list]:
        cigarette_boxes = list(grouped.get("cigarette", [])) + list(grouped.get("smoke", []))
        filtered = {key: list(value) for key, value in aux_grouped.items()}
        if cigarette_boxes and aux_grouped.get("phone"):
            filtered["phone"] = [
                phone_box
                for phone_box in aux_grouped.get("phone", [])
                if not any(cls._looks_like_same_small_object(phone_box, cig_box) for cig_box in cigarette_boxes)
            ]

        mouth_drink_boxes: list[tuple[int, int, int, int]] = []
        if face_bbox is not None:
            mouth_zone = cls._mouth_object_zone(face_bbox)
            for key in ("bottle", "cup", "drink", "water"):
                mouth_drink_boxes.extend(
                    drink_box
                    for drink_box in filtered.get(key, [])
                    if cls._point_in_xywh(cls._box_center_xywh(drink_box), mouth_zone)
                    or cls._intersects_xywh(drink_box, mouth_zone)
                )
            if mouth_drink_boxes and filtered.get("phone"):
                filtered["phone"] = [
                    phone_box
                    for phone_box in filtered.get("phone", [])
                    if not any(cls._looks_like_same_small_object(phone_box, drink_box) for drink_box in mouth_drink_boxes)
                ]

        phone_boxes = list(grouped.get("phone", [])) + list(filtered.get("phone", []))
        if phone_boxes:
            for key in ("bottle", "cup", "drink", "water"):
                filtered[key] = [
                    drink_box
                    for drink_box in filtered.get(key, [])
                    if drink_box in mouth_drink_boxes
                    or not any(cls._looks_like_same_small_object(drink_box, phone_box) for phone_box in phone_boxes)
                ]
        return filtered

    @staticmethod
    def _driver_crop(frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]] | None:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        crop_x1 = max(0, int(x1 - fw * 1.05))
        crop_y1 = max(0, int(y1 - fh * 0.25))
        crop_x2 = min(frame_w, int(x2 + fw * 1.05))
        crop_y2 = min(frame_h, int(y2 + fh * 2.25))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        return frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)

    def _aux_grouped_for_driver(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> Dict[str, list]:
        if self.aux_detector is None:
            return {}
        crop_info = self._driver_crop(frame, face_bbox)
        if crop_info is None:
            return {}
        crop, (offset_x, offset_y) = crop_info
        aux_grouped = self.aux_detector.group_detections(self.aux_detector.detect(crop))
        remapped = {key: [] for key in _AUX_OBJECT_CLASSES}
        for key in _AUX_OBJECT_CLASSES:
            for x, y, w, h in aux_grouped.get(key, []):
                box = (int(x + offset_x), int(y + offset_y), int(w), int(h))
                if self._is_driver_object_box(key, box, face_bbox):
                    remapped[key].append(box)
        return remapped

    def _warm_up(self) -> None:
        """Run one dummy inference to compile MediaPipe GPU kernels."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.face_analyzer.analyze(dummy)
        except Exception:
            pass

    def _stabilize_pose_features(
        self,
        feats: Dict[str, Any],
        track_id: int,
        ts: float,
    ) -> Dict[str, Any]:
        """Keep pose continuity through brief landmark dropouts.

        Turning or looking down can temporarily make the eye landmarks unreliable.
        If the previous pose is recent, keep pitch/yaw/roll so posture behaviors
        are not mislabeled as only eye occlusion.
        """
        hold_seconds = float(self.thresholds.get("eye_occluded_pose_hold_seconds", 1.2))
        landmarks_ok = feats.get("landmarks_468") is not None and not bool(feats.get("eye_occluded", False))
        pose_valid = bool(feats.get("pose_valid", landmarks_ok))
        if pose_valid:
            self._last_pose_by_track[track_id] = {
                "ts": ts,
                "pitch": float(feats.get("pitch", 0.0)),
                "yaw": float(feats.get("yaw", 0.0)),
                "roll": float(feats.get("roll", 0.0)),
            }
            feats["pose_held"] = False
            feats["pose_valid"] = True
            return feats

        last = self._last_pose_by_track.get(track_id)
        if last is None or ts - float(last.get("ts", ts)) > hold_seconds:
            feats["pose_held"] = False
            feats["pose_valid"] = False
            return feats

        stabilized = dict(feats)
        stabilized["pitch"] = float(last.get("pitch", 0.0))
        stabilized["yaw"] = float(last.get("yaw", 0.0))
        stabilized["roll"] = float(last.get("roll", 0.0))
        stabilized["pose_held"] = True
        stabilized["pose_valid"] = True
        return stabilized

    def process(self, frame: np.ndarray, ts: float) -> Dict[str, Any]:
        detections = self.detector.detect(frame)
        grouped = self.detector.group_detections(detections)
        track = self.tracker.update(grouped["face"], frame.shape)

        if track is None:
            return {
                "fatigue": "NORMAL",
                "distraction": "NORMAL",
                "danger": "NORMAL",
                "alerts": [],
                "track_id": -1,
                "debug": {
                    "ear_left": None, "ear_right": None, "mar": None,
                    "pitch": None, "yaw": None, "roll": None,
                    "perclos": None, "nod_freq": None, "yawn_count": None,
                    "gaze_away_duration": None, "continuous_closed": None,
                    "look_left_duration": None, "look_right_duration": None,
                    "head_down_negative_duration": None, "head_down_positive_duration": None,
                    "yaw_suppressed_by_yawn": False,
                    "lstm_score": None, "lstm_pred": None,
                    "eye_occluded": False,
                    "pose_valid": False,
                    "face_bbox": None, "landmarks_global": None, "all_bboxes": grouped,
                    "aux_detector": bool(self.aux_detector is not None),
                    "smoke_detector": bool(self.smoke_detector is not None),
                },
            }

        if self.fsm.on_track(track.track_id):
            self.temporal.reset_track(track.track_id)
            self.fatigue_fsm.reset()
            self.distraction_fsm.reset()
            self.object_smoother.reset()
            self._last_pose_by_track.clear()

        grouped = self._remap_phone_cigarette_confusions(grouped, track.bbox)
        grouped = self._merge_smoke_grouped(grouped, self._smoke_grouped_for_driver(frame, track.bbox))
        if bool(self.thresholds.get("aux_crop_detector_enabled", True)):
            aux_grouped = self._aux_grouped_for_driver(frame, track.bbox)
            aux_grouped = self._suppress_aux_conflicts(grouped, aux_grouped, track.bbox)
            grouped = self._merge_aux_grouped(grouped, aux_grouped)
        grouped = self._remap_phone_drink_confusions(grouped, track.bbox)
        grouped = self.object_smoother.update(grouped)

        x1, y1, x2, y2 = track.bbox
        x, y = max(0, x1), max(0, y1)
        w, h = max(0, x2 - x1), max(0, y2 - y1)

        frame_h, frame_w = frame.shape[:2]
        y_end, x_end = min(frame_h, y + h), min(frame_w, x + w)
        face_roi = frame[y:y_end, x:x_end]
        behavior_feats = self.detector.get_behavior_features(detections)
        feats = behavior_feats if behavior_feats is not None else self.face_analyzer.analyze(face_roi)
        feats = self._stabilize_pose_features(feats, track.track_id, ts)
        metrics = self.temporal.update(feats, track.track_id, ts)
        danger_alerts = self.danger.detect(
            grouped,
            track.bbox,
            ts,
            features=feats,
            metrics=metrics,
            frame_shape=frame.shape,
        )
        raw_behavior = (self.danger.last_context.get("raw") or {}) if isinstance(self.danger.last_context, dict) else {}
        pending_object_behavior = any(
            bool(raw_behavior.get(key))
            for key in ("phone", "phone_call", "phone_use", "drink", "smoke")
        )
        posture_suppressed = (
            pending_object_behavior
            or any(alert in _OBJECT_BEHAVIOR_DISTRACTIONS for alert in danger_alerts)
            or bool(feats.get("is_yawning", False))
        )
        states = state_from_metrics(metrics, self.thresholds, danger_alerts)
        raw_distraction = states["distraction"]
        if raw_distraction in _POSE_DISTRACTIONS and posture_suppressed:
            raw_distraction = "NORMAL"

        posture_alerts: list[str] = []
        if (
            not posture_suppressed
            and float(metrics.get("head_down_duration", 0.0)) >= float(self.thresholds.get("head_down_seconds", 1.5))
        ):
            posture_alerts.append("HEAD_DOWN")
        look_duration = max(
            float(metrics.get("look_left_duration", 0.0)),
            float(metrics.get("look_right_duration", 0.0)),
            float(metrics.get("gaze_away_duration", 0.0)),
        )
        if (
            not posture_suppressed
            and look_duration >= float(self.thresholds.get("look_around_seconds", self.thresholds.get("gaze_away_seconds", 2.0)))
        ):
            posture_alerts.append("LOOK_AROUND")

        if raw_distraction in _OBJECT_BEHAVIOR_DISTRACTIONS:
            self.distraction_fsm.reset()
            states["distraction"] = raw_distraction
        else:
            states["distraction"] = self.distraction_fsm.update(raw_distraction, ts)
        has_behavior_distraction = (
            any(alert in _BEHAVIOR_DISTRACTIONS for alert in danger_alerts)
            or states["distraction"] in _BEHAVIOR_DISTRACTIONS
        )
        strong_eye_fatigue = (
            float(metrics.get("continuous_closed", 0.0)) >= 3.0
            or float(metrics.get("perclos", 0.0)) >= float(self.thresholds.get("perclos_alert", 0.30))
        )
        if has_behavior_distraction and not strong_eye_fatigue:
            states["fatigue"] = "NORMAL"
            self.fatigue_fsm.reset()

        # LSTM augments rule engine — can upgrade but never downgrade hard rule decisions
        raw_seq = metrics.pop("raw_sequence", [])
        lstm_score, lstm_pred = self.lstm.predict(raw_seq)

        rule_fatigue = states["fatigue"]
        if has_behavior_distraction:
            # Phone calls, phone use, drinking, and smoking are distraction
            # events. Do not let pose-heavy LSTM output relabel them as fatigue.
            pass
        elif (
            bool(self.thresholds.get("lstm_can_warn", False))
            and rule_fatigue == "NORMAL"
            and lstm_score > float(self.thresholds.get("lstm_warning_threshold", 0.85))
        ):
            states["fatigue"] = "WARNING"
        elif (
            bool(self.thresholds.get("lstm_can_alert", False))
            and rule_fatigue == "WARNING"
            and lstm_score > float(self.thresholds.get("lstm_alert_threshold", 0.95))
        ):
            states["fatigue"] = "ALERT"
        # ALERT from hard rules (continuous_closed, PERCLOS) is never downgraded by LSTM

        # Apply debounce: upgrade immediately, downgrade only after N frames
        states["fatigue"] = self.fatigue_fsm.update(states["fatigue"])

        alerts = [a for a in [states["distraction"], states["danger"], states["fatigue"]] if a != "NORMAL"]
        if feats.get("eye_occluded") and not has_behavior_distraction and not posture_alerts:
            alerts.append("EYE_OCCLUDED")
        
        # Remap landmarks from ROI space to full-frame space for visualization
        landmarks_468 = feats.get("landmarks_468")
        landmarks_global = None
        if landmarks_468 is not None:
            landmarks_global = landmarks_468.copy()
            landmarks_global[:, 0] += x
            landmarks_global[:, 1] += y

        debug = {
            "ear_left": float(feats["ear_left"]),
            "ear_right": float(feats["ear_right"]),
            "mar": float(feats["mar"]),
            "pitch": float(metrics.get("pitch_corrected", feats["pitch"])),
            "raw_pitch": float(metrics.get("raw_pitch", feats["pitch"])),
            "pitch_baseline": float(metrics.get("pitch_baseline", 0.0)),
            "yaw": float(metrics.get("yaw_corrected", feats["yaw"])),
            "raw_yaw": float(feats["yaw"]),
            "yaw_baseline": float(metrics.get("yaw_baseline", 0.0)),
            "roll": float(feats["roll"]),
            "perclos": float(metrics["perclos"]),
            "nod_freq": float(metrics["nod_freq"]),
            "yawn_count": int(metrics["yawn_count"]),
            "gaze_away_duration": float(metrics["gaze_away_duration"]),
            "look_left_duration": float(metrics.get("look_left_duration", 0.0)),
            "look_right_duration": float(metrics.get("look_right_duration", 0.0)),
            "head_down_duration": float(metrics.get("head_down_duration", 0.0)),
            "head_down_negative_duration": float(metrics.get("head_down_negative_duration", 0.0)),
            "head_down_positive_duration": float(metrics.get("head_down_positive_duration", 0.0)),
            "head_up_duration": float(metrics.get("head_up_duration", 0.0)),
            "continuous_closed": float(metrics.get("continuous_closed", 0.0)),
            "yaw_suppressed_by_yawn": bool(metrics.get("yaw_suppressed_by_yawn", False)),
            "yaw_calibrated": bool(metrics.get("yaw_calibrated", False)),
            "pitch_calibrated": bool(metrics.get("pitch_calibrated", False)),
            "lstm_score": float(lstm_score),
            "lstm_pred": int(lstm_pred),
            "lstm_loaded": bool(self.lstm.model_loaded),
            "lstm_model_kind": str(getattr(self.lstm, "model_kind", "disabled")),
            "lstm_model_path": str(getattr(self.lstm, "model_path", "")),
            "eye_occluded": bool(feats.get("eye_occluded", False)),
            "pose_held": bool(feats.get("pose_held", False)),
            "pose_valid": bool(metrics.get("pose_valid", feats.get("pose_valid", False))),
            "face_bbox": [int(x), int(y), int(w), int(h)],
            "landmarks_global": landmarks_global,
            "all_bboxes": grouped,
            "behavior_alerts": danger_alerts,
            "posture_alerts": posture_alerts,
            "behavior_context": self.danger.last_context,
            "raw_distraction": raw_distraction,
            "stable_distraction": states["distraction"],
            "pending_object_behavior": bool(pending_object_behavior),
            "posture_suppressed": bool(posture_suppressed),
            "behavior_thresholds": {
                "head_down_pitch_deg": float(self.thresholds.get("phone_head_down_pitch_deg", -18.0)),
                "head_down_pitch_mode": str(self.thresholds.get("head_down_pitch_mode", "negative")),
                "head_down_seconds": float(self.thresholds.get("head_down_seconds", 1.5)),
                "yaw_threshold_deg": float(self.thresholds.get("yaw_threshold_deg", 30.0)),
                "look_around_seconds": float(self.thresholds.get("look_around_seconds", self.thresholds.get("gaze_away_seconds", 2.0))),
                "phone_duration_seconds": float(self.thresholds.get("phone_duration_seconds", 2.0)),
                "drink_duration_seconds": float(self.thresholds.get("drink_duration_seconds", 1.2)),
                "smoke_duration_seconds": float(self.thresholds.get("smoke_duration_seconds", 1.5)),
                "distraction_candidate_seconds": float(self.thresholds.get("distraction_candidate_seconds", 0.8)),
                "distraction_state_min_seconds": float(self.thresholds.get("distraction_state_min_seconds", 3.0)),
                "yolo_confidence": float(self.thresholds.get("yolo_confidence", 0.13)),
                "phone_yolo_confidence": float(self.thresholds.get("phone_yolo_confidence", self.thresholds.get("yolo_confidence", 0.13))),
                "aux_yolo_confidence": float(self.thresholds.get("aux_yolo_confidence", 0.13)),
                "aux_phone_yolo_confidence": float(self.thresholds.get("aux_phone_yolo_confidence", self.thresholds.get("aux_yolo_confidence", 0.13))),
            },
            "behavior_mode": bool(self.detector.behavior_mode),
            "aux_detector": bool(self.aux_detector is not None),
            "smoke_detector": bool(self.smoke_detector is not None),
            "driver_object_regions": self._driver_interaction_regions(track.bbox),
        }

        debug["raw_feats"] = feats

        return {
            "fatigue": states["fatigue"],
            "distraction": states["distraction"],
            "danger": states["danger"],
            "alerts": alerts,
            "track_id": int(track.track_id),
            "debug": debug,
        }


def synthetic_frames(width: int, height: int) -> Iterable[np.ndarray]:
    t = 0
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        intensity = int((np.sin(t / 10.0) * 0.5 + 0.5) * 255)
        frame[:, :, :] = intensity
        t += 1
        yield frame


def run_pipeline(source: str, max_frames: int, show: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")

    logging.basicConfig(level=getattr(logging, str(runtime.get("log_level", "INFO"))))
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    width = int(runtime.get("width", 640))
    height = int(runtime.get("height", 480))
    start = time.time()
    processed = 0

    if source == "synthetic":
        frames_iter = synthetic_frames(width, height)
        capture = None
    else:
        capture = cv2.VideoCapture(0 if source == "0" else source)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open source: {source}")
        frames_iter = None

    try:
        while True:
            if frames_iter is not None:
                frame = next(frames_iter)
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            ts = time.time()
            state = pipeline.process(frame, ts)
            processed += 1

            if show:
                cv2.putText(frame, str(state["alerts"]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("dms-pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if processed % 30 == 0:
                elapsed = max(1e-6, time.time() - start)
                fps = processed / elapsed
                logging.info("processed=%s fps=%.2f last=%s", processed, fps, state)

            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if show:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DMS pipeline")
    parser.add_argument("--source", default="synthetic", help="0 | video path | synthetic")
    parser.add_argument("--max-frames", type=int, default=90, help="stop after N frames; <=0 for no limit")
    parser.add_argument("--show", action="store_true", help="show preview window")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(source=args.source, max_frames=args.max_frames, show=args.show)


if __name__ == "__main__":
    main()
