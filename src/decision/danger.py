from __future__ import annotations

from math import hypot
from typing import Dict, List, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xyxy_to_xywh(box: BBox | None) -> BBox | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1)))


def _center(box: BBox) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _bbox_from_points(points: np.ndarray, pad_ratio: float = 0.35) -> BBox | None:
    if points.size == 0:
        return None
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = max(1.0, float(max_xy[0] - min_xy[0]))
    height = max(1.0, float(max_xy[1] - min_xy[1]))
    pad_x = max(3.0, width * pad_ratio)
    pad_y = max(3.0, height * pad_ratio)
    return (
        int(round(float(min_xy[0] - pad_x))),
        int(round(float(min_xy[1] - pad_y))),
        int(round(width + 2 * pad_x)),
        int(round(height + 2 * pad_y)),
    )


def _near_region(obj: BBox, region: BBox, face_diag: float, max_dist_ratio: float = 0.38) -> bool:
    if iou(obj, region) > 0.0:
        return True
    ox, oy = _center(obj)
    rx, ry = _center(region)
    return hypot(ox - rx, oy - ry) <= max(face_diag * max_dist_ratio, 12.0)


class DangerDetector:
    def __init__(
        self,
        phone_iou_threshold: float,
        phone_duration_seconds: float = 2.0,
        drink_duration_seconds: float = 0.6,
        smoke_duration_seconds: float = 0.0,
        seatbelt_grace_seconds: float = 10.0,
        seatbelt_enabled: bool = True,
        head_down_pitch_deg: float = -18.0,
    ) -> None:
        self.phone_iou_threshold = phone_iou_threshold
        self.phone_duration_seconds = phone_duration_seconds
        self.drink_duration_seconds = drink_duration_seconds
        self.smoke_duration_seconds = smoke_duration_seconds
        self.seatbelt_grace_seconds = seatbelt_grace_seconds
        self.seatbelt_enabled = seatbelt_enabled
        self.head_down_pitch_deg = head_down_pitch_deg
        self._phone_first_ts: float | None = None
        self._start_ts: float | None = None
        self._seatbelt_seen: bool = False
        self._first_seen: Dict[str, float | None] = {
            "PHONE": None,
            "PHONE_CALL": None,
            "PHONE_USE": None,
            "SUSPECTED_PHONE_USE": None,
            "DRINK": None,
            "SMOKE": None,
        }
        self.last_context: Dict[str, object] = {}

    def reset(self) -> None:
        self._phone_first_ts = None
        self._start_ts = None
        self._seatbelt_seen = False
        for key in self._first_seen:
            self._first_seen[key] = None
        self.last_context = {}

    def _confirmed(self, key: str, active: bool, ts: float, duration: float) -> bool:
        if active:
            if self._first_seen.get(key) is None:
                self._first_seen[key] = ts
            first_ts = self._first_seen.get(key)
            return first_ts is not None and ts - first_ts >= duration
        self._first_seen[key] = None
        return False

    def _regions(
        self,
        face_bbox_xyxy: BBox | None,
        features: Dict[str, object] | None,
    ) -> Dict[str, BBox]:
        face = _xyxy_to_xywh(face_bbox_xyxy)
        if face is None:
            return {}

        fx, fy, fw, fh = face
        regions: Dict[str, BBox] = {
            "face": face,
            "mouth": (
                int(fx + fw * 0.28),
                int(fy + fh * 0.58),
                max(1, int(fw * 0.44)),
                max(1, int(fh * 0.24)),
            ),
            "left_ear": (
                int(fx - fw * 0.20),
                int(fy + fh * 0.20),
                max(1, int(fw * 0.42)),
                max(1, int(fh * 0.55)),
            ),
            "right_ear": (
                int(fx + fw * 0.78),
                int(fy + fh * 0.20),
                max(1, int(fw * 0.42)),
                max(1, int(fh * 0.55)),
            ),
            "below_face": (
                int(fx - fw * 0.25),
                int(fy + fh * 0.62),
                max(1, int(fw * 1.50)),
                max(1, int(fh * 1.40)),
            ),
        }

        landmarks = None if features is None else features.get("landmarks_468")
        if landmarks is not None:
            try:
                pts = np.asarray(landmarks, dtype=np.float32).copy()
                pts[:, 0] += face_bbox_xyxy[0]
                pts[:, 1] += face_bbox_xyxy[1]
                mouth_pts = pts[[61, 291, 13, 14, 78, 308]]
                mouth = _bbox_from_points(mouth_pts, pad_ratio=0.70)
                if mouth is not None:
                    regions["mouth"] = mouth
            except Exception:
                pass

        return regions

    def _phone_context(
        self,
        phone_boxes: List[BBox],
        regions: Dict[str, BBox],
        features: Dict[str, object] | None,
        metrics: Dict[str, object] | None,
    ) -> tuple[bool, bool, bool]:
        if not phone_boxes or "face" not in regions:
            return False, False, False

        face = regions["face"]
        face_diag = hypot(face[2], face[3])
        pitch = float((features or {}).get("pitch", 0.0))
        head_down_duration = float((metrics or {}).get("head_down_duration", 0.0))

        call_active = False
        use_active = False
        overlap_active = False

        for phone in phone_boxes:
            if iou(phone, face) > self.phone_iou_threshold:
                overlap_active = True

            near_ear = any(
                _near_region(phone, regions[key], face_diag, max_dist_ratio=0.34)
                for key in ("left_ear", "right_ear")
                if key in regions
            )
            if near_ear:
                call_active = True
                continue

            px, py = _center(phone)
            fx, fy, fw, fh = face
            below_or_in_hands = (
                py >= fy + fh * 0.58
                or iou(phone, regions.get("below_face", face)) > 0.0
            )
            head_down = pitch <= self.head_down_pitch_deg or head_down_duration >= 0.5
            if below_or_in_hands or head_down:
                use_active = True

        return overlap_active or call_active or use_active, call_active, use_active

    def _drink_active(self, grouped: Dict[str, List[BBox]], regions: Dict[str, BBox]) -> bool:
        if "mouth" not in regions or "face" not in regions:
            return False
        face_diag = hypot(regions["face"][2], regions["face"][3])
        drink_boxes = (
            grouped.get("bottle", [])
            + grouped.get("cup", [])
            + grouped.get("drink", [])
            + grouped.get("water", [])
        )
        return any(_near_region(box, regions["mouth"], face_diag, max_dist_ratio=0.42) for box in drink_boxes)

    def _smoke_active(self, grouped: Dict[str, List[BBox]], regions: Dict[str, BBox]) -> bool:
        smoke_boxes = grouped.get("cigarette", []) + grouped.get("smoke", [])
        if not smoke_boxes:
            return False
        if "mouth" not in regions or "face" not in regions:
            return True
        face_diag = hypot(regions["face"][2], regions["face"][3])
        return any(_near_region(box, regions["mouth"], face_diag, max_dist_ratio=0.36) for box in smoke_boxes)

    def detect(
        self,
        grouped: Dict[str, List[BBox]],
        face_bbox: BBox | None,
        ts: float,
        features: Dict[str, object] | None = None,
        metrics: Dict[str, object] | None = None,
        frame_shape: Tuple[int, ...] | None = None,
    ) -> List[str]:
        alerts: List[str] = []

        # Initialize start timestamp on first call
        if self._start_ts is None:
            self._start_ts = ts

        regions = self._regions(face_bbox, features)
        phone_active, phone_call_active, phone_use_active = self._phone_context(
            grouped.get("phone", []),
            regions,
            features,
            metrics,
        )
        drink_active = self._drink_active(grouped, regions)
        smoke_active = self._smoke_active(grouped, regions)
        head_down_duration = float((metrics or {}).get("head_down_duration", 0.0))
        pitch = float((features or {}).get("pitch", 0.0))
        suspected_phone_use_active = (
            not grouped.get("phone")
            and "face" in regions
            and (head_down_duration >= 0.8 or pitch <= self.head_down_pitch_deg)
        )

        if self._confirmed("PHONE_CALL", phone_call_active, ts, self.phone_duration_seconds):
            alerts.append("PHONE_CALL")
        if self._confirmed("PHONE_USE", phone_use_active, ts, self.phone_duration_seconds):
            alerts.append("PHONE_USE")
        if self._confirmed(
            "SUSPECTED_PHONE_USE",
            suspected_phone_use_active,
            ts,
            max(1.0, self.phone_duration_seconds * 0.75),
        ):
            alerts.append("SUSPECTED_PHONE_USE")
        if self._confirmed("PHONE", phone_active, ts, self.phone_duration_seconds):
            # Compatibility alert for older UI/tests; specific labels above carry
            # the actual call-vs-use distinction.
            alerts.append("PHONE")
        self._phone_first_ts = self._first_seen.get("PHONE")

        if self._confirmed("DRINK", drink_active, ts, self.drink_duration_seconds):
            alerts.append("DRINK")
        if self._confirmed("SMOKE", smoke_active, ts, self.smoke_duration_seconds):
            alerts.append("SMOKE")

        if self.seatbelt_enabled:
            if grouped.get("seatbelt"):
                self._seatbelt_seen = True
            elif not self._seatbelt_seen and ts - self._start_ts >= self.seatbelt_grace_seconds:
                alerts.append("NO_SEATBELT")

        timers = {
            key: (max(0.0, ts - first_ts) if first_ts is not None else 0.0)
            for key, first_ts in self._first_seen.items()
        }

        self.last_context = {
            "regions": regions,
            "raw": {
                "phone": phone_active,
                "phone_call": phone_call_active,
                "phone_use": phone_use_active,
                "suspected_phone_use": suspected_phone_use_active,
                "drink": drink_active,
                "smoke": smoke_active,
            },
            "timers": timers,
            "frame_shape": frame_shape,
        }

        return alerts
