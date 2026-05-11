from __future__ import annotations

from typing import Dict, List, Tuple


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


class DangerDetector:
    def __init__(
        self,
        phone_iou_threshold: float,
        phone_duration_seconds: float = 2.0,
        seatbelt_grace_seconds: float = 10.0,
    ) -> None:
        self.phone_iou_threshold = phone_iou_threshold
        self.phone_duration_seconds = phone_duration_seconds
        self.seatbelt_grace_seconds = seatbelt_grace_seconds
        self._phone_first_ts: float | None = None
        self._start_ts: float | None = None
        self._seatbelt_seen: bool = False

    def detect(
        self,
        grouped: Dict[str, List[Tuple[int, int, int, int]]],
        face_bbox: Tuple[int, int, int, int] | None,
        ts: float,
    ) -> List[str]:
        alerts: List[str] = []

        # Initialize start timestamp on first call
        if self._start_ts is None:
            self._start_ts = ts

        # PHONE: must be present and IoU > threshold continuously for N seconds
        phone_active = False
        if face_bbox is not None:
            for pb in grouped.get("phone", []):
                if iou(pb, face_bbox) > self.phone_iou_threshold:
                    phone_active = True
                    break
        if phone_active:
            if self._phone_first_ts is None:
                self._phone_first_ts = ts
            if ts - self._phone_first_ts >= self.phone_duration_seconds:
                alerts.append("PHONE")
        else:
            self._phone_first_ts = None

        # SMOKE: detected immediately
        if grouped.get("cigarette"):
            alerts.append("SMOKE")

        # SEATBELT: alert if seatbelt was never detected within the startup grace period.
        # Once seen at any point, _seatbelt_seen locks to True and no further alerts fire.
        if grouped.get("seatbelt"):
            self._seatbelt_seen = True
        if not self._seatbelt_seen and (ts - self._start_ts) >= self.seatbelt_grace_seconds:
            alerts.append("NO_SEATBELT")

        return alerts
