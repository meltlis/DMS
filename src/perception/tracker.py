from dataclasses import dataclass
from math import sqrt
from typing import List, Sequence, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]


class ByteTrackerWrapper:
    """Primary-driver face tracker with timeout and ID changes.

    This is not full ByteTrack. The project observes one driver, so this class
    selects and stabilizes the primary face, keeps it through short detector
    dropouts, and assigns a new ID after a long loss or a clear face switch.
    """

    def __init__(self, lost_ttl_frames: int = 8, reid_iou_threshold: float = 0.08) -> None:
        self.lost_ttl_frames = max(0, int(lost_ttl_frames))
        self.reid_iou_threshold = max(0.0, float(reid_iou_threshold))
        self._track_id = 1
        self._last_bbox: Tuple[int, int, int, int] | None = None
        self._missed_frames = 0

    def reset(self) -> None:
        self._track_id = 1
        self._last_bbox = None
        self._missed_frames = 0

    @staticmethod
    def _area(box: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _center(box: Tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = ByteTrackerWrapper._area(a) + ByteTrackerWrapper._area(b) - inter
        return float(inter / union) if union > 0 else 0.0

    def _driver_anchor(self, frame_shape: Sequence[int] | None) -> tuple[float, float]:
        if not frame_shape or len(frame_shape) < 2:
            return (320.0, 264.0)
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        return (frame_w * 0.5, frame_h * 0.55)

    def _score_box(self, box: Tuple[int, int, int, int], frame_shape: Sequence[int] | None) -> float:
        area_score = sqrt(max(self._area(box), 1.0))
        cx, cy = self._center(box)
        ax, ay = self._driver_anchor(frame_shape)
        dist = sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
        frame_diag = sqrt(640.0 ** 2 + 480.0 ** 2)
        if frame_shape and len(frame_shape) >= 2:
            frame_diag = sqrt(float(frame_shape[0]) ** 2 + float(frame_shape[1]) ** 2)
        center_score = 1.0 - min(dist / max(frame_diag, 1.0), 1.0)
        persist_score = self._iou(box, self._last_bbox) if self._last_bbox is not None else 0.0

        if self._last_bbox is not None:
            return persist_score * 1000.0 + area_score * 0.8 + center_score * 100.0
        return area_score + center_score * 100.0

    def _select_box(
        self,
        face_boxes: List[Tuple[int, int, int, int]],
        frame_shape: Sequence[int] | None,
    ) -> Tuple[int, int, int, int]:
        return max(face_boxes, key=lambda box: self._score_box(box, frame_shape))

    def update(
        self,
        face_boxes: List[Tuple[int, int, int, int]],
        frame_shape: Sequence[int] | None = None,
    ) -> Track | None:
        if face_boxes:
            selected = self._select_box(face_boxes, frame_shape)
            if self._last_bbox is not None and self._iou(selected, self._last_bbox) < self.reid_iou_threshold:
                self._track_id += 1
            self._last_bbox = selected
            self._missed_frames = 0
            return Track(track_id=self._track_id, bbox=selected)

        if self._last_bbox is None:
            return None

        self._missed_frames += 1
        if self._missed_frames > self.lost_ttl_frames:
            self._last_bbox = None
            self._missed_frames = 0
            self._track_id += 1
            return None

        return Track(track_id=self._track_id, bbox=self._last_bbox)
