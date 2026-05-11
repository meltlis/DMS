from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]


class ByteTrackerWrapper:
    """Minimal single-target tracker stub compatible with pipeline contract."""

    def __init__(self) -> None:
        self._track_id = 1
        self._last_bbox: Tuple[int, int, int, int] | None = None

    def update(self, face_boxes: List[Tuple[int, int, int, int]]) -> Track | None:
        if face_boxes:
            self._last_bbox = face_boxes[0]
            return Track(track_id=self._track_id, bbox=self._last_bbox)
        if self._last_bbox is None:
            return None
        return Track(track_id=self._track_id, bbox=self._last_bbox)
