from __future__ import annotations

from src.perception.tracker import ByteTrackerWrapper


class TestByteTrackerWrapper:
    def test_update_returns_track_when_face_present(self) -> None:
        tracker = ByteTrackerWrapper()
        track = tracker.update([(10, 10, 50, 50)])
        assert track is not None
        assert track.track_id == 1
        assert track.bbox == (10, 10, 50, 50)

    def test_update_remembers_last_bbox(self) -> None:
        tracker = ByteTrackerWrapper()
        tracker.update([(10, 10, 50, 50)])
        track = tracker.update([])
        assert track is not None
        assert track.bbox == (10, 10, 50, 50)

    def test_update_returns_none_when_no_history(self) -> None:
        tracker = ByteTrackerWrapper()
        assert tracker.update([]) is None

    def test_update_uses_first_box(self) -> None:
        tracker = ByteTrackerWrapper()
        track = tracker.update([(0, 0, 10, 10), (20, 20, 30, 30)])
        assert track.bbox == (0, 0, 10, 10)
