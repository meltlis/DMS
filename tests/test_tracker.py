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
        assert track.bbox == (20, 20, 30, 30)

    def test_update_prefers_more_central_larger_face(self) -> None:
        tracker = ByteTrackerWrapper()
        # Second face is larger and closer to the expected driver area.
        boxes = [(10, 10, 90, 90), (200, 120, 420, 360)]
        track = tracker.update(boxes, frame_shape=(480, 640, 3))
        assert track is not None
        assert track.bbox == (200, 120, 420, 360)

    def test_update_prefers_temporal_continuity_over_new_face(self) -> None:
        tracker = ByteTrackerWrapper()
        first = tracker.update([(220, 120, 420, 360)], frame_shape=(480, 640, 3))
        assert first is not None
        # First candidate overlaps strongly with the previous track; second is larger
        # but should not steal focus immediately.
        track = tracker.update(
            [(225, 125, 425, 365), (120, 80, 470, 420)],
            frame_shape=(480, 640, 3),
        )
        assert track is not None
        assert track.bbox == (225, 125, 425, 365)

    def test_update_times_out_after_missing_faces(self) -> None:
        tracker = ByteTrackerWrapper(lost_ttl_frames=1)
        first = tracker.update([(10, 10, 50, 50)])
        assert first is not None

        held = tracker.update([])
        assert held is not None
        assert held.track_id == first.track_id

        assert tracker.update([]) is None

    def test_update_assigns_new_id_after_loss(self) -> None:
        tracker = ByteTrackerWrapper(lost_ttl_frames=0)
        first = tracker.update([(10, 10, 50, 50)])
        assert first is not None

        assert tracker.update([]) is None
        second = tracker.update([(10, 10, 50, 50)])
        assert second is not None
        assert second.track_id == first.track_id + 1

    def test_update_assigns_new_id_for_clear_face_switch(self) -> None:
        tracker = ByteTrackerWrapper(reid_iou_threshold=0.10)
        first = tracker.update([(10, 10, 50, 50)])
        assert first is not None

        second = tracker.update([(200, 200, 260, 260)])
        assert second is not None
        assert second.track_id == first.track_id + 1
