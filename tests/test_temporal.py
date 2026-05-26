import pytest

from src.temporal.aggregator import TemporalAggregator


def _features(closed: bool, yawing: bool, yaw: float) -> dict[str, bool | float]:
    return {"eye_closed": closed, "is_yawning": yawing, "yaw": yaw}


def test_temporal_update_metrics() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=2.0, yaw_threshold_deg=30.0)

    m1 = agg.update(_features(True, False, 0.0), track_id=1, ts=0.0)
    assert m1["perclos"] == 1.0
    assert m1["yawn_count"] == 0
    assert m1["continuous_closed"] == 0.0

    m2 = agg.update(_features(False, True, 35.0), track_id=1, ts=1.0)
    assert m2["perclos"] == 0.5
    assert m2["yawn_count"] == 1
    assert m2["gaze_away_duration"] == 0.0
    assert m2["continuous_closed"] == 0.0

    m3 = agg.update(_features(True, True, 35.0), track_id=1, ts=2.0)
    assert abs(float(m3["perclos"]) - (2 / 3)) < 1e-9
    assert m3["yawn_count"] == 2
    assert m3["gaze_away_duration"] == 0.0
    assert m3["look_right_duration"] == 0.0
    assert m3["yaw_suppressed_by_yawn"] is True
    assert m3["continuous_closed"] == 0.0  # just opened at ts=1.0

    m4 = agg.update(_features(False, False, 0.0), track_id=1, ts=3.0)
    assert m4["gaze_away_duration"] == 0.0
    assert m4["look_left_duration"] == 0.0
    assert m4["look_right_duration"] == 0.0


def test_temporal_reset_track() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=2.0, yaw_threshold_deg=30.0)
    agg.update(_features(True, True, 0.0), track_id=7, ts=0.0)
    agg.update(_features(True, True, 0.0), track_id=7, ts=1.0)

    agg.reset_track(7)
    m = agg.update(_features(False, False, 0.0), track_id=7, ts=2.0)
    assert m["perclos"] == 0.0
    assert m["yawn_count"] == 0
    assert m["continuous_closed"] == 0.0


def test_temporal_continuous_closed() -> None:
    agg = TemporalAggregator(fps=30, window_seconds=3.0, yaw_threshold_deg=30.0)
    # Simulate 4 seconds of continuous eye closure @ 30fps
    for i in range(121):
        m = agg.update(_features(True, False, 0.0), track_id=1, ts=i / 30.0)
    assert m["continuous_closed"] == pytest.approx(4.0, abs=0.1)

    # One open frame resets timer
    m = agg.update(_features(False, False, 0.0), track_id=1, ts=121 / 30.0)
    assert m["continuous_closed"] == 0.0


def test_temporal_calibrates_static_yaw_bias() -> None:
    agg = TemporalAggregator(fps=30, window_seconds=3.0, yaw_threshold_deg=30.0)

    for i in range(20):
        m = agg.update(_features(False, False, 35.0), track_id=9, ts=i / 15.0)

    assert float(m["yaw_baseline"]) > 30.0
    assert abs(float(m["yaw_corrected"])) < 5.0
    assert m["gaze_away_duration"] == 0.0


def test_temporal_head_down_can_accept_positive_pitch_mode() -> None:
    agg = TemporalAggregator(
        fps=2,
        window_seconds=3.0,
        yaw_threshold_deg=30.0,
        head_down_pitch_deg=-18.0,
        head_down_pitch_mode="both",
    )

    for i in range(15):
        agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 5.0},
            track_id=3,
            ts=i / 10.0,
        )

    m = agg.update({"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 27.0}, track_id=3, ts=2.0)
    assert m["head_down_duration"] == 0.0
    assert m["head_down_positive_duration"] == 0.0

    m = agg.update({"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 27.0}, track_id=3, ts=3.0)
    assert m["head_down_duration"] == 1.0
    assert m["head_down_positive_duration"] == 1.0


def test_temporal_calibrates_static_pitch_bias() -> None:
    agg = TemporalAggregator(fps=30, window_seconds=3.0, yaw_threshold_deg=30.0)

    for i in range(20):
        m = agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 8.0},
            track_id=10,
            ts=i / 15.0,
        )

    assert float(m["pitch_baseline"]) > 6.0
    assert abs(float(m["pitch_corrected"])) < 5.0
    assert m["head_down_duration"] == 0.0


def test_temporal_head_down_uses_pitch_delta() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0, head_down_pitch_deg=-18.0)

    for i in range(15):
        agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 10.0},
            track_id=11,
            ts=i / 10.0,
        )

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": -12.0},
        track_id=11,
        ts=2.0,
    )
    assert m["head_down_duration"] == 0.0
    assert float(m["pitch_corrected"]) < -18.0

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": -12.0},
        track_id=11,
        ts=3.0,
    )
    assert m["head_down_duration"] == 1.0


def test_temporal_uncalibrated_extreme_pitch_can_trigger_head_down() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0, head_down_pitch_deg=-18.0)

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": -35.0},
        track_id=12,
        ts=0.0,
    )
    assert m["pitch_calibrated"] is False
    assert m["head_down_duration"] == 0.0

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": -35.0},
        track_id=12,
        ts=1.0,
    )
    assert m["pitch_calibrated"] is False
    assert m["head_down_duration"] == 1.0


def test_temporal_yawn_suppresses_look_around_duration() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0)

    for i in range(15):
        agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 0.0},
            track_id=4,
            ts=i / 10.0,
        )

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 35.0, "pitch": 0.0},
        track_id=4,
        ts=2.0,
    )
    assert m["look_right_duration"] == 0.0

    m = agg.update(
        {"eye_closed": False, "is_yawning": True, "yaw": 35.0, "pitch": 0.0},
        track_id=4,
        ts=3.0,
    )
    assert m["look_right_duration"] == 0.0
    assert m["gaze_away_duration"] == 0.0
    assert m["yaw_suppressed_by_yawn"] is True

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 35.0, "pitch": 0.0},
        track_id=4,
        ts=4.0,
    )
    assert m["look_right_duration"] == 0.0


def test_temporal_yaw_wrap_does_not_create_large_turn() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0)

    for i in range(20):
        m = agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": -178.0, "pitch": 0.0},
            track_id=5,
            ts=i / 10.0,
        )

    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 178.0, "pitch": 0.0},
        track_id=5,
        ts=2.1,
    )
    assert abs(float(m["yaw_corrected"])) < 10.0
    assert m["gaze_away_duration"] == 0.0


def test_temporal_does_not_use_yaw_before_calibration() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0)

    for i in range(5):
        m = agg.update(
            {"eye_closed": False, "is_yawning": False, "yaw": 170.0, "pitch": 0.0},
            track_id=8,
            ts=i / 2.0,
        )

    assert m["yaw_baseline"] == 0.0
    assert m["yaw_corrected"] == pytest.approx(-10.0)
    assert m["gaze_away_duration"] == 0.0
    assert m["look_left_duration"] == 0.0
    assert m["look_right_duration"] == 0.0


def test_temporal_invalid_pose_does_not_accumulate_gaze() -> None:
    agg = TemporalAggregator(fps=2, window_seconds=3.0, yaw_threshold_deg=30.0)

    agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 45.0, "pitch": -25.0, "pose_valid": True},
        track_id=6,
        ts=0.0,
    )
    m = agg.update(
        {"eye_closed": False, "is_yawning": False, "yaw": 0.0, "pitch": 0.0, "pose_valid": False},
        track_id=6,
        ts=1.0,
    )
    assert m["pose_valid"] is False
    assert m["gaze_away_duration"] == 0.0
    assert m["head_down_duration"] == 0.0
