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
    assert m3["gaze_away_duration"] == 1.0
    assert m3["continuous_closed"] == 0.0  # just opened at ts=1.0

    m4 = agg.update(_features(False, False, 0.0), track_id=1, ts=3.0)
    assert m4["gaze_away_duration"] == 0.0


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


