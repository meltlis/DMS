from src.decision.rules import distraction_level, fatigue_level, fatigue_score, state_from_metrics


def test_fatigue_score_is_capped_to_one() -> None:
    score = fatigue_score(perclos=1.0, nod_freq=100.0, yawn_count=100)
    assert score == 1.0


def test_fatigue_level_thresholds() -> None:
    assert fatigue_level(perclos=0.10, warning=0.15, alert=0.40) == "NORMAL"
    assert fatigue_level(perclos=0.20, warning=0.15, alert=0.40) == "WARNING"
    assert fatigue_level(perclos=0.50, warning=0.15, alert=0.40) == "WARNING"
    assert fatigue_level(perclos=0.50, warning=0.15, alert=0.40, continuous_closed=1.0) == "ALERT"


def test_fatigue_level_continuous_closed() -> None:
    # Continuous closure takes priority over PERCLOS
    assert fatigue_level(perclos=0.0, warning=0.15, alert=0.40, continuous_closed=0.0) == "NORMAL"
    assert fatigue_level(perclos=0.0, warning=0.15, alert=0.40, continuous_closed=3.0) == "WARNING"
    assert fatigue_level(perclos=0.0, warning=0.15, alert=0.40, continuous_closed=5.0) == "ALERT"
    assert fatigue_level(perclos=0.50, warning=0.15, alert=0.40, continuous_closed=5.0) == "ALERT"


def test_distraction_priority() -> None:
    assert distraction_level(10.0, 2.0, ["PHONE", "SMOKE"]) == "PHONE"
    assert distraction_level(10.0, 2.0, ["SMOKE"]) == "SMOKE"
    assert distraction_level(10.0, 2.0, ["SUSPECTED_PHONE_USE"], head_down_duration=2.0) == "HEAD_DOWN"
    assert distraction_level(0.0, 2.0, [], look_left_duration=2.1) == "LOOK_AROUND"
    assert distraction_level(0.0, 2.0, [], look_right_duration=2.1) == "LOOK_AROUND"
    assert distraction_level(2.1, 2.0, []) == "LOOK_AROUND"


def test_state_from_metrics_maps_channels() -> None:
    metrics = {
        "perclos": 0.45,
        "nod_freq": 0.0,
        "yawn_count": 0,
        "gaze_away_duration": 0.0,
        "continuous_closed": 0.0,
    }
    thresholds = {
        "perclos_warning": 0.15,
        "perclos_alert": 0.40,
        "gaze_away_seconds": 2.0,
    }
    state = state_from_metrics(metrics, thresholds, ["NO_SEATBELT"])
    assert state == {
        "fatigue": "WARNING",
        "distraction": "NORMAL",
        "danger": "NO_SEATBELT",
    }


def test_state_from_metrics_continuous_closed() -> None:
    metrics = {
        "perclos": 0.0,
        "nod_freq": 0.0,
        "yawn_count": 0,
        "gaze_away_duration": 0.0,
        "continuous_closed": 3.5,
    }
    thresholds = {
        "perclos_warning": 0.15,
        "perclos_alert": 0.40,
        "gaze_away_seconds": 2.0,
    }
    state = state_from_metrics(metrics, thresholds, [])
    assert state["fatigue"] == "WARNING"
