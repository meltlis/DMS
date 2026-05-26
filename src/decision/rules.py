from __future__ import annotations

from typing import Dict


def fatigue_score(perclos: float, nod_freq: float, yawn_count: int) -> float:
    return (
        0.65 * min(perclos / 0.55, 1.0)
        + 0.20 * min(nod_freq / 1.2, 1.0)
        + 0.15 * min(yawn_count / 24.0, 1.0)
    )


def fatigue_level(
    perclos: float,
    warning: float,
    alert: float,
    continuous_closed: float = 0.0,
    yawn_count: int = 0,
    nod_freq: float = 0.0,
) -> str:
    # Severe fatigue should require hard evidence, not a single noisy frame.
    if continuous_closed >= 5.0:
        return "ALERT"
    if continuous_closed >= 3.0:
        return "WARNING"

    sustained_yawn = yawn_count >= 18
    frequent_nods = nod_freq >= 0.8

    if perclos >= alert and (continuous_closed >= 1.0 or sustained_yawn or frequent_nods):
        return "ALERT"

    score = fatigue_score(perclos, nod_freq, yawn_count)
    if score >= 0.58 and (perclos >= warning or sustained_yawn or frequent_nods):
        return "WARNING"

    if perclos >= warning or sustained_yawn or frequent_nods:
        return "WARNING"
    return "NORMAL"


def distraction_level(
    gaze_away_duration: float,
    gaze_away_seconds: float,
    danger_alerts: list[str],
    head_down_duration: float = 0.0,
    head_down_seconds: float = 1.5,
    look_left_duration: float = 0.0,
    look_right_duration: float = 0.0,
    look_around_seconds: float | None = None,
) -> str:
    for alert in ("PHONE_CALL", "PHONE_USE", "PHONE", "DRINK", "SMOKE"):
        if alert in danger_alerts:
            return alert
    if head_down_duration >= head_down_seconds:
        return "HEAD_DOWN"
    turn_seconds = gaze_away_seconds if look_around_seconds is None else look_around_seconds
    if look_left_duration >= turn_seconds or look_right_duration >= turn_seconds:
        return "LOOK_AROUND"
    if "SUSPECTED_PHONE_USE" in danger_alerts:
        return "SUSPECTED_PHONE_USE"
    if gaze_away_duration >= gaze_away_seconds:
        return "LOOK_AROUND"
    return "NORMAL"


def state_from_metrics(metrics: Dict[str, float | int], thresholds: Dict[str, float], danger_alerts: list[str]) -> Dict[str, str]:
    fatigue = fatigue_level(
        perclos=float(metrics["perclos"]),
        warning=float(thresholds["perclos_warning"]),
        alert=float(thresholds["perclos_alert"]),
        continuous_closed=float(metrics.get("continuous_closed", 0.0)),
        yawn_count=int(metrics.get("yawn_count", 0)),
        nod_freq=float(metrics.get("nod_freq", 0.0)),
    )
    distraction = distraction_level(
        gaze_away_duration=float(metrics["gaze_away_duration"]),
        gaze_away_seconds=float(thresholds["gaze_away_seconds"]),
        danger_alerts=danger_alerts,
        head_down_duration=float(metrics.get("head_down_duration", 0.0)),
        head_down_seconds=float(thresholds.get("head_down_seconds", 1.5)),
        look_left_duration=float(metrics.get("look_left_duration", 0.0)),
        look_right_duration=float(metrics.get("look_right_duration", 0.0)),
        look_around_seconds=float(thresholds.get("look_around_seconds", thresholds["gaze_away_seconds"])),
    )
    danger = "NO_SEATBELT" if "NO_SEATBELT" in danger_alerts else "NORMAL"
    return {"fatigue": fatigue, "distraction": distraction, "danger": danger}
