from __future__ import annotations

from typing import Dict


def fatigue_score(perclos: float, nod_freq: float, yawn_count: int) -> float:
    return (
        0.5 * min(perclos / 0.40, 1.0)
        + 0.3 * min(nod_freq / 10.0, 1.0)
        + 0.2 * min(yawn_count / 3.0, 1.0)
    )


def fatigue_level(
    perclos: float,
    warning: float,
    alert: float,
    continuous_closed: float = 0.0,
    yawn_count: int = 0,
    nod_freq: float = 0.0,
) -> str:
    # Priority: continuous eye-closure duration (CLAUDE.md acceptance criterion)
    if continuous_closed >= 5.0:
        return "ALERT"
    if continuous_closed >= 3.0:
        return "WARNING"
        
    score = fatigue_score(perclos, nod_freq, yawn_count)
    if score >= 0.5:  # perclos alone at alert threshold (0.40) yields score=0.5 exactly
        return "ALERT"
    if score > 0.25:
        return "WARNING"
        
    # Fallback to PERCLOS thresholds
    if perclos > alert or yawn_count >= 5:
        return "ALERT"
    if perclos > warning or yawn_count >= 2:
        return "WARNING"
    return "NORMAL"


def distraction_level(gaze_away_duration: float, gaze_away_seconds: float, danger_alerts: list[str]) -> str:
    if "PHONE" in danger_alerts:
        return "PHONE"
    if "SMOKE" in danger_alerts:
        return "SMOKE"
    if gaze_away_duration >= gaze_away_seconds:
        return "GAZE_AWAY"
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
    )
    danger = "NO_SEATBELT" if "NO_SEATBELT" in danger_alerts else "NORMAL"
    return {"fatigue": fatigue, "distraction": distraction, "danger": danger}
