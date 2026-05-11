from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict


@dataclass
class FrameFeature:
    ts: float
    eye_closed: bool
    is_yawning: bool
    yaw: float
    pitch: float
    ear: float = 0.0
    mar: float = 0.0
    roll: float = 0.0


class TemporalAggregator:
    def __init__(self, fps: int, window_seconds: float, yaw_threshold_deg: float) -> None:
        self.maxlen = max(1, int(fps * window_seconds))
        self.yaw_threshold_deg = yaw_threshold_deg
        # pitch must cross ±25° to count as a nod (15° was too sensitive)
        self.pitch_threshold_deg = 25.0
        self._windows: Dict[int, Deque[FrameFeature]] = defaultdict(lambda: deque(maxlen=self.maxlen))
        self._closed_since: Dict[int, float | None] = {}

    def reset_track(self, track_id: int) -> None:
        self._windows.pop(track_id, None)
        self._closed_since.pop(track_id, None)

    def update(self, features: Dict[str, float | bool], track_id: int, ts: float) -> Dict[str, float | int]:
        win = self._windows[track_id]
        win.append(
            FrameFeature(
                ts=ts,
                eye_closed=bool(features.get("eye_closed", False)),
                is_yawning=bool(features.get("is_yawning", False)),
                yaw=float(features.get("yaw", 0.0)),
                pitch=float(features.get("pitch", 0.0)),
                ear=float(min(features.get("ear_left", 1.0), features.get("ear_right", 1.0))),
                mar=float(features.get("mar", 0.0)),
                roll=float(features.get("roll", 0.0)),
            )
        )
        total = len(win)
        closed = sum(1 for f in win if f.eye_closed)
        yawning = sum(1 for f in win if f.is_yawning)

        if bool(features["eye_closed"]):
            if self._closed_since.get(track_id) is None:
                self._closed_since[track_id] = ts
            continuous_closed = ts - self._closed_since[track_id]
        else:
            self._closed_since[track_id] = None
            continuous_closed = 0.0

        nod_count = 0
        gaze_away_duration = 0.0

        # Nod: count distinct pitch-drop events (head dips below -pitch_threshold_deg)
        # then normalise to events-per-second over the actual window duration
        prev_pitch = 0.0
        drops = 0
        for f in win:
            if f.pitch < -self.pitch_threshold_deg and prev_pitch >= -self.pitch_threshold_deg:
                drops += 1
            prev_pitch = f.pitch
        window_dur = (win[-1].ts - win[0].ts) if total > 1 else 1.0
        nod_freq = drops / max(window_dur, 1.0)  # events per second

        if total > 0 and abs(win[-1].yaw) > self.yaw_threshold_deg:
            for item in reversed(win):
                if abs(item.yaw) > self.yaw_threshold_deg:
                    gaze_away_duration = win[-1].ts - item.ts
                else:
                    break

        # Extract recent raw states for LSTM
        raw_sequence = [[f.ear, f.mar, f.pitch, f.yaw, f.roll] for f in win]
        
        return {
            "perclos": (closed / total) if total else 0.0,
            "nod_freq": nod_freq,
            "yawn_count": yawning,
            "gaze_away_duration": gaze_away_duration,
            "continuous_closed": continuous_closed,
            "raw_sequence": raw_sequence,
        }
