from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np


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
    pose_valid: bool = True


class TemporalAggregator:
    def __init__(
        self,
        fps: int,
        window_seconds: float,
        yaw_threshold_deg: float,
        head_down_pitch_deg: float = -18.0,
        head_down_pitch_mode: str = "negative",
    ) -> None:
        self.maxlen = max(1, int(fps * window_seconds))
        self.yaw_threshold_deg = yaw_threshold_deg
        self.yaw_calibration_frames = 15
        self.yaw_baseline_alpha = 0.02
        self.yaw_recenter_threshold_deg = max(8.0, yaw_threshold_deg * 0.5)
        self.pitch_calibration_frames = 15
        self.pitch_baseline_alpha = 0.02
        self.pitch_recenter_threshold_deg = max(6.0, abs(float(head_down_pitch_deg)) * 0.35)
        # pitch must cross ±25° to count as a nod (15° was too sensitive)
        self.pitch_threshold_deg = 25.0
        self.head_down_pitch_deg = -abs(float(head_down_pitch_deg))
        self.head_down_pitch_abs_deg = abs(float(head_down_pitch_deg))
        self.head_down_pitch_mode = str(head_down_pitch_mode).lower()
        self._windows: Dict[int, Deque[FrameFeature]] = defaultdict(lambda: deque(maxlen=self.maxlen))
        self._closed_since: Dict[int, float | None] = {}
        self._yaw_baselines: Dict[int, float] = {}
        self._yaw_calibration: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.yaw_calibration_frames)
        )
        self._pitch_baselines: Dict[int, float] = {}
        self._pitch_calibration: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.pitch_calibration_frames)
        )

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    @classmethod
    def _angle_delta_deg(cls, angle: float, baseline: float) -> float:
        return cls._wrap_angle_deg(angle - baseline)

    @classmethod
    def _uncalibrated_yaw_delta(cls, raw_yaw: float) -> float:
        # MediaPipe often reports frontal faces near +/-180 deg in this setup,
        # while some backends report them near 0. Handle both conventions.
        if abs(raw_yaw) > 90.0:
            return cls._angle_delta_deg(raw_yaw, 180.0)
        return raw_yaw

    def reset_track(self, track_id: int) -> None:
        self._windows.pop(track_id, None)
        self._closed_since.pop(track_id, None)
        self._yaw_baselines.pop(track_id, None)
        self._yaw_calibration.pop(track_id, None)
        self._pitch_baselines.pop(track_id, None)
        self._pitch_calibration.pop(track_id, None)

    def _normalize_yaw(self, track_id: int, raw_yaw: float) -> tuple[float, float | None]:
        baseline = self._yaw_baselines.get(track_id)
        if baseline is None:
            provisional_yaw = self._uncalibrated_yaw_delta(raw_yaw)
            if abs(provisional_yaw) <= max(12.0, self.yaw_threshold_deg * 1.5):
                calibration = self._yaw_calibration[track_id]
                calibration.append(raw_yaw)
            else:
                calibration = self._yaw_calibration[track_id]
            if len(calibration) >= self.yaw_calibration_frames:
                baseline = float(np.median(np.asarray(calibration, dtype=np.float32)))
                self._yaw_baselines[track_id] = baseline

        if baseline is None:
            return self._uncalibrated_yaw_delta(raw_yaw), None

        corrected_yaw = self._angle_delta_deg(raw_yaw, baseline)
        if abs(corrected_yaw) < self.yaw_recenter_threshold_deg:
            baseline = self._wrap_angle_deg(baseline + self.yaw_baseline_alpha * corrected_yaw)
            self._yaw_baselines[track_id] = baseline
            corrected_yaw = self._angle_delta_deg(raw_yaw, baseline)

        return corrected_yaw, baseline

    def _normalize_pitch(self, track_id: int, raw_pitch: float) -> tuple[float, float | None]:
        baseline = self._pitch_baselines.get(track_id)
        if baseline is None:
            if abs(raw_pitch) <= max(12.0, self.head_down_pitch_abs_deg * 0.75):
                calibration = self._pitch_calibration[track_id]
                calibration.append(raw_pitch)
            else:
                calibration = self._pitch_calibration[track_id]
            if len(calibration) >= self.pitch_calibration_frames:
                baseline = float(np.median(np.asarray(calibration, dtype=np.float32)))
                self._pitch_baselines[track_id] = baseline

        if baseline is None:
            return raw_pitch, None

        corrected_pitch = raw_pitch - baseline
        if abs(corrected_pitch) < self.pitch_recenter_threshold_deg:
            baseline = baseline + self.pitch_baseline_alpha * corrected_pitch
            self._pitch_baselines[track_id] = baseline
            corrected_pitch = raw_pitch - baseline

        return corrected_pitch, baseline

    def update(self, features: Dict[str, float | bool], track_id: int, ts: float) -> Dict[str, float | int]:
        win = self._windows[track_id]
        raw_yaw = float(features.get("yaw", 0.0))
        raw_pitch = float(features.get("pitch", 0.0))
        pose_valid = bool(features.get("pose_valid", True))
        if pose_valid:
            corrected_yaw, yaw_baseline = self._normalize_yaw(track_id, raw_yaw)
            corrected_pitch, pitch_baseline = self._normalize_pitch(track_id, raw_pitch)
        else:
            corrected_yaw = 0.0
            yaw_baseline = self._yaw_baselines.get(track_id, 0.0)
            corrected_pitch = 0.0
            pitch_baseline = self._pitch_baselines.get(track_id, 0.0)
        win.append(
            FrameFeature(
                ts=ts,
                eye_closed=bool(features.get("eye_closed", False)),
                is_yawning=bool(features.get("is_yawning", False)),
                yaw=corrected_yaw,
                pitch=corrected_pitch,
                ear=float(min(features.get("ear_left", 1.0), features.get("ear_right", 1.0))),
                mar=float(features.get("mar", 0.0)),
                roll=float(features.get("roll", 0.0)) if pose_valid else 0.0,
                pose_valid=pose_valid,
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
        look_left_duration = 0.0
        look_right_duration = 0.0
        head_down_duration = 0.0
        head_down_negative_duration = 0.0
        head_down_positive_duration = 0.0
        head_up_duration = 0.0
        yaw_suppressed_by_yawn = False

        # Nod: count distinct pitch-drop events (head dips below -pitch_threshold_deg)
        # then normalise to events-per-second over the actual window duration
        prev_pitch = 0.0
        drops = 0
        for f in win:
            if not f.pose_valid:
                prev_pitch = 0.0
                continue
            if f.pitch < -self.pitch_threshold_deg and prev_pitch >= -self.pitch_threshold_deg:
                drops += 1
            prev_pitch = f.pitch
        window_dur = (win[-1].ts - win[0].ts) if total > 1 else 1.0
        nod_freq = drops / max(window_dur, 1.0)  # events per second

        if total > 0:
            current = win[-1]
            current_yaw = current.yaw
            yaw_suppressed_by_yawn = current.is_yawning and abs(current_yaw) > self.yaw_threshold_deg
            yaw_valid = current.pose_valid and not current.is_yawning
            if yaw_valid and abs(current_yaw) > self.yaw_threshold_deg:
                for item in reversed(win):
                    if not item.pose_valid or item.is_yawning:
                        break
                    if abs(item.yaw) > self.yaw_threshold_deg:
                        gaze_away_duration = win[-1].ts - item.ts
                    else:
                        break
            if yaw_valid and current_yaw <= -self.yaw_threshold_deg:
                for item in reversed(win):
                    if not item.pose_valid or item.is_yawning:
                        break
                    if item.yaw <= -self.yaw_threshold_deg:
                        look_left_duration = win[-1].ts - item.ts
                    else:
                        break
            elif yaw_valid and current_yaw >= self.yaw_threshold_deg:
                for item in reversed(win):
                    if not item.pose_valid or item.is_yawning:
                        break
                    if item.yaw >= self.yaw_threshold_deg:
                        look_right_duration = win[-1].ts - item.ts
                    else:
                        break

        if total > 0 and win[-1].pose_valid and win[-1].pitch <= self.head_down_pitch_deg:
            for item in reversed(win):
                if not item.pose_valid:
                    break
                if item.pitch <= self.head_down_pitch_deg:
                    head_down_negative_duration = win[-1].ts - item.ts
                else:
                    break

        if total > 0 and win[-1].pose_valid and win[-1].pitch >= self.head_down_pitch_abs_deg:
            for item in reversed(win):
                if not item.pose_valid:
                    break
                if item.pitch >= self.head_down_pitch_abs_deg:
                    head_down_positive_duration = win[-1].ts - item.ts
                else:
                    break

        if self.head_down_pitch_mode == "positive":
            head_down_duration = head_down_positive_duration
        elif self.head_down_pitch_mode == "both":
            head_down_duration = max(head_down_negative_duration, head_down_positive_duration)
        else:
            head_down_duration = head_down_negative_duration

        if total > 0 and win[-1].pose_valid and win[-1].pitch > self.pitch_threshold_deg:
            for item in reversed(win):
                if not item.pose_valid:
                    break
                if item.pitch > self.pitch_threshold_deg:
                    head_up_duration = win[-1].ts - item.ts
                else:
                    break

        # Extract recent raw states for LSTM
        raw_sequence = [[f.ear, f.mar, f.pitch, f.yaw, f.roll] for f in win]
        
        return {
            "perclos": (closed / total) if total else 0.0,
            "nod_freq": nod_freq,
            "yawn_count": yawning,
            "gaze_away_duration": gaze_away_duration,
            "look_left_duration": look_left_duration,
            "look_right_duration": look_right_duration,
            "head_down_duration": head_down_duration,
            "head_down_negative_duration": head_down_negative_duration,
            "head_down_positive_duration": head_down_positive_duration,
            "head_up_duration": head_up_duration,
            "continuous_closed": continuous_closed,
            "yaw_corrected": corrected_yaw,
            "yaw_baseline": yaw_baseline if yaw_baseline is not None else 0.0,
            "pitch_corrected": corrected_pitch,
            "pitch_baseline": pitch_baseline if pitch_baseline is not None else 0.0,
            "raw_pitch": raw_pitch,
            "yaw_calibrated": yaw_baseline is not None,
            "pitch_calibrated": pitch_baseline is not None,
            "pose_valid": pose_valid,
            "yaw_suppressed_by_yawn": yaw_suppressed_by_yawn,
            "raw_sequence": raw_sequence,
        }
