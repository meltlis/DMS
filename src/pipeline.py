from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np
import yaml

from src.decision.danger import DangerDetector
from src.decision.fsm import PipelineFSM, FatigueStateFSM
from src.decision.rules import state_from_metrics
from src.decision.lstm_classifier import LSTMClassifier
from src.features.face_analyzer import FaceAnalyzer
from src.perception.detector import YOLODetector
from src.perception.tracker import ByteTrackerWrapper
from src.temporal.aggregator import TemporalAggregator


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DMSPipeline:
    def __init__(self, thresholds: Dict[str, float], runtime: Dict[str, Any]) -> None:
        self.thresholds = thresholds
        self.runtime = runtime
        # Prefer fine-tuned 4-class DMS model (phone/cigarette/seatbelt/face)
        dms4class_model = Path(__file__).resolve().parents[1] / "runs" / "detect" / "dms4class" / "weights" / "best.pt"
        standard_model = Path(__file__).resolve().parents[1] / "weights" / "yolov11n.pt"
        if dms4class_model.exists():
            chosen_model = str(dms4class_model)
        elif standard_model.exists():
            chosen_model = str(standard_model)
        else:
            chosen_model = None
        self.detector = YOLODetector(
            model_path=chosen_model,
            device=str(runtime.get("device", "cpu")),
            conf_threshold=float(thresholds.get("yolo_confidence", 0.5))
        )
        self.tracker = ByteTrackerWrapper()
        self.face_analyzer = FaceAnalyzer(
            ear_threshold=float(thresholds["ear_threshold"]),
            mar_threshold=float(thresholds["mar_threshold"]),
        )
        self.temporal = TemporalAggregator(
            fps=int(runtime.get("fps", 30)),
            window_seconds=float(thresholds["window_seconds"]),
            yaw_threshold_deg=float(thresholds["yaw_threshold_deg"]),
        )
        self.danger = DangerDetector(
            phone_iou_threshold=float(thresholds["phone_iou_threshold"]),
            phone_duration_seconds=float(thresholds["phone_duration_seconds"]),
            seatbelt_grace_seconds=10.0,
        )
        lstm_path = Path(__file__).resolve().parents[3] / "Drowsiness-Detection-based-on-yolo11-and-LSTM" / "lstm_model.pth"
        self.lstm = LSTMClassifier(model_path=str(lstm_path), seq_len=30, thresholds=thresholds)
        self.fsm = PipelineFSM()
        self.fatigue_fsm = FatigueStateFSM(downgrade_frames=10)
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one dummy inference to compile MediaPipe GPU kernels."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.face_analyzer.analyze(dummy)
        except Exception:
            pass

    def process(self, frame: np.ndarray, ts: float) -> Dict[str, Any]:
        detections = self.detector.detect(frame)
        grouped = YOLODetector.to_grouped_dict(detections, self.detector.behavior_mode)
        track = self.tracker.update(grouped["face"])

        if track is None:
            return {
                "fatigue": "NORMAL",
                "distraction": "NORMAL",
                "danger": "NORMAL",
                "alerts": [],
                "track_id": -1,
                "debug": {
                    "ear_left": None, "ear_right": None, "mar": None,
                    "pitch": None, "yaw": None, "roll": None,
                    "perclos": None, "nod_freq": None, "yawn_count": None,
                    "gaze_away_duration": None, "continuous_closed": None,
                    "lstm_score": None, "lstm_pred": None,
                    "face_bbox": None, "landmarks_global": None, "all_bboxes": grouped,
                },
            }

        if self.fsm.on_track(track.track_id):
            self.temporal.reset_track(track.track_id)
            self.fatigue_fsm.reset()

        x1, y1, x2, y2 = track.bbox
        x, y = max(0, x1), max(0, y1)
        w, h = max(0, x2 - x1), max(0, y2 - y1)

        frame_h, frame_w = frame.shape[:2]
        y_end, x_end = min(frame_h, y + h), min(frame_w, x + w)
        face_roi = frame[y:y_end, x:x_end]
        feats = self.face_analyzer.analyze(face_roi)
        metrics = self.temporal.update(feats, track.track_id, ts)
        danger_alerts = self.danger.detect(grouped, track.bbox, ts)
        states = state_from_metrics(metrics, self.thresholds, danger_alerts)

        # LSTM augments rule engine — can upgrade but never downgrade hard rule decisions
        raw_seq = metrics.pop("raw_sequence", [])
        lstm_score, lstm_pred = self.lstm.predict(raw_seq)

        rule_fatigue = states["fatigue"]
        if rule_fatigue == "NORMAL" and lstm_score > 0.65:
            states["fatigue"] = "WARNING"
        elif rule_fatigue == "WARNING" and lstm_score > 0.80:
            states["fatigue"] = "ALERT"
        # ALERT from hard rules (continuous_closed, PERCLOS) is never downgraded by LSTM

        # Apply debounce: upgrade immediately, downgrade only after N frames
        states["fatigue"] = self.fatigue_fsm.update(states["fatigue"])

        alerts = [a for a in [states["distraction"], states["danger"], states["fatigue"]] if a != "NORMAL"]
        
        # Remap landmarks from ROI space to full-frame space for visualization
        landmarks_468 = feats.get("landmarks_468")
        landmarks_global = None
        if landmarks_468 is not None:
            landmarks_global = landmarks_468.copy()
            landmarks_global[:, 0] += x
            landmarks_global[:, 1] += y

        debug = {
            "ear_left": float(feats["ear_left"]),
            "ear_right": float(feats["ear_right"]),
            "mar": float(feats["mar"]),
            "pitch": float(feats["pitch"]),
            "yaw": float(feats["yaw"]),
            "roll": float(feats["roll"]),
            "perclos": float(metrics["perclos"]),
            "nod_freq": float(metrics["nod_freq"]),
            "yawn_count": int(metrics["yawn_count"]),
            "gaze_away_duration": float(metrics["gaze_away_duration"]),
            "continuous_closed": float(metrics.get("continuous_closed", 0.0)),
            "lstm_score": float(lstm_score),
            "lstm_pred": int(lstm_pred),
            "face_bbox": [int(x), int(y), int(w), int(h)],
            "landmarks_global": landmarks_global,
            "all_bboxes": grouped
        }

        debug["raw_feats"] = feats

        return {
            "fatigue": states["fatigue"],
            "distraction": states["distraction"],
            "danger": states["danger"],
            "alerts": alerts,
            "track_id": int(track.track_id),
            "debug": debug,
        }


def synthetic_frames(width: int, height: int) -> Iterable[np.ndarray]:
    t = 0
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        intensity = int((np.sin(t / 10.0) * 0.5 + 0.5) * 255)
        frame[:, :, :] = intensity
        t += 1
        yield frame


def run_pipeline(source: str, max_frames: int, show: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")

    logging.basicConfig(level=getattr(logging, str(runtime.get("log_level", "INFO"))))
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    width = int(runtime.get("width", 640))
    height = int(runtime.get("height", 480))
    start = time.time()
    processed = 0

    if source == "synthetic":
        frames_iter = synthetic_frames(width, height)
        capture = None
    else:
        capture = cv2.VideoCapture(0 if source == "0" else source)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open source: {source}")
        frames_iter = None

    try:
        while True:
            if frames_iter is not None:
                frame = next(frames_iter)
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            ts = time.time()
            state = pipeline.process(frame, ts)
            processed += 1

            if show:
                cv2.putText(frame, str(state["alerts"]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("dms-pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if processed % 30 == 0:
                elapsed = max(1e-6, time.time() - start)
                fps = processed / elapsed
                logging.info("processed=%s fps=%.2f last=%s", processed, fps, state)

            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if show:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DMS pipeline")
    parser.add_argument("--source", default="synthetic", help="0 | video path | synthetic")
    parser.add_argument("--max-frames", type=int, default=90, help="stop after N frames; <=0 for no limit")
    parser.add_argument("--show", action="store_true", help="show preview window")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(source=args.source, max_frames=args.max_frames, show=args.show)


if __name__ == "__main__":
    main()
