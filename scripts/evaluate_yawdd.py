"""Evaluate DMS pipeline on YawDD Mirror dataset.

Ground truth is embedded in filenames:
  *-Normal.*  -> not yawning (gt=0)
  *-Yawning.* -> yawning     (gt=1)
  *-Talking.* -> excluded (distraction, not fatigue)

Metric: per-video yawning detection accuracy using the pipeline's
is_yawning signal (MAR-based) and fatigue state output.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import DMSPipeline, load_yaml


@dataclass
class YawResult:
    video: str
    label: str        # Normal / Yawning / Talking
    gt_yawning: int   # 1 = yawning video, 0 = normal
    frames: int
    yawn_frames: int          # frames where is_yawning=True
    warning_alert_frames: int
    yawn_ratio: float         # yawn_frames / frames
    warning_alert_ratio: float
    predicted_yawning: int    # 1 if yawn_ratio > threshold


def parse_label(stem: str) -> str:
    """Extract Normal / Yawning / Talking from filename stem."""
    stem_lower = stem.lower()
    if "yawning" in stem_lower:
        return "Yawning"
    if "talking" in stem_lower:
        return "Talking"
    return "Normal"


def reset_pipeline(pipeline: DMSPipeline) -> None:
    pipeline.fatigue_fsm.reset()
    pipeline.fsm.current_track_id = None
    pipeline.fsm.reset_count = 0
    pipeline.fsm.history = []
    pipeline.temporal._windows.clear()
    pipeline.temporal._closed_since.clear()
    pipeline.danger._phone_first_ts = None
    pipeline.danger._start_ts = None
    pipeline.danger._seatbelt_seen = False


def eval_video(
    pipeline: DMSPipeline,
    video_path: Path,
    max_frames: int,
    frame_stride: int,
    yawn_ratio_threshold: float,
) -> YawResult:
    reset_pipeline(pipeline)
    label = parse_label(video_path.stem)
    gt_yawning = 1 if label == "Yawning" else 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    processed = 0
    used = 0
    yawn_frames = 0
    warning_alert_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            if frame_stride > 1 and processed % frame_stride != 0:
                continue

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0.001:
                ts = processed / 30.0

            state = pipeline.process(frame, ts)
            used += 1

            if state["debug"].get("mar") is not None:
                raw_feats = state["debug"].get("raw_feats", {})
                if raw_feats.get("is_yawning", False):
                    yawn_frames += 1

            if state["fatigue"] in {"WARNING", "ALERT"}:
                warning_alert_frames += 1

            if max_frames > 0 and used >= max_frames:
                break
    finally:
        cap.release()

    yawn_ratio = (yawn_frames / used) if used else 0.0
    wa_ratio = (warning_alert_frames / used) if used else 0.0
    pred = 1 if yawn_ratio >= yawn_ratio_threshold else 0

    return YawResult(
        video=str(video_path),
        label=label,
        gt_yawning=gt_yawning,
        frames=used,
        yawn_frames=yawn_frames,
        warning_alert_frames=warning_alert_frames,
        yawn_ratio=yawn_ratio,
        warning_alert_ratio=wa_ratio,
        predicted_yawning=pred,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on YawDD Mirror dataset")
    parser.add_argument("--yawdd-mirror", default="../dataset/YawDD/Mirror/Mirror",
                        help="Path to YawDD Mirror directory")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--yawn-ratio-threshold", type=float, default=0.05,
                        help="Min fraction of yawning frames to predict yawning")
    parser.add_argument("--out-csv", default="reports/yawdd_eval.csv")
    parser.add_argument("--out-json", default="reports/yawdd_summary.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    mirror_dir = (root / args.yawdd_mirror).resolve()
    if not mirror_dir.exists():
        print(f"[ERROR] Mirror dir not found: {mirror_dir}")
        return

    videos = sorted(mirror_dir.rglob("*.avi")) + sorted(mirror_dir.rglob("*.mp4"))
    print(f"Found {len(videos)} videos in {mirror_dir}")

    results: list[YawResult] = []
    for video in videos:
        label = parse_label(video.stem)
        if label == "Talking":
            continue  # exclude talking — not a fatigue label

        r = eval_video(pipeline, video, args.max_frames, args.frame_stride, args.yawn_ratio_threshold)
        results.append(r)
        print(f"[{r.label:7s}] {video.name:<50s} yawn_ratio={r.yawn_ratio:.3f} pred={r.predicted_yawning} gt={r.gt_yawning}")

    # Accuracy
    valid = [r for r in results if r.label in {"Normal", "Yawning"}]
    correct = sum(1 for r in valid if r.predicted_yawning == r.gt_yawning)
    accuracy = correct / len(valid) if valid else None

    tp = sum(1 for r in valid if r.predicted_yawning == 1 and r.gt_yawning == 1)
    fp = sum(1 for r in valid if r.predicted_yawning == 1 and r.gt_yawning == 0)
    tn = sum(1 for r in valid if r.predicted_yawning == 0 and r.gt_yawning == 0)
    fn = sum(1 for r in valid if r.predicted_yawning == 0 and r.gt_yawning == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    summary = {
        "config": {
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "yawn_ratio_threshold": args.yawn_ratio_threshold,
        },
        "yawdd_mirror": {
            "videos_evaluated": len(valid),
            "yawning_accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        },
    }

    out_csv = (root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(YawResult.__dataclass_fields__))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    out_json = (root / args.out_json).resolve()
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
