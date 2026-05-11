from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from src.pipeline import DMSPipeline, load_yaml


@dataclass
class VideoEvalResult:
    dataset: str
    video: str
    frames: int
    elapsed_sec: float
    avg_fps: float
    fatigue_alert_frames: int
    distraction_alert_frames: int
    danger_alert_frames: int
    total_alert_events: int


def iter_videos(root: Path, pattern: str, limit: int) -> Iterable[Path]:
    files = sorted(root.rglob(pattern))
    if limit > 0:
        return files[:limit]
    return files


def evaluate_video(pipeline: DMSPipeline, video_path: Path, max_frames: int) -> VideoEvalResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    processed = 0
    alert_counter: Counter[str] = Counter()
    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            state = pipeline.process(frame, time.time())
            processed += 1
            alert_counter.update(state["alerts"])

            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        cap.release()

    elapsed = max(1e-6, time.time() - start)
    return VideoEvalResult(
        dataset=video_path.parents[1].name,
        video=str(video_path),
        frames=processed,
        elapsed_sec=elapsed,
        avg_fps=processed / elapsed,
        fatigue_alert_frames=alert_counter["WARNING"] + alert_counter["ALERT"],
        distraction_alert_frames=alert_counter["PHONE"] + alert_counter["SMOKE"] + alert_counter["GAZE_AWAY"],
        danger_alert_frames=alert_counter["NO_SEATBELT"],
        total_alert_events=sum(alert_counter.values()),
    )


def write_csv(results: list[VideoEvalResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "video",
                "frames",
                "elapsed_sec",
                "avg_fps",
                "fatigue_alert_frames",
                "distraction_alert_frames",
                "danger_alert_frames",
                "total_alert_events",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch evaluate videos and summarize FPS/alerts")
    parser.add_argument(
        "--dataset-root",
        action="append",
        dest="dataset_roots",
        help="dataset root to scan recursively for videos",
    )
    parser.add_argument("--pattern", default="*.mp4", help="glob pattern for videos")
    parser.add_argument("--max-frames", type=int, default=300, help="max frames per video; <=0 means full video")
    parser.add_argument("--limit-per-dataset", type=int, default=3, help="max videos per dataset root; <=0 means all")
    parser.add_argument("--out-csv", default="reports/batch_summary.csv", help="output csv path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]

    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    dataset_roots = args.dataset_roots or [
        str((root / ".." / "dataset" / "inner_mirror").resolve()),
        str((root / ".." / "dataset" / "DROZY" / "DROZY" / "videos_i8").resolve()),
    ]

    results: list[VideoEvalResult] = []
    for ds in dataset_roots:
        ds_path = Path(ds)
        if not ds_path.exists():
            print(f"[skip] dataset root not found: {ds_path}")
            continue

        videos = list(iter_videos(ds_path, args.pattern, args.limit_per_dataset))
        if not videos:
            print(f"[skip] no videos found in: {ds_path}")
            continue

        for video in videos:
            try:
                result = evaluate_video(pipeline, video, args.max_frames)
                results.append(result)
                print(
                    f"[ok] {video.name} frames={result.frames} "
                    f"fps={result.avg_fps:.2f} alerts={result.total_alert_events}"
                )
            except Exception as exc:  # pragma: no cover
                print(f"[error] {video}: {exc}")

    out_csv = (root / args.out_csv).resolve()
    write_csv(results, out_csv)

    if results:
        avg_fps = sum(r.avg_fps for r in results) / len(results)
        total_alerts = sum(r.total_alert_events for r in results)
        print(f"\nsummary videos={len(results)} avg_fps={avg_fps:.2f} total_alert_events={total_alerts}")
    else:
        print("\nsummary videos=0")
    print(f"csv={out_csv}")


if __name__ == "__main__":
    main()
