from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.pipeline import DMSPipeline, load_yaml


@dataclass
class VideoFatigueEval:
    dataset: str
    video: str
    subject: int | None
    test: int | None
    frames: int
    warning_or_alert_frames: int
    warning_or_alert_ratio: float
    predicted_sleepy: int
    gt_sleepy: int | None
    kss: int | None


def parse_drozy_kss(kss_path: Path) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    rows = [line.strip() for line in kss_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for subject_idx, row in enumerate(rows, start=1):
        cols = [int(x) for x in row.split()]
        for test_idx, kss in enumerate(cols, start=1):
            mapping[(subject_idx, test_idx)] = kss
    return mapping


def parse_subject_test_from_stem(stem: str) -> tuple[int | None, int | None]:
    # DROZY filenames are like "1-1".
    parts = stem.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None, None


def reset_pipeline_state(pipeline: DMSPipeline) -> None:
    """Reset all stateful components between videos for independent evaluation."""
    pipeline.fatigue_fsm.reset()
    pipeline.fsm.current_track_id = None
    pipeline.fsm.reset_count = 0
    pipeline.fsm.history = []
    pipeline.tracker.reset()
    pipeline.object_smoother.reset()
    pipeline.temporal._windows.clear()  # type: ignore[attr-defined]
    pipeline.temporal._closed_since.clear()  # type: ignore[attr-defined]
    pipeline.danger._phone_first_ts = None
    pipeline.danger._start_ts = None
    pipeline.danger._seatbelt_seen = False


def eval_video_fatigue(
    pipeline: DMSPipeline,
    video_path: Path,
    dataset_name: str,
    sleepy_ratio_threshold: float,
    max_frames: int,
    frame_stride: int,
    drozy_kss_map: dict[tuple[int, int], int] | None = None,
) -> VideoFatigueEval:
    reset_pipeline_state(pipeline)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    processed = 0
    used = 0
    warning_or_alert = 0
    ts = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1

            if frame_stride > 1 and (processed % frame_stride != 0):
                continue

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0.001:
                ts = processed / 30.0

            state = pipeline.process(frame, ts)
            used += 1
            if state["fatigue"] in {"WARNING", "ALERT"}:
                warning_or_alert += 1

            if max_frames > 0 and used >= max_frames:
                break
    finally:
        cap.release()

    ratio = (warning_or_alert / used) if used else 0.0
    pred_sleepy = 1 if ratio >= sleepy_ratio_threshold else 0

    subject, test = parse_subject_test_from_stem(video_path.stem)
    gt_sleepy: int | None = None
    kss: int | None = None
    if drozy_kss_map is not None and subject is not None and test is not None:
        kss = drozy_kss_map.get((subject, test))
        if kss is not None and kss > 0:
            gt_sleepy = 1 if kss >= 7 else 0

    return VideoFatigueEval(
        dataset=dataset_name,
        video=str(video_path),
        subject=subject,
        test=test,
        frames=used,
        warning_or_alert_frames=warning_or_alert,
        warning_or_alert_ratio=ratio,
        predicted_sleepy=pred_sleepy,
        gt_sleepy=gt_sleepy,
        kss=kss,
    )


def write_csv(path: Path, rows: list[VideoFatigueEval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "video",
                "subject",
                "test",
                "frames",
                "warning_or_alert_frames",
                "warning_or_alert_ratio",
                "predicted_sleepy",
                "gt_sleepy",
                "kss",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fatigue accuracy on datasets")
    parser.add_argument("--drozy-videos", default="../dataset/DROZY/DROZY/videos_i8", help="DROZY videos path")
    parser.add_argument("--drozy-kss", default="../dataset/DROZY/DROZY/KSS.txt", help="DROZY KSS file")
    parser.add_argument("--inner-videos", default="../dataset/inner_mirror", help="inner_mirror videos root")
    parser.add_argument("--max-frames", type=int, default=3000, help="max sampled frames per video; <=0 full")
    parser.add_argument("--frame-stride", type=int, default=3, help="sample every Nth frame")
    parser.add_argument("--min-reliable-frames", type=int, default=300, help="minimum sampled frames per video for reliable summary")
    parser.add_argument("--sleepy-ratio-threshold", type=float, default=0.15, help="predict sleepy if warning/alert ratio >= this value")
    parser.add_argument("--out-csv", default="reports/accuracy_eval.csv", help="output csv")
    parser.add_argument("--out-json", default="reports/accuracy_summary.json", help="output summary json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]

    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    drozy_dir = (root / args.drozy_videos).resolve()
    drozy_kss = (root / args.drozy_kss).resolve()
    inner_dir = (root / args.inner_videos).resolve()

    results: list[VideoFatigueEval] = []

    # DROZY: has KSS labels
    if drozy_dir.exists() and drozy_kss.exists():
        kss_map = parse_drozy_kss(drozy_kss)
        for video in sorted(drozy_dir.glob("*.mp4")):
            row = eval_video_fatigue(
                pipeline=pipeline,
                video_path=video,
                dataset_name="DROZY",
                sleepy_ratio_threshold=args.sleepy_ratio_threshold,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                drozy_kss_map=kss_map,
            )
            results.append(row)
            print(
                f"[DROZY] {video.name} ratio={row.warning_or_alert_ratio:.3f} "
                f"pred={row.predicted_sleepy} gt={row.gt_sleepy}"
            )

    # inner_mirror: no ground-truth fatigue labels in files (only calibration metadata)
    if inner_dir.exists():
        for video in sorted(inner_dir.rglob("*.mp4")):
            row = eval_video_fatigue(
                pipeline=pipeline,
                video_path=video,
                dataset_name="inner_mirror",
                sleepy_ratio_threshold=args.sleepy_ratio_threshold,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                drozy_kss_map=None,
            )
            results.append(row)
            print(f"[inner_mirror] {video.name} ratio={row.warning_or_alert_ratio:.3f} pred={row.predicted_sleepy}")

    write_csv((root / args.out_csv).resolve(), results)

    drozy_valid = [r for r in results if r.dataset == "DROZY" and r.gt_sleepy is not None]
    if drozy_valid:
        correct = sum(1 for r in drozy_valid if r.predicted_sleepy == r.gt_sleepy)
        drozy_acc = correct / len(drozy_valid)
    else:
        drozy_acc = None

    inner_valid = [r for r in results if r.dataset == "inner_mirror" and r.gt_sleepy is not None]
    inner_acc = None
    if inner_valid:
        correct = sum(1 for r in inner_valid if r.predicted_sleepy == r.gt_sleepy)
        inner_acc = correct / len(inner_valid)

    summary = {
        "config": {
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "min_reliable_frames": args.min_reliable_frames,
            "sleepy_ratio_threshold": args.sleepy_ratio_threshold,
        },
        "drozy": {
            "videos_total": len([r for r in results if r.dataset == "DROZY"]),
            "videos_with_gt": len(drozy_valid),
            "fatigue_binary_accuracy": drozy_acc,
            "gt_rule": "sleepy if KSS>=7",
        },
        "inner_mirror": {
            "videos_total": len([r for r in results if r.dataset == "inner_mirror"]),
            "videos_with_gt": len(inner_valid),
            "fatigue_binary_accuracy": inner_acc,
            "note": "No fatigue ground-truth labels found in dataset files; accuracy is not computable.",
        },
    }
    evaluated = [r for r in results if r.gt_sleepy is not None]
    min_used_frames = min((r.frames for r in evaluated), default=0)
    summary["reliability"] = {
        "is_reliable": bool(evaluated) and min_used_frames >= args.min_reliable_frames,
        "min_used_frames": min_used_frames,
        "videos_with_ground_truth": len(evaluated),
        "note": (
            "Use summary accuracy only when each evaluated video has enough sampled frames; "
            "very small max_frames values are smoke tests, not statistically reliable accuracy."
        ),
    }

    out_json = (root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"csv={(root / args.out_csv).resolve()}")
    print(f"json={out_json}")


if __name__ == "__main__":
    main()
