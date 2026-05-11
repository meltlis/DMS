from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

# Allow imports from parent directory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import DMSPipeline, load_yaml


def profile(source: str, duration: int) -> None:
    root = Path(__file__).resolve().parents[1]
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)

    if source == "synthetic":
        from src.pipeline import synthetic_frames
        frames_iter = synthetic_frames(int(runtime.get("width", 640)), int(runtime.get("height", 480)))
        capture = None
    else:
        capture = cv2.VideoCapture(0 if source == "0" else source)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open source: {source}")
        frames_iter = None

    processed = 0
    start = time.time()
    end_time = start + duration
    fps_samples: list[float] = []

    try:
        while time.time() < end_time:
            if frames_iter is not None:
                frame = next(frames_iter)
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            ts = time.time()
            pipeline.process(frame, ts)
            processed += 1

            if processed % 30 == 0:
                elapsed = max(1e-6, ts - start)
                fps = processed / elapsed
                fps_samples.append(fps)
                logging.info("processed=%d fps=%.2f", processed, fps)
    finally:
        if capture is not None:
            capture.release()

    total_elapsed = max(1e-6, time.time() - start)
    avg_fps = processed / total_elapsed
    logging.info("=" * 50)
    logging.info("Profile complete: %d frames in %.1fs", processed, total_elapsed)
    logging.info("Average FPS: %.2f", avg_fps)
    if fps_samples:
        logging.info("Min FPS: %.2f", min(fps_samples))
        logging.info("Max FPS: %.2f", max(fps_samples))
    logging.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="DMS FPS profiling")
    parser.add_argument("--source", default="synthetic", help="video path | 0 | synthetic")
    parser.add_argument("--duration", type=int, default=60, help="seconds to run")
    args = parser.parse_args()
    profile(args.source, args.duration)


if __name__ == "__main__":
    main()
