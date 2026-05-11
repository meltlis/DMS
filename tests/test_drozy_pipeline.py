from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest

from src.pipeline import DMSPipeline, load_yaml

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "DROZY" / "DROZY"
VIDEOS_DIR = DATASET_ROOT / "videos_i8"


def find_test_video() -> Path | None:
    candidates = sorted(VIDEOS_DIR.glob("*.mp4"))
    return candidates[0] if candidates else None


class TestDROZYPipeline:
    """End-to-end pipeline tests using real DROZY drowsiness videos."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> DMSPipeline:
        root = Path(__file__).resolve().parents[1]
        thresholds = load_yaml(root / "configs" / "thresholds.yaml")
        runtime = load_yaml(root / "configs" / "runtime.yaml")
        return DMSPipeline(thresholds=thresholds, runtime=runtime)

    def test_drozy_video_opens(self) -> None:
        video = find_test_video()
        if video is None:
            pytest.skip("No DROZY video found")
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # DROZY videos are 512x424 @ 30fps
        assert w == 512.0
        assert h == 424.0
        assert fps == 30.0
        cap.release()

    def test_pipeline_runs_on_drozy_without_crash(self, pipeline: DMSPipeline) -> None:
        video = find_test_video()
        if video is None:
            pytest.skip("No DROZY video found")
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        max_frames = 90
        processed = 0
        ts_base = time.time()
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            ts = ts_base + processed / 30.0
            state = pipeline.process(frame, ts)
            assert isinstance(state, dict)
            assert state["fatigue"] in {"NORMAL", "WARNING", "ALERT"}
            assert state["distraction"] in {"NORMAL", "GAZE_AWAY", "PHONE", "SMOKE"}
            assert state["danger"] in {"NORMAL", "NO_SEATBELT"}
            processed += 1
        cap.release()
        assert processed > 0

    def test_drozy_long_run_no_memory_leak_proxy(self, pipeline: DMSPipeline) -> None:
        """Run 3-second window * 2 to verify deque bounded growth."""
        video = find_test_video()
        if video is None:
            pytest.skip("No DROZY video found")
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        max_frames = 180  # 6 seconds @ 30fps
        processed = 0
        ts_base = time.time()
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            ts = ts_base + processed / 30.0
            pipeline.process(frame, ts)
            processed += 1
        cap.release()
        # If temporal window leaked, this would be very slow or crash.
        assert processed == max_frames
