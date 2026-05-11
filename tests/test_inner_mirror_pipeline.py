from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import DMSPipeline, load_yaml

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "inner_mirror"


def find_test_video() -> Path | None:
    candidates = list(DATASET_ROOT.rglob("*.mp4"))
    return candidates[0] if candidates else None


class TestInnerMirrorPipeline:
    """End-to-end pipeline tests using real inner_mirror videos."""

    @pytest.fixture(scope="class")
    def pipeline(self) -> DMSPipeline:
        root = Path(__file__).resolve().parents[1]
        thresholds = load_yaml(root / "configs" / "thresholds.yaml")
        runtime = load_yaml(root / "configs" / "runtime.yaml")
        return DMSPipeline(thresholds=thresholds, runtime=runtime)

    def test_video_opens_and_has_frames(self) -> None:
        video = find_test_video()
        if video is None:
            pytest.skip("No inner_mirror video found")
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame is not None
        assert frame.ndim == 3
        cap.release()

    def test_pipeline_runs_on_inner_mirror_without_crash(self, pipeline: DMSPipeline) -> None:
        video = find_test_video()
        if video is None:
            pytest.skip("No inner_mirror video found")
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
            assert "fatigue" in state
            assert "distraction" in state
            assert "danger" in state
            assert "alerts" in state
            assert "track_id" in state
            assert "debug" in state
            debug = state["debug"]
            assert "ear_left" in debug
            processed += 1
        cap.release()
        assert processed > 0, "pipeline should process at least one frame"

    def test_pipeline_fps_on_inner_mirror(self, pipeline: DMSPipeline) -> None:
        video = find_test_video()
        if video is None:
            pytest.skip("No inner_mirror video found")
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        max_frames = 300
        processed = 0
        ts_base = time.time()
        start = time.time()
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            ts = ts_base + processed / 30.0
            pipeline.process(frame, ts)
            processed += 1
        elapsed = time.time() - start
        cap.release()
        fps = processed / max(elapsed, 1e-6)
        # Stub pipeline should easily exceed 30 FPS; this validates overhead.
        assert fps > 10.0, f"pipeline FPS too low: {fps:.2f}"
