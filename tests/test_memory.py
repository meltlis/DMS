from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import DMSPipeline, load_yaml


class TestMemoryLeakProxy:
    """Proxy tests for '3-second window running 1 hour without memory leak'.

    We cannot run a full hour in CI, so we verify:
    1. deque maxlen is respected (bounded growth).
    2. processing 10x window frames does not grow memory.
    """

    @pytest.fixture(scope="class")
    def pipeline(self) -> DMSPipeline:
        root = Path(__file__).resolve().parents[1]
        thresholds = load_yaml(root / "configs" / "thresholds.yaml")
        runtime = load_yaml(root / "configs" / "runtime.yaml")
        return DMSPipeline(thresholds=thresholds, runtime=runtime)

    def test_temporal_window_bounded(self, pipeline: DMSPipeline) -> None:
        """TemporalAggregator window should never exceed fps * window_seconds."""
        fps = 30
        window_seconds = 3.0
        expected_maxlen = int(fps * window_seconds)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(expected_maxlen * 3):
            pipeline.process(frame, ts=i / fps)

        # Check internal deque length
        for win in pipeline.temporal._windows.values():
            assert len(win) <= expected_maxlen

    def test_memory_stable_over_many_frames(self, pipeline: DMSPipeline) -> None:
        """Process many frames and ensure allocated memory does not grow unboundedly."""
        tracemalloc.start()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm-up to stabilise internal caches
        for i in range(300):
            pipeline.process(frame, ts=i / 30.0)

        _, peak_before = tracemalloc.get_traced_memory()

        # Main batch
        for i in range(300, 300 + 3000):
            pipeline.process(frame, ts=i / 30.0)

        current, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Allow modest growth (e.g. Python allocator overhead), but reject unbounded leak
        growth = peak_after - peak_before
        # 10 MB threshold is generous; a true leak would be GBs after 3k frames.
        assert growth < 10 * 1024 * 1024, f"memory grew by {growth / 1024 / 1024:.2f} MB"
