from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import DMSPipeline, load_yaml, run_pipeline, synthetic_frames


class TestSyntheticFrames:
    def test_generator_yields_frames(self) -> None:
        gen = synthetic_frames(640, 480)
        f1 = next(gen)
        f2 = next(gen)
        assert f1.shape == (480, 640, 3)
        assert f2.shape == (480, 640, 3)
        assert not np.array_equal(f1, f2)


class TestDMSPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self) -> DMSPipeline:
        root = Path(__file__).resolve().parents[1]
        thresholds = load_yaml(root / "configs" / "thresholds.yaml")
        runtime = load_yaml(root / "configs" / "runtime.yaml")
        return DMSPipeline(thresholds=thresholds, runtime=runtime)

    def test_process_returns_full_state_dict(self, pipeline: DMSPipeline) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.time()
        state = pipeline.process(frame, ts)
        assert set(state.keys()) == {"fatigue", "distraction", "danger", "alerts", "track_id", "debug"}
        debug = state["debug"]
        assert "ear_left" in debug
        assert "ear_right" in debug
        assert "mar" in debug
        assert "pitch" in debug
        assert "yaw" in debug
        assert "roll" in debug
        assert "perclos" in debug
        assert "nod_freq" in debug
        assert "yawn_count" in debug
        assert "gaze_away_duration" in debug

    def test_process_with_empty_frame(self, pipeline: DMSPipeline) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.time()
        state = pipeline.process(frame, ts)
        assert state["track_id"] == 1
        assert state["fatigue"] in {"NORMAL", "WARNING", "ALERT"}

    def test_track_switch_resets_temporal(self, pipeline: DMSPipeline) -> None:
        """Simulate track switch by manipulating tracker state directly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.time()
        # First call establishes track 1
        pipeline.process(frame, ts)
        # Manually switch track id
        pipeline.tracker._track_id = 2
        state = pipeline.process(frame, ts + 0.1)
        # FSM should detect switch and temporal should reset
        assert pipeline.fsm.current_track_id == 2
        assert pipeline.fsm.reset_count == 1
        # After reset perclos should be 0 or based on single frame
        assert state["debug"]["perclos"] >= 0.0

    def test_seatbelt_alert_after_grace_period(self, pipeline: DMSPipeline) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.time()
        # First frame sets start_ts; within grace period no alert
        state = pipeline.process(frame, ts)
        assert "NO_SEATBELT" not in state["alerts"]
        # After grace period, stub detector never returns seatbelt
        state = pipeline.process(frame, ts + 10.0)
        assert "NO_SEATBELT" in state["alerts"]
        assert state["danger"] == "NO_SEATBELT"

    def test_run_pipeline_with_synthetic(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.INFO):
            run_pipeline(source="synthetic", max_frames=60, show=False)
        assert any("processed=" in rec.message for rec in caplog.records)
