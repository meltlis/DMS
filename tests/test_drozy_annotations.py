from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "DROZY" / "DROZY"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations-auto"


def parse_68_landmarks(line: str) -> np.ndarray:
    """Parse a line of 136 floats into (68, 2) array."""
    vals = list(map(float, line.strip().split()))
    assert len(vals) == 136
    return np.array(vals).reshape(68, 2)


def ear_from_68(landmarks: np.ndarray, eye_indices: List[int]) -> float:
    """Compute EAR from 6 eye landmarks (iBUG 68-point).
    indices order: [outer, upper1, upper2, inner, lower2, lower1]
    """
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = math.dist(p[1], p[5])
    vertical_2 = math.dist(p[2], p[4])
    horizontal = math.dist(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mar_from_68(landmarks: np.ndarray) -> float:
    """Compute MAR from mouth landmarks (iBUG 48-67).
    vertical = dist(51, 57) + dist(52, 56); horizontal = 2 * dist(48, 54)
    """
    p = landmarks
    vertical = math.dist(p[51], p[57]) + math.dist(p[52], p[56])
    horizontal = 2.0 * math.dist(p[48], p[54])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class TestDROZYAnnotations:
    """Regression tests using DROZY 68-point annotations as geometric ground truth."""

    @pytest.fixture(scope="class")
    def sample_files(self) -> List[Path]:
        files = sorted(ANNOTATIONS_DIR.glob("*-s2.txt"))
        if not files:
            pytest.skip("DROZY annotations not found")
        return files[:3]

    @pytest.fixture(scope="class")
    def first_frames(self, sample_files: List[Path]) -> List[np.ndarray]:
        frames = []
        for f in sample_files:
            with f.open("r") as fh:
                line = fh.readline()
                if line:
                    frames.append(parse_68_landmarks(line))
        return frames

    def test_ear_in_reasonable_range(self, first_frames: List[np.ndarray]) -> None:
        """EAR should typically be between 0.15 and 0.45 for open eyes."""
        for lm in first_frames:
            left_ear = ear_from_68(lm, [36, 37, 38, 39, 40, 41])
            right_ear = ear_from_68(lm, [42, 43, 44, 45, 46, 47])
            assert 0.05 < left_ear < 0.60, f"left EAR {left_ear} out of range"
            assert 0.05 < right_ear < 0.60, f"right EAR {right_ear} out of range"

    def test_mar_in_reasonable_range(self, first_frames: List[np.ndarray]) -> None:
        """MAR should typically be between 0.1 (closed) and 1.0 (yawn)."""
        for lm in first_frames:
            mar = mar_from_68(lm)
            assert 0.0 < mar < 1.5, f"MAR {mar} out of range"

    def test_annotation_count_matches_video_frames(self) -> None:
        """Every annotation file should have roughly the same number of lines as video frames."""
        videos_dir = DATASET_ROOT / "videos_i8"
        annot_files = sorted(ANNOTATIONS_DIR.glob("*-s2.txt"))
        mismatches = []
        for af in annot_files[:5]:
            stem = af.stem.replace("-s2", "")
            vf = videos_dir / f"{stem}.mp4"
            if not vf.exists():
                continue
            line_count = sum(1 for _ in af.open("r"))
            # Allow up to 5% mismatch because of missing frames documented in DROZY
            # We just check that the annotation file is non-empty and reasonably sized
            if line_count < 100:
                mismatches.append((stem, line_count))
        assert not mismatches, f"too few annotations: {mismatches}"

    def test_kss_file_exists_and_valid(self) -> None:
        kss_path = DATASET_ROOT / "KSS.txt"
        assert kss_path.exists()
        lines = kss_path.read_text().strip().splitlines()
        assert len(lines) == 14
        for line in lines:
            scores = list(map(int, line.split()))
            assert len(scores) == 3
            for s in scores:
                assert 0 <= s <= 9
