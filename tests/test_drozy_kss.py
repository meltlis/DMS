from __future__ import annotations

from pathlib import Path

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "DROZY" / "DROZY"


class TestDROZYKSS:
    """Validate DROZY KSS (Karolinska Sleepiness Scale) data integrity."""

    @pytest.fixture(scope="class")
    def kss_matrix(self) -> list[list[int]]:
        kss_path = DATASET_ROOT / "KSS.txt"
        if not kss_path.exists():
            pytest.skip("KSS.txt not found")
        lines = kss_path.read_text().strip().splitlines()
        matrix = [list(map(int, line.split())) for line in lines]
        return matrix

    def test_kss_has_14_subjects(self, kss_matrix: list[list[int]]) -> None:
        assert len(kss_matrix) == 14

    def test_kss_has_3_tests_per_subject(self, kss_matrix: list[list[int]]) -> None:
        for row in kss_matrix:
            assert len(row) == 3

    def test_kss_scores_in_valid_range(self, kss_matrix: list[list[int]]) -> None:
        for row in kss_matrix:
            for score in row:
                assert 1 <= score <= 9 or score == 0, f"invalid KSS score {score}"

    def test_kss_subject_7_test_1_is_missing(self, kss_matrix: list[list[int]]) -> None:
        """Documented missing test should be marked 0."""
        assert kss_matrix[6][0] == 0

    def test_kss_distribution_sanity(self, kss_matrix: list[list[int]]) -> None:
        flat = [s for row in kss_matrix for s in row if s != 0]
        assert len(flat) > 0
        mean_score = sum(flat) / len(flat)
        # DROZY is a drowsiness dataset; mean should be > 3
        assert mean_score > 3.0, f"unexpected low mean KSS {mean_score}"

    def test_high_kss_exists_for_fatigue_testing(self, kss_matrix: list[list[int]]) -> None:
        flat = [s for row in kss_matrix for s in row if s != 0]
        assert max(flat) >= 7, "expected at least one high KSS score for fatigue validation"
