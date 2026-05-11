from __future__ import annotations

import pytest

from src.decision.danger import DangerDetector, iou


class TestIoU:
    def test_identical_boxes(self) -> None:
        assert iou((10, 10, 50, 50), (10, 10, 50, 50)) == 1.0

    def test_no_overlap(self) -> None:
        assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0

    def test_partial_overlap(self) -> None:
        a = (0, 0, 10, 10)
        b = (5, 5, 10, 10)
        inter = 5 * 5
        union = 100 + 100 - 25
        assert iou(a, b) == pytest.approx(inter / union)

    def test_zero_area(self) -> None:
        assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


class TestDangerDetector:
    def test_no_alerts_when_everything_safe(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        grouped = {"face": [(10, 10, 50, 50)], "phone": [], "cigarette": [], "seatbelt": [(0, 0, 10, 10)]}
        alerts = dd.detect(grouped, (10, 10, 50, 50), ts=0.0)
        assert alerts == []

    def test_phone_alert_after_duration(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        face = (10, 10, 50, 50)
        phone = (20, 20, 30, 30)
        grouped = {"face": [face], "phone": [phone], "cigarette": [], "seatbelt": [(0, 0, 10, 10)]}
        # Before duration threshold
        assert "PHONE" not in dd.detect(grouped, face, ts=0.0)
        assert "PHONE" not in dd.detect(grouped, face, ts=1.0)
        # At exactly 2 seconds
        assert "PHONE" in dd.detect(grouped, face, ts=2.0)
        # After threshold still triggers
        assert "PHONE" in dd.detect(grouped, face, ts=3.0)

    def test_phone_resets_when_lost(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        face = (10, 10, 50, 50)
        phone = (20, 20, 30, 30)
        grouped_with = {"face": [face], "phone": [phone], "cigarette": [], "seatbelt": [(0, 0, 10, 10)]}
        grouped_without = {"face": [face], "phone": [], "cigarette": [], "seatbelt": [(0, 0, 10, 10)]}
        dd.detect(grouped_with, face, ts=0.0)
        dd.detect(grouped_without, face, ts=1.0)
        # Timer reset at ts=1.0, need another 2 seconds from re-detection
        assert "PHONE" not in dd.detect(grouped_with, face, ts=2.5)
        assert "PHONE" in dd.detect(grouped_with, face, ts=4.5)

    def test_smoke_alert_immediate(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        grouped = {"face": [], "phone": [], "cigarette": [(0, 0, 10, 10)], "seatbelt": [(0, 0, 10, 10)]}
        alerts = dd.detect(grouped, None, ts=0.0)
        assert "SMOKE" in alerts

    def test_no_seatbelt_during_grace_period(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        grouped = {"face": [], "phone": [], "cigarette": [], "seatbelt": []}
        alerts = dd.detect(grouped, None, ts=5.0)
        assert "NO_SEATBELT" not in alerts

    def test_no_seatbelt_after_grace_period(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        grouped = {"face": [], "phone": [], "cigarette": [], "seatbelt": []}
        # First call sets start_ts; grace period counts from there
        dd.detect(grouped, None, ts=0.0)
        alerts = dd.detect(grouped, None, ts=10.0)
        assert "NO_SEATBELT" in alerts
        # Continues to alert
        alerts = dd.detect(grouped, None, ts=15.0)
        assert "NO_SEATBELT" in alerts

    def test_seatbelt_seen_never_alerts_again(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        grouped_with = {"face": [], "phone": [], "cigarette": [], "seatbelt": [(0, 0, 10, 10)]}
        grouped_without = {"face": [], "phone": [], "cigarette": [], "seatbelt": []}
        # Seatbelt seen during grace period
        dd.detect(grouped_with, None, ts=5.0)
        # After grace period, even if missing, no alert
        alerts = dd.detect(grouped_without, None, ts=15.0)
        assert "NO_SEATBELT" not in alerts

    def test_multiple_alerts(self) -> None:
        dd = DangerDetector(phone_iou_threshold=0.10, phone_duration_seconds=2.0, seatbelt_grace_seconds=10.0)
        face = (10, 10, 50, 50)
        phone = (20, 20, 30, 30)
        grouped = {"face": [face], "phone": [phone], "cigarette": [(0, 0, 10, 10)], "seatbelt": []}
        # First call at ts=0.0 sets start_ts; warm-up phone timer
        dd.detect(grouped, face, ts=0.0)
        # At ts=10.0 phone timer (10.0 - 0.0 = 10.0 >= 2.0) and seatbelt grace expired
        alerts = dd.detect(grouped, face, ts=10.0)
        assert "PHONE" in alerts
        assert "SMOKE" in alerts
        assert "NO_SEATBELT" in alerts
