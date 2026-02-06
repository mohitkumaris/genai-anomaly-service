"""Unit tests for anomaly detectors."""

from datetime import datetime, timedelta

import pytest

from anomaly.detectors import (
    CostAnomalyDetector,
    LatencyAnomalyDetector,
    PolicyAnomalyDetector,
    QualityAnomalyDetector,
)
from anomaly.models import (
    ActualOutcome,
    BaselineMetrics,
    ComparisonPair,
    PredictionRecord,
    TimeWindow,
)


@pytest.fixture
def time_window() -> TimeWindow:
    """Create a test time window."""
    now = datetime.utcnow()
    return TimeWindow(start=now - timedelta(hours=24), end=now)


@pytest.fixture
def baseline() -> BaselineMetrics:
    """Create a test baseline with realistic values."""
    return BaselineMetrics(
        mean=100.0,
        std=20.0,
        p50=95.0,
        p90=130.0,
        p99=160.0,
        min_value=50.0,
        max_value=200.0,
        sample_count=100,
    )


@pytest.fixture
def quality_baseline() -> BaselineMetrics:
    """Create a test baseline for quality scores (0-1 range)."""
    return BaselineMetrics(
        mean=0.85,
        std=0.1,
        p50=0.87,
        p90=0.95,
        p99=0.98,
        min_value=0.5,
        max_value=1.0,
        sample_count=100,
    )


def create_comparison_pair(
    trace_id: str,
    actual_cost: float = 0.0,
    predicted_cost: float | None = None,
    actual_quality: float | None = None,
    predicted_quality: float | None = None,
    actual_latency: float = 0.0,
    predicted_latency: float | None = None,
    actual_policy_passed: bool | None = None,
    predicted_policy_pass: bool | None = None,
) -> ComparisonPair:
    """Helper to create a comparison pair for testing."""
    now = datetime.utcnow()
    actual = ActualOutcome(
        trace_id=trace_id,
        timestamp=now,
        estimated_cost_usd=actual_cost,
        quality_score=actual_quality,
        latency_ms=actual_latency,
        policy_passed=actual_policy_passed,
    )
    predicted = PredictionRecord(
        prediction_id=f"pred_{trace_id}",
        trace_id=trace_id,
        timestamp=now,
        predicted_cost_usd=predicted_cost,
        predicted_quality_score=predicted_quality,
        predicted_latency_ms=predicted_latency,
        predicted_policy_pass=predicted_policy_pass,
        confidence=0.9,
    )
    return ComparisonPair(actual=actual, predicted=predicted)


class TestCostAnomalyDetector:
    """Tests for CostAnomalyDetector."""

    def test_no_anomaly_when_within_threshold(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that no anomaly is detected when deviation is within threshold."""
        detector = CostAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_1",
            actual_cost=105.0,  # 5 above predicted, but within 2 std
            predicted_cost=100.0,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is None

    def test_anomaly_when_exceeds_threshold(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that anomaly is detected when deviation exceeds threshold."""
        detector = CostAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_2",
            actual_cost=200.0,  # 100 above predicted, well beyond 2 std (40)
            predicted_cost=100.0,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is not None
        assert result.anomaly_type.value == "cost"
        assert result.observed_value == 200.0
        assert result.expected_value == 100.0
        assert result.deviation_score > 2.0

    def test_deterministic_output(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that detector produces identical output for identical input."""
        detector = CostAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_deterministic",
            actual_cost=200.0,
            predicted_cost=100.0,
        )

        result1 = detector.detect(pair, baseline, time_window)
        result2 = detector.detect(pair, baseline, time_window)

        assert result1 is not None
        assert result2 is not None
        assert result1.deviation_score == result2.deviation_score
        assert result1.confidence == result2.confidence

    def test_no_data_returns_none(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that missing cost data returns None."""
        detector = CostAnomalyDetector()
        pair = create_comparison_pair(
            trace_id="test_no_data",
            actual_cost=100.0,
            predicted_cost=None,  # No prediction
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is None


class TestQualityAnomalyDetector:
    """Tests for QualityAnomalyDetector."""

    def test_no_anomaly_when_quality_acceptable(
        self, time_window: TimeWindow, quality_baseline: BaselineMetrics
    ) -> None:
        """Test that no anomaly is detected when quality is acceptable."""
        detector = QualityAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_quality_ok",
            actual_quality=0.85,
            predicted_quality=0.87,
        )

        result = detector.detect(pair, quality_baseline, time_window)
        assert result is None

    def test_anomaly_when_quality_low(
        self, time_window: TimeWindow, quality_baseline: BaselineMetrics
    ) -> None:
        """Test that anomaly is detected when quality is significantly low."""
        detector = QualityAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_low_quality",
            actual_quality=0.5,  # Much lower than predicted
            predicted_quality=0.87,
        )

        result = detector.detect(pair, quality_baseline, time_window)
        assert result is not None
        assert result.anomaly_type.value == "quality"


class TestLatencyAnomalyDetector:
    """Tests for LatencyAnomalyDetector."""

    def test_no_anomaly_when_latency_normal(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that no anomaly is detected when latency is normal."""
        detector = LatencyAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_latency_ok",
            actual_latency=110.0,
            predicted_latency=100.0,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is None

    def test_anomaly_when_latency_high(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that anomaly is detected when latency exceeds P99."""
        detector = LatencyAnomalyDetector(z_score_threshold=2.0, p99_multiplier=1.2)
        pair = create_comparison_pair(
            trace_id="test_high_latency",
            actual_latency=250.0,  # Well above P99 (160) * 1.2 = 192
            predicted_latency=100.0,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is not None
        assert result.anomaly_type.value == "latency"
        assert result.metadata.get("exceeds_p99") is True

    def test_no_anomaly_for_low_latency(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that low latency (good) does not trigger anomaly."""
        detector = LatencyAnomalyDetector(z_score_threshold=2.0)
        pair = create_comparison_pair(
            trace_id="test_low_latency",
            actual_latency=50.0,  # Lower than predicted (good!)
            predicted_latency=100.0,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is None  # Low latency is not an anomaly


class TestPolicyAnomalyDetector:
    """Tests for PolicyAnomalyDetector."""

    def test_no_anomaly_when_outcome_matches(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that no anomaly is detected when policy outcome matches prediction."""
        detector = PolicyAnomalyDetector()
        pair = create_comparison_pair(
            trace_id="test_policy_match",
            actual_policy_passed=True,
            predicted_policy_pass=True,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is None

    def test_anomaly_when_unexpected_fail(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that anomaly is detected for unexpected policy failure."""
        detector = PolicyAnomalyDetector()
        pair = create_comparison_pair(
            trace_id="test_unexpected_fail",
            actual_policy_passed=False,
            predicted_policy_pass=True,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is not None
        assert result.anomaly_type.value == "policy"
        assert result.metadata.get("is_unexpected_fail") is True

    def test_anomaly_when_unexpected_pass(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that anomaly is detected for unexpected policy pass."""
        detector = PolicyAnomalyDetector()
        pair = create_comparison_pair(
            trace_id="test_unexpected_pass",
            actual_policy_passed=True,
            predicted_policy_pass=False,
        )

        result = detector.detect(pair, baseline, time_window)
        assert result is not None
        assert result.anomaly_type.value == "policy"
        assert result.metadata.get("is_unexpected_pass") is True


class TestDetectorDeterminism:
    """Tests for detector determinism - critical requirement."""

    def test_all_detectors_deterministic(
        self, time_window: TimeWindow, baseline: BaselineMetrics
    ) -> None:
        """Test that all detectors produce identical output for same input."""
        detectors = [
            (CostAnomalyDetector(), create_comparison_pair(
                "det_cost", actual_cost=200.0, predicted_cost=100.0
            )),
            (LatencyAnomalyDetector(), create_comparison_pair(
                "det_latency", actual_latency=250.0, predicted_latency=100.0
            )),
        ]

        for detector, pair in detectors:
            results = [detector.detect(pair, baseline, time_window) for _ in range(5)]

            # All results should be identical (or all None)
            if results[0] is not None:
                for result in results[1:]:
                    assert result is not None
                    assert result.deviation_score == results[0].deviation_score
                    assert result.confidence == results[0].confidence
