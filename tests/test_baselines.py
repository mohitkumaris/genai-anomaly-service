"""Unit tests for baseline computation."""

import pytest

from anomaly.baselines import (
    StatisticalBaselineCalculator,
    compute_deviation_score,
    compute_z_score,
)


class TestStatisticalBaselineCalculator:
    """Tests for StatisticalBaselineCalculator."""

    @pytest.fixture
    def calculator(self) -> StatisticalBaselineCalculator:
        """Create a calculator instance."""
        return StatisticalBaselineCalculator()

    def test_compute_basic_metrics(self, calculator: StatisticalBaselineCalculator) -> None:
        """Test basic metric computation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        metrics = calculator.compute(values)

        assert metrics.mean == 30.0
        assert metrics.sample_count == 5
        assert metrics.min_value == 10.0
        assert metrics.max_value == 50.0

    def test_compute_std(self, calculator: StatisticalBaselineCalculator) -> None:
        """Test standard deviation computation."""
        # Values with known std
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = calculator.compute(values)

        # Mean = 3, Population stdev should be sqrt(2) ≈ 1.414
        assert abs(metrics.std - 1.4142135623730951) < 0.0001

    def test_compute_percentiles(self, calculator: StatisticalBaselineCalculator) -> None:
        """Test percentile computation."""
        values = list(range(1, 101))  # 1 to 100
        metrics = calculator.compute(values)

        # With 100 values, percentiles should be close to their value
        assert abs(metrics.p50 - 50) < 2
        assert abs(metrics.p90 - 90) < 2
        assert abs(metrics.p99 - 99) < 2

    def test_compute_requires_min_samples(
        self, calculator: StatisticalBaselineCalculator
    ) -> None:
        """Test that computation fails with insufficient samples."""
        with pytest.raises(ValueError, match="Insufficient data"):
            calculator.compute([1.0])  # Only 1 sample

    def test_compute_deterministic(self, calculator: StatisticalBaselineCalculator) -> None:
        """Test that computation is deterministic."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

        metrics1 = calculator.compute(values)
        metrics2 = calculator.compute(values)

        assert metrics1.mean == metrics2.mean
        assert metrics1.std == metrics2.std
        assert metrics1.p50 == metrics2.p50
        assert metrics1.p90 == metrics2.p90
        assert metrics1.p99 == metrics2.p99

    def test_create_snapshot(self, calculator: StatisticalBaselineCalculator) -> None:
        """Test baseline snapshot creation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        snapshot = calculator.create_snapshot(
            metric_type="cost",
            values=values,
            time_window_hours=24,
            source_description="Test data",
        )

        assert snapshot.metric_type == "cost"
        assert snapshot.metrics.mean == 30.0
        assert snapshot.time_window_hours == 24
        assert snapshot.algorithm_version == calculator.algorithm_version


class TestZScoreComputation:
    """Tests for z-score helper functions."""

    def test_z_score_basic(self) -> None:
        """Test basic z-score computation."""
        z = compute_z_score(value=120.0, mean=100.0, std=10.0)
        assert z == 2.0

    def test_z_score_negative(self) -> None:
        """Test negative z-score."""
        z = compute_z_score(value=80.0, mean=100.0, std=10.0)
        assert z == -2.0

    def test_z_score_zero_std(self) -> None:
        """Test z-score with zero standard deviation."""
        z_same = compute_z_score(value=100.0, mean=100.0, std=0.0)
        assert z_same == 0.0

        z_diff = compute_z_score(value=110.0, mean=100.0, std=0.0)
        assert z_diff == float("inf")


class TestDeviationScore:
    """Tests for deviation score computation."""

    def test_deviation_score_basic(self) -> None:
        """Test basic deviation score computation."""
        score = compute_deviation_score(observed=120.0, expected=100.0, baseline_std=10.0)
        assert score == 2.0

    def test_deviation_score_negative_deviation(self) -> None:
        """Test deviation score is always positive."""
        score = compute_deviation_score(observed=80.0, expected=100.0, baseline_std=10.0)
        assert score == 2.0  # Absolute value

    def test_deviation_score_zero_std(self) -> None:
        """Test deviation score with zero baseline std."""
        score_same = compute_deviation_score(observed=100.0, expected=100.0, baseline_std=0.0)
        assert score_same == 0.0

        score_diff = compute_deviation_score(observed=110.0, expected=100.0, baseline_std=0.0)
        assert score_diff == float("inf")
