"""Statistical baseline calculator implementation."""

import statistics
from datetime import datetime
from typing import Sequence

from anomaly.baselines.interface import BaselineCalculator
from anomaly.models.baseline import BaselineMetrics, BaselineSnapshot


class StatisticalBaselineCalculator(BaselineCalculator):
    """Computes baseline metrics using standard statistical methods.
    
    This calculator uses simple, well-understood statistical measures:
    - Mean: Arithmetic average
    - Standard deviation: Population standard deviation
    - Percentiles: 50th (median), 90th, and 99th
    
    All computations are deterministic and produce identical results
    for the same input data.
    
    DESIGN: Uses population stdev (not sample) for consistency with
    the anomaly detection algorithms. This is a deliberate choice.
    """

    _ALGORITHM_VERSION = "1.0.0"
    _MIN_SAMPLES = 2  # Minimum samples for meaningful statistics

    @property
    def algorithm_version(self) -> str:
        """Get the algorithm version for this calculator."""
        return self._ALGORITHM_VERSION

    def compute(self, values: Sequence[float]) -> BaselineMetrics:
        """Compute baseline metrics from a sequence of values.
        
        Args:
            values: Sequence of numeric values to compute baseline from
            
        Returns:
            Computed baseline metrics
            
        Raises:
            ValueError: If fewer than MIN_SAMPLES values provided
        """
        values_list = list(values)
        n = len(values_list)

        if n < self._MIN_SAMPLES:
            raise ValueError(
                f"Insufficient data: need at least {self._MIN_SAMPLES} samples, got {n}"
            )

        # Sort once for percentile calculations
        sorted_values = sorted(values_list)

        # Core statistics
        mean = statistics.mean(values_list)
        std = statistics.pstdev(values_list)  # Population stdev for consistency

        # Percentiles
        p50 = self._percentile(sorted_values, 50)
        p90 = self._percentile(sorted_values, 90)
        p99 = self._percentile(sorted_values, 99)

        return BaselineMetrics(
            mean=mean,
            std=std,
            p50=p50,
            p90=p90,
            p99=p99,
            min_value=sorted_values[0],
            max_value=sorted_values[-1],
            sample_count=n,
        )

    def create_snapshot(
        self,
        metric_type: str,
        values: Sequence[float],
        time_window_hours: int,
        source_description: str = "",
    ) -> BaselineSnapshot:
        """Create a complete baseline snapshot.
        
        Args:
            metric_type: Type of metric (e.g., "cost", "latency")
            values: Sequence of numeric values
            time_window_hours: Hours of historical data
            source_description: Description of data source
            
        Returns:
            Complete baseline snapshot with metadata
        """
        metrics = self.compute(values)
        return BaselineSnapshot(
            metric_type=metric_type,
            metrics=metrics,
            computed_at=datetime.utcnow(),
            algorithm_version=self.algorithm_version,
            time_window_hours=time_window_hours,
            source_description=source_description,
        )

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        """Calculate percentile from sorted values using linear interpolation.
        
        Uses the same algorithm as numpy.percentile with 'linear' interpolation.
        This ensures deterministic, reproducible results.
        
        Args:
            sorted_values: Pre-sorted list of values
            percentile: Percentile to compute (0-100)
            
        Returns:
            Computed percentile value
        """
        if not sorted_values:
            raise ValueError("Cannot compute percentile of empty sequence")

        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]

        # Calculate index using linear interpolation
        k = (percentile / 100.0) * (n - 1)
        f = int(k)
        c = f + 1

        if c >= n:
            return sorted_values[-1]

        # Linear interpolation between adjacent values
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score for a value given mean and standard deviation.
    
    Pure function for z-score calculation.
    
    Args:
        value: The value to compute z-score for
        mean: Population mean
        std: Population standard deviation
        
    Returns:
        Z-score (signed, can be negative)
    """
    if std == 0:
        return 0.0 if value == mean else float("inf") if value > mean else float("-inf")
    return (value - mean) / std


def compute_deviation_score(
    observed: float,
    expected: float,
    baseline_std: float,
) -> float:
    """Compute a normalized deviation score.
    
    The deviation score is the absolute z-score of the observed value
    relative to the expected value and baseline standard deviation.
    
    Args:
        observed: Observed (actual) value
        expected: Expected (predicted) value
        baseline_std: Baseline standard deviation for normalization
        
    Returns:
        Non-negative deviation score (higher = more anomalous)
    """
    if baseline_std == 0:
        return 0.0 if observed == expected else float("inf")
    return abs(observed - expected) / baseline_std
