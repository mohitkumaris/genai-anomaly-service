"""Baseline models for statistical computations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BaselineMetrics:
    """Statistical baseline metrics for a given metric type.
    
    These metrics are computed from historical data and used as
    reference points for anomaly detection.
    
    All metrics are computed deterministically from the same input data.
    """

    mean: float
    std: float  # Standard deviation
    p50: float  # Median (50th percentile)
    p90: float  # 90th percentile
    p99: float  # 99th percentile
    min_value: float
    max_value: float
    sample_count: int

    def __post_init__(self) -> None:
        """Validate baseline metrics constraints."""
        if self.std < 0:
            raise ValueError("Standard deviation cannot be negative")
        if self.sample_count < 0:
            raise ValueError("Sample count cannot be negative")
        if self.min_value > self.max_value:
            raise ValueError("Min value cannot be greater than max value")

    @property
    def coefficient_of_variation(self) -> float:
        """Calculate the coefficient of variation (CV)."""
        if self.mean == 0:
            return float("inf") if self.std > 0 else 0.0
        return self.std / abs(self.mean)

    @property
    def interquartile_range(self) -> float:
        """Estimate IQR from available percentiles (p90 - p50 as approximation)."""
        return self.p90 - self.p50

    def z_score(self, value: float) -> float:
        """Calculate z-score for a given value relative to this baseline."""
        if self.std == 0:
            return 0.0 if value == self.mean else float("inf")
        return (value - self.mean) / self.std

    def is_above_percentile(self, value: float, percentile: float) -> bool:
        """Check if a value is above a given percentile threshold."""
        if percentile >= 99:
            return value > self.p99
        elif percentile >= 90:
            return value > self.p90
        elif percentile >= 50:
            return value > self.p50
        return False


@dataclass(frozen=True)
class BaselineSnapshot:
    """A snapshot of baseline metrics at a specific point in time.
    
    Baselines are versioned and timestamped for audit purposes.
    """

    metric_type: str  # e.g., "cost", "latency", "quality"
    metrics: BaselineMetrics
    computed_at: datetime
    algorithm_version: str
    time_window_hours: int  # Hours of historical data used
    source_description: str = ""  # Description of data source
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type,
            "metrics": {
                "mean": self.metrics.mean,
                "std": self.metrics.std,
                "p50": self.metrics.p50,
                "p90": self.metrics.p90,
                "p99": self.metrics.p99,
                "min_value": self.metrics.min_value,
                "max_value": self.metrics.max_value,
                "sample_count": self.metrics.sample_count,
            },
            "computed_at": self.computed_at.isoformat(),
            "algorithm_version": self.algorithm_version,
            "time_window_hours": self.time_window_hours,
            "source_description": self.source_description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineSnapshot":
        """Create a BaselineSnapshot from a dictionary."""
        metrics_data = data["metrics"]
        return cls(
            metric_type=data["metric_type"],
            metrics=BaselineMetrics(
                mean=metrics_data["mean"],
                std=metrics_data["std"],
                p50=metrics_data["p50"],
                p90=metrics_data["p90"],
                p99=metrics_data["p99"],
                min_value=metrics_data["min_value"],
                max_value=metrics_data["max_value"],
                sample_count=metrics_data["sample_count"],
            ),
            computed_at=datetime.fromisoformat(data["computed_at"]),
            algorithm_version=data["algorithm_version"],
            time_window_hours=data["time_window_hours"],
            source_description=data.get("source_description", ""),
            metadata=data.get("metadata", {}),
        )
