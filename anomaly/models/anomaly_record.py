"""Anomaly record models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected.
    
    Each type corresponds to a specific dimension of system behavior:
    - COST: Token/API cost deviations from predictions
    - QUALITY: Response quality score anomalies
    - LATENCY: Response time anomalies
    - POLICY: Policy outcome deviations
    """

    COST = "cost"
    QUALITY = "quality"
    LATENCY = "latency"
    POLICY = "policy"


@dataclass(frozen=True)
class TimeWindow:
    """Represents a time window for anomaly analysis.
    
    Immutable to ensure consistency in anomaly records.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate time window constraints."""
        if self.start >= self.end:
            raise ValueError("Time window start must be before end")

    @property
    def duration_seconds(self) -> float:
        """Get the duration of the time window in seconds."""
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True)
class AnomalyRecord:
    """Immutable anomaly record representing a detected deviation.
    
    This is the core output of the anomaly detection system.
    All fields are required as per the specification.
    
    CRITICAL: Once created, anomaly records are immutable and append-only.
    They are advisory metadata only and do NOT trigger any actions.
    """

    # Required fields per specification
    anomaly_type: AnomalyType
    observed_value: float
    expected_value: float
    deviation_score: float
    confidence: float
    algorithm_version: str
    time_window: TimeWindow

    # Record identification and metadata
    record_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional context for explainability
    metric_name: str = ""
    source_id: str = ""  # Reference to source data (trace_id, prediction_id, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate anomaly record constraints."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.algorithm_version:
            raise ValueError("Algorithm version is required")

    @property
    def deviation_percentage(self) -> float:
        """Calculate the percentage deviation from expected value."""
        if self.expected_value == 0:
            return float("inf") if self.observed_value != 0 else 0.0
        return abs(self.observed_value - self.expected_value) / abs(self.expected_value) * 100

    @property
    def is_positive_deviation(self) -> bool:
        """Check if the deviation is positive (observed > expected)."""
        return self.observed_value > self.expected_value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": str(self.record_id),
            "anomaly_type": self.anomaly_type.value,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "deviation_score": self.deviation_score,
            "confidence": self.confidence,
            "algorithm_version": self.algorithm_version,
            "time_window": {
                "start": self.time_window.start.isoformat(),
                "end": self.time_window.end.isoformat(),
            },
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnomalyRecord":
        """Create an AnomalyRecord from a dictionary."""
        return cls(
            record_id=UUID(data["record_id"]),
            anomaly_type=AnomalyType(data["anomaly_type"]),
            observed_value=data["observed_value"],
            expected_value=data["expected_value"],
            deviation_score=data["deviation_score"],
            confidence=data["confidence"],
            algorithm_version=data["algorithm_version"],
            time_window=TimeWindow(
                start=datetime.fromisoformat(data["time_window"]["start"]),
                end=datetime.fromisoformat(data["time_window"]["end"]),
            ),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_name=data.get("metric_name", ""),
            source_id=data.get("source_id", ""),
            metadata=data.get("metadata", {}),
        )
