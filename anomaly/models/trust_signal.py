"""Trust signal models for aggregated anomaly assessments."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from anomaly.models.anomaly_record import AnomalyType


class TrustLevel(str, Enum):
    """Trust level categories based on anomaly analysis.
    
    These levels are advisory only and do NOT trigger any actions.
    """

    HIGH = "high"  # No significant anomalies detected
    MEDIUM = "medium"  # Some anomalies detected, warrants attention
    LOW = "low"  # Multiple or severe anomalies detected
    UNKNOWN = "unknown"  # Insufficient data for assessment


@dataclass(frozen=True)
class AnomalyCount:
    """Count of anomalies by type."""

    cost: int = 0
    quality: int = 0
    latency: int = 0
    policy: int = 0

    @property
    def total(self) -> int:
        """Get the total number of anomalies."""
        return self.cost + self.quality + self.latency + self.policy

    def get_count(self, anomaly_type: AnomalyType) -> int:
        """Get count for a specific anomaly type."""
        mapping = {
            AnomalyType.COST: self.cost,
            AnomalyType.QUALITY: self.quality,
            AnomalyType.LATENCY: self.latency,
            AnomalyType.POLICY: self.policy,
        }
        return mapping[anomaly_type]


@dataclass(frozen=True)
class TrustSignal:
    """Aggregated trust signal based on anomaly analysis.
    
    Trust signals summarize the anomaly state for a given time window
    and entity (e.g., model, endpoint, user). They are purely informational
    and do NOT influence execution behavior.
    
    IMPORTANT: Trust signals are advisory metadata only.
    """

    trust_level: TrustLevel
    anomaly_counts: AnomalyCount
    confidence: float  # Overall confidence in the trust assessment
    computed_at: datetime
    algorithm_version: str

    # Scope of the trust signal
    entity_type: str = ""  # e.g., "model", "endpoint", "user"
    entity_id: str = ""

    # Additional context
    time_window_hours: int = 24
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trust signal constraints."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return self.anomaly_counts.total > 0

    @property
    def severity_score(self) -> float:
        """Calculate a normalized severity score (0.0 to 1.0)."""
        if self.trust_level == TrustLevel.HIGH:
            return 0.0
        elif self.trust_level == TrustLevel.MEDIUM:
            return 0.5
        elif self.trust_level == TrustLevel.LOW:
            return 1.0
        return 0.5  # Unknown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trust_level": self.trust_level.value,
            "anomaly_counts": {
                "cost": self.anomaly_counts.cost,
                "quality": self.anomaly_counts.quality,
                "latency": self.anomaly_counts.latency,
                "policy": self.anomaly_counts.policy,
                "total": self.anomaly_counts.total,
            },
            "confidence": self.confidence,
            "computed_at": self.computed_at.isoformat(),
            "algorithm_version": self.algorithm_version,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "time_window_hours": self.time_window_hours,
            "has_anomalies": self.has_anomalies,
            "severity_score": self.severity_score,
            "metadata": self.metadata,
        }


def compute_trust_level(
    anomaly_count: AnomalyCount,
    high_threshold: int = 0,
    medium_threshold: int = 3,
) -> TrustLevel:
    """Compute trust level based on anomaly counts.
    
    Pure function for deterministic trust level computation.
    
    Args:
        anomaly_count: Count of anomalies by type
        high_threshold: Max anomalies for HIGH trust (default: 0)
        medium_threshold: Max anomalies for MEDIUM trust (default: 3)
    
    Returns:
        Computed trust level
    """
    total = anomaly_count.total
    if total <= high_threshold:
        return TrustLevel.HIGH
    elif total <= medium_threshold:
        return TrustLevel.MEDIUM
    else:
        return TrustLevel.LOW
