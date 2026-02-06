"""Input data models for anomaly detection.

These models represent the data ingested from external systems:
- LLMOps: Historical actual outcomes
- ML Service: Historical predictions

This service has READ-ONLY access to this data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ActualOutcome:
    """Represents an actual outcome record from LLMOps.
    
    This is the observed, real-world result of a GenAI operation.
    READ-ONLY: This service does not modify these records.
    """

    trace_id: str
    timestamp: datetime

    # Cost metrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Quality metrics
    quality_score: float | None = None  # 0.0 to 1.0 if available

    # Latency metrics
    latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None

    # Policy metrics
    policy_passed: bool | None = None
    policy_violations: list[str] = field(default_factory=list)

    # Context
    model_id: str = ""
    endpoint_id: str = ""
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRecord:
    """Represents a prediction record from the ML service.
    
    This is what the system predicted would happen for a given operation.
    READ-ONLY: This service does not modify these records.
    """

    prediction_id: str
    trace_id: str  # Links to ActualOutcome
    timestamp: datetime

    # Predicted cost
    predicted_tokens: int | None = None
    predicted_cost_usd: float | None = None

    # Predicted quality
    predicted_quality_score: float | None = None

    # Predicted latency
    predicted_latency_ms: float | None = None

    # Predicted policy outcome
    predicted_policy_pass: bool | None = None

    # Prediction metadata
    model_version: str = ""
    confidence: float = 0.0  # Confidence in the prediction
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonPair:
    """A paired actual vs predicted record for comparison.
    
    This is the primary input for anomaly detection.
    """

    actual: ActualOutcome
    predicted: PredictionRecord

    def __post_init__(self) -> None:
        """Validate the comparison pair."""
        if self.actual.trace_id != self.predicted.trace_id:
            raise ValueError("Actual and predicted must have matching trace_id")

    @property
    def has_cost_data(self) -> bool:
        """Check if cost comparison is possible."""
        return self.predicted.predicted_cost_usd is not None

    @property
    def has_quality_data(self) -> bool:
        """Check if quality comparison is possible."""
        return (
            self.actual.quality_score is not None
            and self.predicted.predicted_quality_score is not None
        )

    @property
    def has_latency_data(self) -> bool:
        """Check if latency comparison is possible."""
        return self.predicted.predicted_latency_ms is not None

    @property
    def has_policy_data(self) -> bool:
        """Check if policy comparison is possible."""
        return (
            self.actual.policy_passed is not None
            and self.predicted.predicted_policy_pass is not None
        )

    @property
    def cost_deviation(self) -> float | None:
        """Calculate cost deviation if data available."""
        if not self.has_cost_data:
            return None
        return self.actual.estimated_cost_usd - (self.predicted.predicted_cost_usd or 0)

    @property
    def quality_deviation(self) -> float | None:
        """Calculate quality deviation if data available."""
        if not self.has_quality_data:
            return None
        return (self.actual.quality_score or 0) - (self.predicted.predicted_quality_score or 0)

    @property
    def latency_deviation(self) -> float | None:
        """Calculate latency deviation if data available."""
        if not self.has_latency_data:
            return None
        return self.actual.latency_ms - (self.predicted.predicted_latency_ms or 0)

    @property
    def policy_mismatch(self) -> bool | None:
        """Check if policy outcome differs from prediction."""
        if not self.has_policy_data:
            return None
        return self.actual.policy_passed != self.predicted.predicted_policy_pass


@dataclass(frozen=True)
class HistoricalBatch:
    """A batch of historical data for baseline computation or replay analysis.
    
    Used for batch/snapshot-based ingestion as per specification.
    """

    actuals: list[ActualOutcome]
    predictions: list[PredictionRecord]
    batch_id: str
    timestamp: datetime
    source: str = ""  # Description of data source

    def get_comparison_pairs(self) -> list[ComparisonPair]:
        """Create comparison pairs by matching trace_ids."""
        predictions_by_trace = {p.trace_id: p for p in self.predictions}
        pairs = []
        for actual in self.actuals:
            if actual.trace_id in predictions_by_trace:
                pairs.append(ComparisonPair(
                    actual=actual,
                    predicted=predictions_by_trace[actual.trace_id],
                ))
        return pairs
