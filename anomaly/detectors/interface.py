"""Interface for anomaly detectors."""

from abc import ABC, abstractmethod
from typing import Sequence

from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class AnomalyDetector(ABC):
    """Abstract interface for anomaly detection algorithms.
    
    All implementations MUST be:
    - Deterministic: Same inputs produce identical outputs
    - Pure: No side effects, no external state
    - Versioned: Algorithm version must be tracked
    - Explainable: Results include deviation context
    
    CRITICAL: Detectors produce advisory metadata only.
    They do NOT trigger any actions or enforcement.
    """

    @property
    @abstractmethod
    def anomaly_type(self) -> AnomalyType:
        """Get the type of anomaly this detector identifies."""
        ...

    @property
    @abstractmethod
    def algorithm_version(self) -> str:
        """Get the algorithm version for this detector."""
        ...

    @abstractmethod
    def detect(
        self,
        pair: ComparisonPair,
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> AnomalyRecord | None:
        """Detect an anomaly in a single comparison pair.
        
        Args:
            pair: Actual vs predicted comparison
            baseline: Historical baseline metrics
            time_window: Time window for this analysis
            
        Returns:
            AnomalyRecord if anomaly detected, None otherwise
        """
        ...

    @abstractmethod
    def detect_batch(
        self,
        pairs: Sequence[ComparisonPair],
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Detect anomalies in a batch of comparison pairs.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baseline: Historical baseline metrics
            time_window: Time window for this analysis
            
        Returns:
            List of detected anomalies (may be empty)
        """
        ...

    def _compute_confidence(
        self,
        deviation_score: float,
        baseline: BaselineMetrics,
    ) -> float:
        """Compute confidence level for an anomaly detection.
        
        Higher deviation scores and more baseline samples increase confidence.
        
        Args:
            deviation_score: The computed deviation score
            baseline: Baseline metrics (sample_count affects confidence)
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        # Base confidence from sample count (more samples = higher confidence)
        sample_factor = min(1.0, baseline.sample_count / 100)

        # Adjust based on deviation score magnitude
        # Very high deviations are more certainly anomalies
        deviation_factor = min(1.0, deviation_score / 5.0)

        # Combined confidence
        confidence = 0.5 * sample_factor + 0.5 * deviation_factor
        return min(1.0, max(0.0, confidence))
