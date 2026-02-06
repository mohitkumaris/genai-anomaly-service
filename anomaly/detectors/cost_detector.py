"""Cost anomaly detector implementation."""

from typing import Sequence

from anomaly.baselines.statistical import compute_deviation_score
from anomaly.detectors.interface import AnomalyDetector
from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class CostAnomalyDetector(AnomalyDetector):
    """Detects anomalies in cost predictions vs actual costs.
    
    Uses z-score based deviation detection with configurable threshold.
    An anomaly is flagged when the deviation between predicted and actual
    cost exceeds the threshold (in standard deviations).
    
    Algorithm:
    1. Extract actual and predicted cost from comparison pair
    2. Compute deviation score (|actual - predicted| / baseline_std)
    3. If deviation > threshold, create anomaly record
    
    This detector is deterministic: identical inputs produce identical outputs.
    """

    _ALGORITHM_VERSION = "1.0.0"

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the cost anomaly detector.
        
        Args:
            z_score_threshold: Number of standard deviations for anomaly
            min_confidence: Minimum confidence to report an anomaly
        """
        self._z_score_threshold = z_score_threshold
        self._min_confidence = min_confidence

    @property
    def anomaly_type(self) -> AnomalyType:
        """Get the type of anomaly this detector identifies."""
        return AnomalyType.COST

    @property
    def algorithm_version(self) -> str:
        """Get the algorithm version for this detector."""
        return self._ALGORITHM_VERSION

    def detect(
        self,
        pair: ComparisonPair,
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> AnomalyRecord | None:
        """Detect a cost anomaly in a single comparison pair.
        
        Args:
            pair: Actual vs predicted comparison
            baseline: Historical baseline metrics for cost
            time_window: Time window for this analysis
            
        Returns:
            AnomalyRecord if cost anomaly detected, None otherwise
        """
        if not pair.has_cost_data:
            return None

        actual_cost = pair.actual.estimated_cost_usd
        predicted_cost = pair.predicted.predicted_cost_usd or 0.0

        # Compute deviation score
        deviation_score = compute_deviation_score(
            observed=actual_cost,
            expected=predicted_cost,
            baseline_std=baseline.std,
        )

        # Check if deviation exceeds threshold
        if deviation_score < self._z_score_threshold:
            return None

        # Compute confidence
        confidence = self._compute_confidence(deviation_score, baseline)
        if confidence < self._min_confidence:
            return None

        return AnomalyRecord(
            anomaly_type=AnomalyType.COST,
            observed_value=actual_cost,
            expected_value=predicted_cost,
            deviation_score=deviation_score,
            confidence=confidence,
            algorithm_version=self.algorithm_version,
            time_window=time_window,
            metric_name="estimated_cost_usd",
            source_id=pair.actual.trace_id,
            metadata={
                "threshold": self._z_score_threshold,
                "baseline_mean": baseline.mean,
                "baseline_std": baseline.std,
            },
        )

    def detect_batch(
        self,
        pairs: Sequence[ComparisonPair],
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Detect cost anomalies in a batch of comparison pairs.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baseline: Historical baseline metrics for cost
            time_window: Time window for this analysis
            
        Returns:
            List of detected cost anomalies
        """
        anomalies = []
        for pair in pairs:
            anomaly = self.detect(pair, baseline, time_window)
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies
