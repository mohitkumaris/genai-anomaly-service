"""Latency anomaly detector implementation."""

from typing import Sequence

from anomaly.baselines.statistical import compute_deviation_score
from anomaly.detectors.interface import AnomalyDetector
from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class LatencyAnomalyDetector(AnomalyDetector):
    """Detects anomalies in latency predictions vs actual latency.
    
    Uses P99-based detection with a fallback to z-score:
    - Primary: Flags latencies that exceed historical P99
    - Secondary: Uses z-score for significant deviations
    
    Latency anomalies are critical for SLA compliance tracking.
    
    Algorithm:
    1. Extract actual and predicted latency
    2. Check if actual latency exceeds P99 threshold
    3. Compute deviation score from prediction
    4. Create anomaly if P99 exceeded OR significant deviation
    
    This detector is deterministic: identical inputs produce identical outputs.
    """

    _ALGORITHM_VERSION = "1.0.0"

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        p99_multiplier: float = 1.2,  # Flag if > 1.2x P99
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the latency anomaly detector.
        
        Args:
            z_score_threshold: Number of standard deviations for anomaly
            p99_multiplier: Flag if latency exceeds P99 * this multiplier
            min_confidence: Minimum confidence to report an anomaly
        """
        self._z_score_threshold = z_score_threshold
        self._p99_multiplier = p99_multiplier
        self._min_confidence = min_confidence

    @property
    def anomaly_type(self) -> AnomalyType:
        """Get the type of anomaly this detector identifies."""
        return AnomalyType.LATENCY

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
        """Detect a latency anomaly in a single comparison pair.
        
        Args:
            pair: Actual vs predicted comparison
            baseline: Historical baseline metrics for latency
            time_window: Time window for this analysis
            
        Returns:
            AnomalyRecord if latency anomaly detected, None otherwise
        """
        if not pair.has_latency_data:
            return None

        actual_latency = pair.actual.latency_ms
        predicted_latency = pair.predicted.predicted_latency_ms or 0.0

        # Compute deviation score
        deviation_score = compute_deviation_score(
            observed=actual_latency,
            expected=predicted_latency,
            baseline_std=baseline.std,
        )

        # Check if latency exceeds P99 threshold
        p99_threshold = baseline.p99 * self._p99_multiplier
        exceeds_p99 = actual_latency > p99_threshold

        # Check for significant deviation from prediction
        is_significant_deviation = deviation_score >= self._z_score_threshold

        # Flag if either condition is met (but only for high latency)
        # We only care about latency being TOO HIGH, not too low
        is_high_latency = actual_latency > predicted_latency
        if not ((exceeds_p99 or is_significant_deviation) and is_high_latency):
            return None

        # Compute confidence
        confidence = self._compute_confidence(deviation_score, baseline)

        # Boost confidence if exceeds P99
        if exceeds_p99:
            confidence = min(1.0, confidence + 0.15)

        if confidence < self._min_confidence:
            return None

        return AnomalyRecord(
            anomaly_type=AnomalyType.LATENCY,
            observed_value=actual_latency,
            expected_value=predicted_latency,
            deviation_score=deviation_score,
            confidence=confidence,
            algorithm_version=self.algorithm_version,
            time_window=time_window,
            metric_name="latency_ms",
            source_id=pair.actual.trace_id,
            metadata={
                "z_score_threshold": self._z_score_threshold,
                "p99_threshold": p99_threshold,
                "exceeds_p99": exceeds_p99,
                "baseline_p99": baseline.p99,
                "baseline_mean": baseline.mean,
            },
        )

    def detect_batch(
        self,
        pairs: Sequence[ComparisonPair],
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Detect latency anomalies in a batch of comparison pairs.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baseline: Historical baseline metrics for latency
            time_window: Time window for this analysis
            
        Returns:
            List of detected latency anomalies
        """
        anomalies = []
        for pair in pairs:
            anomaly = self.detect(pair, baseline, time_window)
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies
