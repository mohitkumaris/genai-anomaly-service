"""Quality anomaly detector implementation."""

from typing import Sequence

from anomaly.baselines.statistical import compute_deviation_score
from anomaly.detectors.interface import AnomalyDetector
from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class QualityAnomalyDetector(AnomalyDetector):
    """Detects anomalies in quality score predictions vs actual scores.
    
    Uses both z-score and percentile-based detection:
    - Z-score: Flags large deviations from expected quality
    - Percentile: Flags quality scores below historical thresholds
    
    Quality anomalies are particularly important as they may indicate
    model degradation or content quality issues.
    
    Algorithm:
    1. Extract actual and predicted quality scores
    2. Compute deviation score
    3. Check if below percentile threshold OR deviation is significant
    4. Create anomaly record if either condition is met
    
    This detector is deterministic: identical inputs produce identical outputs.
    """

    _ALGORITHM_VERSION = "1.0.0"

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        percentile_threshold: float = 0.10,  # Flag if below p10
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the quality anomaly detector.
        
        Args:
            z_score_threshold: Number of standard deviations for anomaly
            percentile_threshold: Quality below this percentile is anomalous
            min_confidence: Minimum confidence to report an anomaly
        """
        self._z_score_threshold = z_score_threshold
        self._percentile_threshold = percentile_threshold
        self._min_confidence = min_confidence

    @property
    def anomaly_type(self) -> AnomalyType:
        """Get the type of anomaly this detector identifies."""
        return AnomalyType.QUALITY

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
        """Detect a quality anomaly in a single comparison pair.
        
        Args:
            pair: Actual vs predicted comparison
            baseline: Historical baseline metrics for quality
            time_window: Time window for this analysis
            
        Returns:
            AnomalyRecord if quality anomaly detected, None otherwise
        """
        if not pair.has_quality_data:
            return None

        actual_quality = pair.actual.quality_score or 0.0
        predicted_quality = pair.predicted.predicted_quality_score or 0.0

        # Compute deviation score
        deviation_score = compute_deviation_score(
            observed=actual_quality,
            expected=predicted_quality,
            baseline_std=baseline.std,
        )

        # Check for significant deviation
        is_significant_deviation = deviation_score >= self._z_score_threshold

        # Check if quality is below acceptable threshold
        # Using p50 (median) as reference, check if significantly below
        is_below_threshold = actual_quality < (baseline.p50 * (1 - self._percentile_threshold))

        # Flag if either condition is met
        if not (is_significant_deviation or is_below_threshold):
            return None

        # Compute confidence
        confidence = self._compute_confidence(deviation_score, baseline)

        # Boost confidence if quality is below threshold
        if is_below_threshold:
            confidence = min(1.0, confidence + 0.1)

        if confidence < self._min_confidence:
            return None

        return AnomalyRecord(
            anomaly_type=AnomalyType.QUALITY,
            observed_value=actual_quality,
            expected_value=predicted_quality,
            deviation_score=deviation_score,
            confidence=confidence,
            algorithm_version=self.algorithm_version,
            time_window=time_window,
            metric_name="quality_score",
            source_id=pair.actual.trace_id,
            metadata={
                "z_score_threshold": self._z_score_threshold,
                "percentile_threshold": self._percentile_threshold,
                "is_deviation_based": is_significant_deviation,
                "is_below_threshold": is_below_threshold,
                "baseline_median": baseline.p50,
            },
        )

    def detect_batch(
        self,
        pairs: Sequence[ComparisonPair],
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Detect quality anomalies in a batch of comparison pairs.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baseline: Historical baseline metrics for quality
            time_window: Time window for this analysis
            
        Returns:
            List of detected quality anomalies
        """
        anomalies = []
        for pair in pairs:
            anomaly = self.detect(pair, baseline, time_window)
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies
