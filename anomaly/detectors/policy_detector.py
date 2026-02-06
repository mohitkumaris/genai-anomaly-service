"""Policy outcome anomaly detector implementation."""

from typing import Sequence

from anomaly.detectors.interface import AnomalyDetector
from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class PolicyAnomalyDetector(AnomalyDetector):
    """Detects anomalies in policy outcome predictions.
    
    Unlike other detectors that use continuous metrics, policy anomalies
    are detected based on binary outcome mismatches:
    - Predicted pass, actual fail
    - Predicted fail, actual pass
    
    Both directions are flagged as they indicate prediction model issues.
    
    Algorithm:
    1. Extract actual and predicted policy outcomes
    2. Check for mismatch (predicted != actual)
    3. Create anomaly record with appropriate context
    
    This detector is deterministic: identical inputs produce identical outputs.
    """

    _ALGORITHM_VERSION = "1.0.0"

    def __init__(
        self,
        min_confidence: float = 0.5,
        weight_unexpected_fail: float = 1.5,  # Higher weight for unexpected failures
    ) -> None:
        """Initialize the policy anomaly detector.
        
        Args:
            min_confidence: Minimum confidence to report an anomaly
            weight_unexpected_fail: Weight multiplier for unexpected failures
        """
        self._min_confidence = min_confidence
        self._weight_unexpected_fail = weight_unexpected_fail

    @property
    def anomaly_type(self) -> AnomalyType:
        """Get the type of anomaly this detector identifies."""
        return AnomalyType.POLICY

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
        """Detect a policy anomaly in a single comparison pair.
        
        Args:
            pair: Actual vs predicted comparison
            baseline: Historical baseline metrics (used for confidence)
            time_window: Time window for this analysis
            
        Returns:
            AnomalyRecord if policy mismatch detected, None otherwise
        """
        if not pair.has_policy_data:
            return None

        actual_passed = pair.actual.policy_passed
        predicted_pass = pair.predicted.predicted_policy_pass

        # Check for mismatch
        if actual_passed == predicted_pass:
            return None

        # Determine the type of mismatch
        is_unexpected_fail = predicted_pass and not actual_passed
        is_unexpected_pass = not predicted_pass and actual_passed

        # Compute deviation score
        # For binary outcomes, deviation is 1.0 for any mismatch
        deviation_score = 1.0 * (self._weight_unexpected_fail if is_unexpected_fail else 1.0)

        # Compute confidence based on baseline sample count
        base_confidence = min(1.0, baseline.sample_count / 100)

        # Adjust based on prediction confidence
        prediction_confidence = pair.predicted.confidence
        confidence = 0.6 * base_confidence + 0.4 * prediction_confidence

        if confidence < self._min_confidence:
            return None

        # Encode outcomes as numeric for the record
        observed_value = 1.0 if actual_passed else 0.0
        expected_value = 1.0 if predicted_pass else 0.0

        return AnomalyRecord(
            anomaly_type=AnomalyType.POLICY,
            observed_value=observed_value,
            expected_value=expected_value,
            deviation_score=deviation_score,
            confidence=confidence,
            algorithm_version=self.algorithm_version,
            time_window=time_window,
            metric_name="policy_outcome",
            source_id=pair.actual.trace_id,
            metadata={
                "is_unexpected_fail": is_unexpected_fail,
                "is_unexpected_pass": is_unexpected_pass,
                "policy_violations": pair.actual.policy_violations,
                "prediction_confidence": prediction_confidence,
            },
        )

    def detect_batch(
        self,
        pairs: Sequence[ComparisonPair],
        baseline: BaselineMetrics,
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Detect policy anomalies in a batch of comparison pairs.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baseline: Historical baseline metrics
            time_window: Time window for this analysis
            
        Returns:
            List of detected policy anomalies
        """
        anomalies = []
        for pair in pairs:
            anomaly = self.detect(pair, baseline, time_window)
            if anomaly is not None:
                anomalies.append(anomaly)
        return anomalies
