"""Detector registry for managing and accessing anomaly detectors."""

from typing import Sequence

from anomaly.config.settings import DetectorConfig, get_settings
from anomaly.detectors.cost_detector import CostAnomalyDetector
from anomaly.detectors.interface import AnomalyDetector
from anomaly.detectors.latency_detector import LatencyAnomalyDetector
from anomaly.detectors.policy_detector import PolicyAnomalyDetector
from anomaly.detectors.quality_detector import QualityAnomalyDetector
from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.models.baseline import BaselineMetrics
from anomaly.models.input_data import ComparisonPair


class DetectorRegistry:
    """Registry for managing anomaly detector instances.
    
    Provides:
    - Factory methods for creating configured detectors
    - Batch detection across all enabled detectors
    - Detector lookup by anomaly type
    
    This registry is the primary entry point for anomaly detection.
    """

    def __init__(self) -> None:
        """Initialize the detector registry with default detectors."""
        self._detectors: dict[AnomalyType, AnomalyDetector] = {}
        self._initialize_default_detectors()

    def _initialize_default_detectors(self) -> None:
        """Initialize detectors based on settings."""
        settings = get_settings()

        if settings.cost_detector.enabled:
            self._detectors[AnomalyType.COST] = self._create_cost_detector(
                settings.cost_detector
            )

        if settings.quality_detector.enabled:
            self._detectors[AnomalyType.QUALITY] = self._create_quality_detector(
                settings.quality_detector
            )

        if settings.latency_detector.enabled:
            self._detectors[AnomalyType.LATENCY] = self._create_latency_detector(
                settings.latency_detector
            )

        if settings.policy_detector.enabled:
            self._detectors[AnomalyType.POLICY] = self._create_policy_detector(
                settings.policy_detector
            )

    @staticmethod
    def _create_cost_detector(config: DetectorConfig) -> CostAnomalyDetector:
        """Create a configured cost anomaly detector."""
        return CostAnomalyDetector(
            z_score_threshold=config.threshold.z_score_threshold,
            min_confidence=config.threshold.confidence_threshold,
        )

    @staticmethod
    def _create_quality_detector(config: DetectorConfig) -> QualityAnomalyDetector:
        """Create a configured quality anomaly detector."""
        return QualityAnomalyDetector(
            z_score_threshold=config.threshold.z_score_threshold,
            min_confidence=config.threshold.confidence_threshold,
        )

    @staticmethod
    def _create_latency_detector(config: DetectorConfig) -> LatencyAnomalyDetector:
        """Create a configured latency anomaly detector."""
        return LatencyAnomalyDetector(
            z_score_threshold=config.threshold.z_score_threshold,
            min_confidence=config.threshold.confidence_threshold,
        )

    @staticmethod
    def _create_policy_detector(config: DetectorConfig) -> PolicyAnomalyDetector:
        """Create a configured policy anomaly detector."""
        return PolicyAnomalyDetector(
            min_confidence=config.threshold.confidence_threshold,
        )

    def get_detector(self, anomaly_type: AnomalyType) -> AnomalyDetector | None:
        """Get a detector by anomaly type.
        
        Args:
            anomaly_type: Type of anomaly to get detector for
            
        Returns:
            The detector if registered and enabled, None otherwise
        """
        return self._detectors.get(anomaly_type)

    def register_detector(
        self,
        anomaly_type: AnomalyType,
        detector: AnomalyDetector,
    ) -> None:
        """Register a custom detector.
        
        Args:
            anomaly_type: Type of anomaly this detector handles
            detector: The detector instance
        """
        if detector.anomaly_type != anomaly_type:
            raise ValueError(
                f"Detector anomaly type {detector.anomaly_type} does not match "
                f"registration type {anomaly_type}"
            )
        self._detectors[anomaly_type] = detector

    def list_enabled_types(self) -> list[AnomalyType]:
        """List all enabled anomaly types."""
        return list(self._detectors.keys())

    def detect_all(
        self,
        pairs: Sequence[ComparisonPair],
        baselines: dict[AnomalyType, BaselineMetrics],
        time_window: TimeWindow,
    ) -> list[AnomalyRecord]:
        """Run all enabled detectors on the given data.
        
        Args:
            pairs: Sequence of actual vs predicted comparisons
            baselines: Baseline metrics by anomaly type
            time_window: Time window for this analysis
            
        Returns:
            Combined list of all detected anomalies
        """
        all_anomalies: list[AnomalyRecord] = []

        for anomaly_type, detector in self._detectors.items():
            baseline = baselines.get(anomaly_type)
            if baseline is None:
                continue

            anomalies = detector.detect_batch(pairs, baseline, time_window)
            all_anomalies.extend(anomalies)

        return all_anomalies


# Global registry instance
_registry: DetectorRegistry | None = None


def get_detector_registry() -> DetectorRegistry:
    """Get or create the global detector registry."""
    global _registry
    if _registry is None:
        _registry = DetectorRegistry()
    return _registry
