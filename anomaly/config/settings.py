"""GenAI Anomaly Service configuration settings."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    COST = "cost"
    QUALITY = "quality"
    LATENCY = "latency"
    POLICY = "policy"


@dataclass(frozen=True)
class ThresholdConfig:
    """Threshold configuration for anomaly detection."""

    z_score_threshold: float = 2.0  # Standard deviations for z-score
    percentile_threshold: float = 0.95  # Percentile for outlier detection
    min_samples: int = 10  # Minimum samples required for detection
    confidence_threshold: float = 0.8  # Minimum confidence for reporting


@dataclass(frozen=True)
class TimeWindowConfig:
    """Time window configuration for analysis."""

    default_window: timedelta = field(default_factory=lambda: timedelta(hours=24))
    min_window: timedelta = field(default_factory=lambda: timedelta(hours=1))
    max_window: timedelta = field(default_factory=lambda: timedelta(days=30))


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for individual detectors."""

    enabled: bool = True
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    """Global settings for the anomaly service."""

    # Algorithm versioning - MUST be updated when detection logic changes
    algorithm_version: str = "1.0.0"

    # Service identification
    service_name: str = "genai-anomaly-service"
    service_version: str = "0.1.0"

    # Time window defaults
    time_window: TimeWindowConfig = field(default_factory=TimeWindowConfig)

    # Per-detector configurations
    cost_detector: DetectorConfig = field(default_factory=DetectorConfig)
    quality_detector: DetectorConfig = field(default_factory=DetectorConfig)
    latency_detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy_detector: DetectorConfig = field(default_factory=DetectorConfig)

    # Storage settings
    storage_path: str = "./data/anomalies"
    enable_file_storage: bool = True

    # API settings
    api_prefix: str = "/api/v1"
    debug: bool = False


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(settings: Settings) -> None:
    """Configure the global settings (primarily for testing)."""
    global _settings
    _settings = settings
