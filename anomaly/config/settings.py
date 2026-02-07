"""GenAI Anomaly Service configuration settings."""

import os
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

    # LLMOps integration (READ-ONLY, PULL-ONLY)
    # DESIGN RULES:
    # - HTTP GET only (no writes)
    # - Fail-open: return empty on any error
    # - Timeout ≤1s, no retries
    llmops_enabled: bool = True
    llmops_base_url: str = "http://localhost:8100"
    llmops_timeout_ms: int = 1000  # ≤1 second, no retries
    llmops_data_dir: str | None = None  # For file-based reading

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

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            algorithm_version=os.getenv("ALGORITHM_VERSION", "1.0.0"),
            llmops_enabled=os.getenv("LLMOPS_ENABLED", "true").lower() == "true",
            llmops_base_url=os.getenv("LLMOPS_BASE_URL", "http://localhost:8100"),
            llmops_timeout_ms=int(os.getenv("LLMOPS_TIMEOUT_MS", "1000")),
            llmops_data_dir=os.getenv("LLMOPS_DATA_DIR"),
            storage_path=os.getenv("STORAGE_PATH", "./data/anomalies"),
            enable_file_storage=os.getenv("ENABLE_FILE_STORAGE", "true").lower() == "true",
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure(settings: Settings) -> None:
    """Configure the global settings (primarily for testing)."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to None (for testing)."""
    global _settings
    _settings = None
