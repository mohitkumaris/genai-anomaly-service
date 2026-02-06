"""Interface for baseline calculators."""

from abc import ABC, abstractmethod
from typing import Sequence

from anomaly.models.baseline import BaselineMetrics, BaselineSnapshot


class BaselineCalculator(ABC):
    """Abstract interface for baseline computation strategies.
    
    Implementations must be:
    - Deterministic: Same inputs produce same outputs
    - Pure: No side effects
    - Versioned: Algorithm version must be tracked
    """

    @property
    @abstractmethod
    def algorithm_version(self) -> str:
        """Get the algorithm version for this calculator."""
        ...

    @abstractmethod
    def compute(self, values: Sequence[float]) -> BaselineMetrics:
        """Compute baseline metrics from a sequence of values.
        
        Args:
            values: Sequence of numeric values to compute baseline from
            
        Returns:
            Computed baseline metrics
            
        Raises:
            ValueError: If insufficient data for computation
        """
        ...

    @abstractmethod
    def create_snapshot(
        self,
        metric_type: str,
        values: Sequence[float],
        time_window_hours: int,
        source_description: str = "",
    ) -> BaselineSnapshot:
        """Create a complete baseline snapshot.
        
        Args:
            metric_type: Type of metric (e.g., "cost", "latency")
            values: Sequence of numeric values
            time_window_hours: Hours of historical data
            source_description: Description of data source
            
        Returns:
            Complete baseline snapshot with metadata
        """
        ...
