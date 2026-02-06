"""Interface for anomaly storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow


class AnomalyStoreFilter:
    """Filter criteria for querying anomaly records.
    
    All filter criteria are optional and applied with AND logic.
    """

    def __init__(
        self,
        anomaly_types: list[AnomalyType] | None = None,
        time_window: TimeWindow | None = None,
        min_confidence: float | None = None,
        source_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        """Initialize filter criteria.
        
        Args:
            anomaly_types: Filter by anomaly types (OR within list)
            time_window: Filter by time range
            min_confidence: Minimum confidence threshold
            source_ids: Filter by source IDs (OR within list)
            limit: Maximum number of records to return
        """
        self.anomaly_types = anomaly_types
        self.time_window = time_window
        self.min_confidence = min_confidence
        self.source_ids = source_ids
        self.limit = limit

    def matches(self, record: AnomalyRecord) -> bool:
        """Check if a record matches this filter.
        
        Args:
            record: The anomaly record to check
            
        Returns:
            True if the record matches all filter criteria
        """
        # Type filter
        if self.anomaly_types and record.anomaly_type not in self.anomaly_types:
            return False

        # Time window filter
        if self.time_window:
            if record.timestamp < self.time_window.start:
                return False
            if record.timestamp > self.time_window.end:
                return False

        # Confidence filter
        if self.min_confidence and record.confidence < self.min_confidence:
            return False

        # Source ID filter
        if self.source_ids and record.source_id not in self.source_ids:
            return False

        return True


class AnomalyStore(ABC):
    """Abstract interface for anomaly record persistence.
    
    CRITICAL PROPERTIES:
    - Append-only: Records can only be added, never modified or deleted
    - Immutable: Once written, records cannot be changed
    - Replayable: All records can be queried for replay analysis
    - Auditable: Full history is preserved
    
    This interface ensures storage implementations comply with the
    architectural requirement of append-only, immutable records.
    """

    @abstractmethod
    def append(self, record: AnomalyRecord) -> None:
        """Append a single anomaly record to the store.
        
        Args:
            record: The anomaly record to store
            
        Note:
            This is an append-only operation. The record becomes
            immutable once stored and cannot be modified or deleted.
        """
        ...

    @abstractmethod
    def append_batch(self, records: Sequence[AnomalyRecord]) -> int:
        """Append multiple anomaly records to the store.
        
        Args:
            records: Sequence of anomaly records to store
            
        Returns:
            Number of records successfully stored
            
        Note:
            This is an atomic append operation for efficiency.
        """
        ...

    @abstractmethod
    def query(self, filter_criteria: AnomalyStoreFilter) -> list[AnomalyRecord]:
        """Query anomaly records matching the filter criteria.
        
        Args:
            filter_criteria: Filter criteria for the query
            
        Returns:
            List of matching anomaly records (order by timestamp desc)
        """
        ...

    @abstractmethod
    def get_by_id(self, record_id: str) -> AnomalyRecord | None:
        """Get a specific anomaly record by ID.
        
        Args:
            record_id: The record ID (UUID string)
            
        Returns:
            The anomaly record if found, None otherwise
        """
        ...

    @abstractmethod
    def replay(self, time_window: TimeWindow) -> list[AnomalyRecord]:
        """Retrieve all records within a time window for replay analysis.
        
        Args:
            time_window: Time window to replay
            
        Returns:
            All anomaly records within the time window (chronological order)
        """
        ...

    @abstractmethod
    def count(self, filter_criteria: AnomalyStoreFilter | None = None) -> int:
        """Count anomaly records, optionally filtered.
        
        Args:
            filter_criteria: Optional filter criteria
            
        Returns:
            Count of matching records
        """
        ...

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with statistics like total_records, records_by_type, etc.
        """
        ...


class BaseAnomalyStore(AnomalyStore):
    """Base implementation with common functionality."""

    def _validate_record(self, record: AnomalyRecord) -> None:
        """Validate a record before storage.
        
        Args:
            record: The record to validate
            
        Raises:
            ValueError: If the record is invalid
        """
        if not record.algorithm_version:
            raise ValueError("Record must have an algorithm version")
        if not record.time_window:
            raise ValueError("Record must have a time window")

    def _sort_by_timestamp(
        self,
        records: list[AnomalyRecord],
        descending: bool = True,
    ) -> list[AnomalyRecord]:
        """Sort records by timestamp.
        
        Args:
            records: Records to sort
            descending: If True, newest first
            
        Returns:
            Sorted list of records
        """
        return sorted(records, key=lambda r: r.timestamp, reverse=descending)
