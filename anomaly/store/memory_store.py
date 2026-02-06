"""In-memory anomaly store implementation."""

import threading
from collections import defaultdict
from typing import Any, Sequence
from uuid import UUID

from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.store.interface import AnomalyStoreFilter, BaseAnomalyStore


class MemoryAnomalyStore(BaseAnomalyStore):
    """Thread-safe in-memory implementation of AnomalyStore.
    
    This implementation is primarily for testing and development.
    It maintains the append-only semantics required by the specification.
    
    PROPERTIES:
    - Thread-safe: Uses lock for concurrent access
    - Append-only: No delete or update operations
    - In-memory: Data is lost on process restart
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._records: list[AnomalyRecord] = []
        self._by_id: dict[UUID, AnomalyRecord] = {}
        self._by_type: dict[AnomalyType, list[AnomalyRecord]] = defaultdict(list)
        self._lock = threading.Lock()

    def append(self, record: AnomalyRecord) -> None:
        """Append a single anomaly record to the store.
        
        Args:
            record: The anomaly record to store
        """
        self._validate_record(record)

        with self._lock:
            if record.record_id in self._by_id:
                raise ValueError(f"Record with ID {record.record_id} already exists")

            self._records.append(record)
            self._by_id[record.record_id] = record
            self._by_type[record.anomaly_type].append(record)

    def append_batch(self, records: Sequence[AnomalyRecord]) -> int:
        """Append multiple anomaly records to the store.
        
        Args:
            records: Sequence of anomaly records to store
            
        Returns:
            Number of records successfully stored
        """
        count = 0
        for record in records:
            try:
                self.append(record)
                count += 1
            except ValueError:
                # Skip duplicate records
                continue
        return count

    def query(self, filter_criteria: AnomalyStoreFilter) -> list[AnomalyRecord]:
        """Query anomaly records matching the filter criteria.
        
        Args:
            filter_criteria: Filter criteria for the query
            
        Returns:
            List of matching anomaly records (newest first)
        """
        with self._lock:
            # Start with type-filtered records if types specified
            if filter_criteria.anomaly_types:
                candidates = []
                for at in filter_criteria.anomaly_types:
                    candidates.extend(self._by_type.get(at, []))
            else:
                candidates = list(self._records)

            # Apply remaining filters
            results = [r for r in candidates if filter_criteria.matches(r)]

            # Sort by timestamp (newest first)
            results = self._sort_by_timestamp(results, descending=True)

            # Apply limit
            if filter_criteria.limit:
                results = results[: filter_criteria.limit]

            return results

    def get_by_id(self, record_id: str) -> AnomalyRecord | None:
        """Get a specific anomaly record by ID.
        
        Args:
            record_id: The record ID (UUID string)
            
        Returns:
            The anomaly record if found, None otherwise
        """
        with self._lock:
            try:
                uuid = UUID(record_id)
                return self._by_id.get(uuid)
            except ValueError:
                return None

    def replay(self, time_window: TimeWindow) -> list[AnomalyRecord]:
        """Retrieve all records within a time window for replay analysis.
        
        Args:
            time_window: Time window to replay
            
        Returns:
            All anomaly records within the time window (chronological order)
        """
        with self._lock:
            results = [
                r
                for r in self._records
                if time_window.start <= r.timestamp <= time_window.end
            ]
            # Chronological order for replay
            return self._sort_by_timestamp(results, descending=False)

    def count(self, filter_criteria: AnomalyStoreFilter | None = None) -> int:
        """Count anomaly records, optionally filtered.
        
        Args:
            filter_criteria: Optional filter criteria
            
        Returns:
            Count of matching records
        """
        with self._lock:
            if filter_criteria is None:
                return len(self._records)

            return sum(1 for r in self._records if filter_criteria.matches(r))

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        with self._lock:
            return {
                "total_records": len(self._records),
                "records_by_type": {
                    at.value: len(records) for at, records in self._by_type.items()
                },
                "storage_type": "memory",
                "is_persistent": False,
            }

    def clear(self) -> None:
        """Clear all records (for testing only).
        
        WARNING: This violates append-only semantics and should
        only be used in test environments.
        """
        with self._lock:
            self._records.clear()
            self._by_id.clear()
            self._by_type.clear()
