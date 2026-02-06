"""File-based anomaly store implementation."""

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from anomaly.models.anomaly_record import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.store.interface import AnomalyStoreFilter, BaseAnomalyStore


class FileAnomalyStore(BaseAnomalyStore):
    """File-based implementation of AnomalyStore using JSON Lines format.
    
    This implementation stores anomaly records in a JSON Lines (JSONL) file,
    where each line is a complete JSON object representing one record.
    
    PROPERTIES:
    - Append-only: New records are appended to the file
    - Immutable: Once written, records cannot be modified
    - Persistent: Data survives process restart
    - Auditable: Full history is preserved in the file
    
    FILE FORMAT:
    - One JSON object per line
    - Append-only (file is never truncated or rewritten)
    - Uses file locking for concurrent access safety
    """

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize the file-based store.
        
        Args:
            storage_path: Path to the storage file or directory
        """
        self._storage_path = Path(storage_path)

        if self._storage_path.is_dir():
            self._file_path = self._storage_path / "anomalies.jsonl"
        else:
            self._file_path = self._storage_path

        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self._file_path.exists():
            self._file_path.touch()

    def append(self, record: AnomalyRecord) -> None:
        """Append a single anomaly record to the store.
        
        Args:
            record: The anomaly record to store
        """
        self._validate_record(record)

        # Check for duplicate
        existing = self.get_by_id(str(record.record_id))
        if existing is not None:
            raise ValueError(f"Record with ID {record.record_id} already exists")

        # Serialize and append
        line = json.dumps(record.to_dict()) + "\n"

        with open(self._file_path, "a", encoding="utf-8") as f:
            # Use file locking for concurrent access
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def append_batch(self, records: Sequence[AnomalyRecord]) -> int:
        """Append multiple anomaly records to the store.
        
        Args:
            records: Sequence of anomaly records to store
            
        Returns:
            Number of records successfully stored
        """
        # Validate all records first
        for record in records:
            self._validate_record(record)

        # Get existing IDs to check for duplicates
        existing_ids = {str(r.record_id) for r in self._read_all_records()}

        # Prepare lines to write
        lines_to_write = []
        count = 0
        for record in records:
            if str(record.record_id) in existing_ids:
                continue
            lines_to_write.append(json.dumps(record.to_dict()) + "\n")
            existing_ids.add(str(record.record_id))
            count += 1

        if not lines_to_write:
            return 0

        # Write all at once with lock
        with open(self._file_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.writelines(lines_to_write)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return count

    def query(self, filter_criteria: AnomalyStoreFilter) -> list[AnomalyRecord]:
        """Query anomaly records matching the filter criteria.
        
        Args:
            filter_criteria: Filter criteria for the query
            
        Returns:
            List of matching anomaly records (newest first)
        """
        records = self._read_all_records()

        # Apply filters
        results = [r for r in records if filter_criteria.matches(r)]

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
        try:
            target_uuid = UUID(record_id)
        except ValueError:
            return None

        for record in self._read_all_records():
            if record.record_id == target_uuid:
                return record
        return None

    def replay(self, time_window: TimeWindow) -> list[AnomalyRecord]:
        """Retrieve all records within a time window for replay analysis.
        
        Args:
            time_window: Time window to replay
            
        Returns:
            All anomaly records within the time window (chronological order)
        """
        records = self._read_all_records()

        results = [
            r
            for r in records
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
        records = self._read_all_records()

        if filter_criteria is None:
            return len(records)

        return sum(1 for r in records if filter_criteria.matches(r))

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        records = self._read_all_records()

        records_by_type: dict[str, int] = {}
        for record in records:
            type_name = record.anomaly_type.value
            records_by_type[type_name] = records_by_type.get(type_name, 0) + 1

        file_size = self._file_path.stat().st_size if self._file_path.exists() else 0

        return {
            "total_records": len(records),
            "records_by_type": records_by_type,
            "storage_type": "file",
            "file_path": str(self._file_path),
            "file_size_bytes": file_size,
            "is_persistent": True,
        }

    def _read_all_records(self) -> list[AnomalyRecord]:
        """Read all records from the file.
        
        Returns:
            List of all anomaly records
        """
        records = []

        if not self._file_path.exists():
            return records

        with open(self._file_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = AnomalyRecord.from_dict(data)
                        records.append(record)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip malformed lines
                        continue
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return records
