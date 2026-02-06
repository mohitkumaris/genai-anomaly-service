"""Unit tests for anomaly storage."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from anomaly.models import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.store import AnomalyStoreFilter, FileAnomalyStore, MemoryAnomalyStore


def create_test_record(
    anomaly_type: AnomalyType = AnomalyType.COST,
    deviation_score: float = 2.5,
    confidence: float = 0.8,
    timestamp: datetime | None = None,
) -> AnomalyRecord:
    """Create a test anomaly record."""
    now = datetime.utcnow()
    return AnomalyRecord(
        anomaly_type=anomaly_type,
        observed_value=150.0,
        expected_value=100.0,
        deviation_score=deviation_score,
        confidence=confidence,
        algorithm_version="1.0.0",
        time_window=TimeWindow(start=now - timedelta(hours=1), end=now),
        timestamp=timestamp or now,
        metric_name="test_metric",
        source_id="test_source",
    )


class TestMemoryAnomalyStore:
    """Tests for MemoryAnomalyStore."""

    @pytest.fixture
    def store(self) -> MemoryAnomalyStore:
        """Create a fresh store instance."""
        return MemoryAnomalyStore()

    def test_append_and_retrieve(self, store: MemoryAnomalyStore) -> None:
        """Test appending and retrieving a record."""
        record = create_test_record()
        store.append(record)

        retrieved = store.get_by_id(str(record.record_id))
        assert retrieved is not None
        assert retrieved.record_id == record.record_id
        assert retrieved.anomaly_type == record.anomaly_type

    def test_append_only_no_duplicates(self, store: MemoryAnomalyStore) -> None:
        """Test that duplicate records are rejected."""
        record = create_test_record()
        store.append(record)

        with pytest.raises(ValueError, match="already exists"):
            store.append(record)

    def test_query_by_type(self, store: MemoryAnomalyStore) -> None:
        """Test querying by anomaly type."""
        store.append(create_test_record(anomaly_type=AnomalyType.COST))
        store.append(create_test_record(anomaly_type=AnomalyType.LATENCY))
        store.append(create_test_record(anomaly_type=AnomalyType.COST))

        filter_criteria = AnomalyStoreFilter(anomaly_types=[AnomalyType.COST])
        results = store.query(filter_criteria)

        assert len(results) == 2
        assert all(r.anomaly_type == AnomalyType.COST for r in results)

    def test_query_by_confidence(self, store: MemoryAnomalyStore) -> None:
        """Test querying by minimum confidence."""
        store.append(create_test_record(confidence=0.5))
        store.append(create_test_record(confidence=0.9))
        store.append(create_test_record(confidence=0.7))

        filter_criteria = AnomalyStoreFilter(min_confidence=0.8)
        results = store.query(filter_criteria)

        assert len(results) == 1
        assert results[0].confidence == 0.9

    def test_replay_chronological_order(self, store: MemoryAnomalyStore) -> None:
        """Test that replay returns records in chronological order."""
        now = datetime.utcnow()
        store.append(create_test_record(timestamp=now - timedelta(hours=2)))
        store.append(create_test_record(timestamp=now - timedelta(hours=1)))
        store.append(create_test_record(timestamp=now))

        time_window = TimeWindow(start=now - timedelta(hours=3), end=now)
        results = store.replay(time_window)

        assert len(results) == 3
        # Should be in chronological order (oldest first)
        assert results[0].timestamp < results[1].timestamp < results[2].timestamp

    def test_count(self, store: MemoryAnomalyStore) -> None:
        """Test record counting."""
        assert store.count() == 0

        store.append(create_test_record())
        store.append(create_test_record())

        assert store.count() == 2

    def test_statistics(self, store: MemoryAnomalyStore) -> None:
        """Test storage statistics."""
        store.append(create_test_record(anomaly_type=AnomalyType.COST))
        store.append(create_test_record(anomaly_type=AnomalyType.LATENCY))

        stats = store.get_statistics()
        assert stats["total_records"] == 2
        assert stats["records_by_type"]["cost"] == 1
        assert stats["records_by_type"]["latency"] == 1
        assert stats["storage_type"] == "memory"
        assert stats["is_persistent"] is False


class TestFileAnomalyStore:
    """Tests for FileAnomalyStore."""

    @pytest.fixture
    def store(self) -> FileAnomalyStore:
        """Create a store with a temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileAnomalyStore(Path(tmpdir) / "anomalies.jsonl")

    def test_append_and_retrieve(self, store: FileAnomalyStore) -> None:
        """Test appending and retrieving a record."""
        record = create_test_record()
        store.append(record)

        retrieved = store.get_by_id(str(record.record_id))
        assert retrieved is not None
        assert retrieved.record_id == record.record_id

    def test_persistence(self) -> None:
        """Test that data persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "anomalies.jsonl"

            # Create first store and add record
            store1 = FileAnomalyStore(file_path)
            record = create_test_record()
            store1.append(record)

            # Create second store from same file
            store2 = FileAnomalyStore(file_path)
            retrieved = store2.get_by_id(str(record.record_id))

            assert retrieved is not None
            assert retrieved.record_id == record.record_id

    def test_statistics_persistent(self) -> None:
        """Test that file store reports correct statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileAnomalyStore(Path(tmpdir) / "anomalies.jsonl")
            store.append(create_test_record())

            stats = store.get_statistics()
            assert stats["total_records"] == 1
            assert stats["storage_type"] == "file"
            assert stats["is_persistent"] is True


class TestAppendOnlySemantics:
    """Tests to verify append-only semantics are maintained."""

    def test_memory_store_append_only(self) -> None:
        """Test that memory store is truly append-only."""
        store = MemoryAnomalyStore()
        record = create_test_record()
        store.append(record)

        # Verify no update methods exist
        assert not hasattr(store, "update")
        assert not hasattr(store, "delete")

        # Verify the record is immutable (frozen dataclass)
        with pytest.raises(AttributeError):
            record.deviation_score = 5.0  # type: ignore

    def test_file_store_append_only(self) -> None:
        """Test that file store is truly append-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileAnomalyStore(Path(tmpdir) / "anomalies.jsonl")
            record = create_test_record()
            store.append(record)

            # Verify no update methods exist
            assert not hasattr(store, "update")
            assert not hasattr(store, "delete")
