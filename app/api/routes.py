"""Anomaly API routes."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from anomaly.config import AnomalyType, get_settings
from anomaly.models import (
    AnomalyCount,
    AnomalyRecord,
    TimeWindow,
    TrustLevel,
    TrustSignal,
    compute_trust_level,
)
from anomaly.store import AnomalyStore, AnomalyStoreFilter, MemoryAnomalyStore

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])

# Store instance (will be configured in main.py)
_store: AnomalyStore | None = None


def get_store() -> AnomalyStore:
    """Get the anomaly store instance."""
    global _store
    if _store is None:
        _store = MemoryAnomalyStore()
    return _store


def set_store(store: AnomalyStore) -> None:
    """Set the anomaly store instance."""
    global _store
    _store = store


# --- Request/Response Models ---


class TimeWindowRequest(BaseModel):
    """Time window for queries and analysis."""

    start: datetime = Field(..., description="Start of time window (inclusive)")
    end: datetime = Field(..., description="End of time window (inclusive)")


class AnomalyRecordResponse(BaseModel):
    """Response model for an anomaly record."""

    record_id: str
    anomaly_type: str
    observed_value: float
    expected_value: float
    deviation_score: float
    confidence: float
    algorithm_version: str
    time_window: dict[str, str]
    timestamp: str
    metric_name: str
    source_id: str
    deviation_percentage: float
    is_positive_deviation: bool
    metadata: dict[str, Any]

    @classmethod
    def from_record(cls, record: AnomalyRecord) -> "AnomalyRecordResponse":
        """Create response from domain model."""
        return cls(
            record_id=str(record.record_id),
            anomaly_type=record.anomaly_type.value,
            observed_value=record.observed_value,
            expected_value=record.expected_value,
            deviation_score=record.deviation_score,
            confidence=record.confidence,
            algorithm_version=record.algorithm_version,
            time_window={
                "start": record.time_window.start.isoformat(),
                "end": record.time_window.end.isoformat(),
            },
            timestamp=record.timestamp.isoformat(),
            metric_name=record.metric_name,
            source_id=record.source_id,
            deviation_percentage=record.deviation_percentage,
            is_positive_deviation=record.is_positive_deviation,
            metadata=record.metadata,
        )


class AnomalyListResponse(BaseModel):
    """Response model for a list of anomalies."""

    anomalies: list[AnomalyRecordResponse]
    total_count: int
    returned_count: int


class ReplayRequest(BaseModel):
    """Request model for replay analysis."""

    time_window: TimeWindowRequest


class ReplayResponse(BaseModel):
    """Response model for replay analysis."""

    time_window: dict[str, str]
    anomalies: list[AnomalyRecordResponse]
    count: int
    summary: dict[str, int]


class TrustSignalResponse(BaseModel):
    """Response model for trust signals."""

    trust_level: str
    anomaly_counts: dict[str, int]
    confidence: float
    computed_at: str
    algorithm_version: str
    time_window_hours: int
    has_anomalies: bool
    severity_score: float


class StorageStatsResponse(BaseModel):
    """Response model for storage statistics."""

    total_records: int
    records_by_type: dict[str, int]
    storage_type: str
    is_persistent: bool


# --- Endpoints ---


@router.get("", response_model=AnomalyListResponse)
async def list_anomalies(
    anomaly_type: list[str] | None = Query(default=None, description="Filter by anomaly types"),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    start: datetime | None = Query(default=None, description="Time window start"),
    end: datetime | None = Query(default=None, description="Time window end"),
    limit: int = Query(default=100, ge=1, le=1000),
    store: AnomalyStore = Depends(get_store),
) -> AnomalyListResponse:
    """List anomaly records with optional filters.
    
    Returns anomalies matching the specified criteria, ordered by
    timestamp (newest first).
    
    Note: This endpoint only reads data. It does not trigger any actions.
    """
    # Build filter
    anomaly_types = None
    if anomaly_type:
        try:
            anomaly_types = [AnomalyType(t) for t in anomaly_type]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid anomaly type: {e}")

    time_window = None
    if start and end:
        if start >= end:
            raise HTTPException(status_code=400, detail="Start must be before end")
        time_window = TimeWindow(start=start, end=end)

    filter_criteria = AnomalyStoreFilter(
        anomaly_types=anomaly_types,
        time_window=time_window,
        min_confidence=min_confidence,
        limit=limit,
    )

    # Query
    records = store.query(filter_criteria)
    total_count = store.count(filter_criteria)

    return AnomalyListResponse(
        anomalies=[AnomalyRecordResponse.from_record(r) for r in records],
        total_count=total_count,
        returned_count=len(records),
    )


@router.get("/stats", response_model=StorageStatsResponse)
async def get_statistics(
    store: AnomalyStore = Depends(get_store),
) -> StorageStatsResponse:
    """Get storage statistics.
    
    Returns statistics about the anomaly store including
    total record count and breakdown by type.
    """
    stats = store.get_statistics()
    return StorageStatsResponse(
        total_records=stats.get("total_records", 0),
        records_by_type=stats.get("records_by_type", {}),
        storage_type=stats.get("storage_type", "unknown"),
        is_persistent=stats.get("is_persistent", False),
    )


@router.get("/trust-signals/current", response_model=TrustSignalResponse)
async def get_trust_signal(
    time_window_hours: int = Query(default=24, ge=1, le=720),
    store: AnomalyStore = Depends(get_store),
) -> TrustSignalResponse:
    """Get the current trust signal based on recent anomalies.
    
    Computes an aggregated trust signal from anomalies detected
    within the specified time window.
    
    Note: Trust signals are advisory metadata only. They do NOT
    trigger any actions or enforcement.
    """
    settings = get_settings()

    # Define time window
    end = datetime.utcnow()
    from datetime import timedelta
    start = end - timedelta(hours=time_window_hours)
    time_window = TimeWindow(start=start, end=end)

    # Count anomalies by type
    filter_criteria = AnomalyStoreFilter(time_window=time_window)
    records = store.query(filter_criteria)

    counts = {"cost": 0, "quality": 0, "latency": 0, "policy": 0}
    for record in records:
        type_name = record.anomaly_type.value
        if type_name in counts:
            counts[type_name] += 1

    anomaly_count = AnomalyCount(
        cost=counts["cost"],
        quality=counts["quality"],
        latency=counts["latency"],
        policy=counts["policy"],
    )

    # Compute trust level
    trust_level = compute_trust_level(anomaly_count)

    # Compute confidence based on sample size
    total_records = store.count()
    confidence = min(1.0, total_records / 100) if total_records > 0 else 0.5

    signal = TrustSignal(
        trust_level=trust_level,
        anomaly_counts=anomaly_count,
        confidence=confidence,
        computed_at=datetime.utcnow(),
        algorithm_version=settings.algorithm_version,
        time_window_hours=time_window_hours,
    )

    return TrustSignalResponse(
        trust_level=signal.trust_level.value,
        anomaly_counts={
            "cost": signal.anomaly_counts.cost,
            "quality": signal.anomaly_counts.quality,
            "latency": signal.anomaly_counts.latency,
            "policy": signal.anomaly_counts.policy,
            "total": signal.anomaly_counts.total,
        },
        confidence=signal.confidence,
        computed_at=signal.computed_at.isoformat(),
        algorithm_version=signal.algorithm_version,
        time_window_hours=signal.time_window_hours,
        has_anomalies=signal.has_anomalies,
        severity_score=signal.severity_score,
    )


@router.post("/analyze/replay", response_model=ReplayResponse)
async def replay_analysis(
    request: ReplayRequest,
    store: AnomalyStore = Depends(get_store),
) -> ReplayResponse:
    """Replay anomaly analysis for a specific time window.
    
    Retrieves all anomaly records within the specified time window
    for historical analysis and auditing.
    
    Note: This is a read-only operation that does not modify any data
    or trigger any actions.
    """
    if request.time_window.start >= request.time_window.end:
        raise HTTPException(status_code=400, detail="Start must be before end")

    time_window = TimeWindow(
        start=request.time_window.start,
        end=request.time_window.end,
    )

    # Get records in chronological order
    records = store.replay(time_window)

    # Compute summary
    summary: dict[str, int] = {}
    for record in records:
        type_name = record.anomaly_type.value
        summary[type_name] = summary.get(type_name, 0) + 1

    return ReplayResponse(
        time_window={
            "start": time_window.start.isoformat(),
            "end": time_window.end.isoformat(),
        },
        anomalies=[AnomalyRecordResponse.from_record(r) for r in records],
        count=len(records),
        summary=summary,
    )


@router.get("/{record_id}", response_model=AnomalyRecordResponse)
async def get_anomaly(
    record_id: str,
    store: AnomalyStore = Depends(get_store),
) -> AnomalyRecordResponse:
    """Get a specific anomaly record by ID.
    
    Args:
        record_id: UUID of the anomaly record
        
    Returns:
        The anomaly record if found
        
    Raises:
        404: If record not found
    """
    # Validate UUID format
    try:
        UUID(record_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid record ID format")

    record = store.get_by_id(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Anomaly record not found")

    return AnomalyRecordResponse.from_record(record)

