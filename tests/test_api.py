"""Integration tests for API endpoints."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from anomaly.models import AnomalyRecord, AnomalyType, TimeWindow
from anomaly.store import MemoryAnomalyStore
from app.api.routes import set_store
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client with a fresh memory store."""
    store = MemoryAnomalyStore()
    set_store(store)
    return TestClient(app)


@pytest.fixture
def populated_store() -> MemoryAnomalyStore:
    """Create a store with some test records."""
    store = MemoryAnomalyStore()
    now = datetime.utcnow()
    time_window = TimeWindow(start=now - timedelta(hours=1), end=now)

    records = [
        AnomalyRecord(
            anomaly_type=AnomalyType.COST,
            observed_value=150.0,
            expected_value=100.0,
            deviation_score=2.5,
            confidence=0.85,
            algorithm_version="1.0.0",
            time_window=time_window,
            metric_name="cost_usd",
            source_id="trace_1",
        ),
        AnomalyRecord(
            anomaly_type=AnomalyType.LATENCY,
            observed_value=500.0,
            expected_value=200.0,
            deviation_score=3.0,
            confidence=0.9,
            algorithm_version="1.0.0",
            time_window=time_window,
            metric_name="latency_ms",
            source_id="trace_2",
        ),
        AnomalyRecord(
            anomaly_type=AnomalyType.QUALITY,
            observed_value=0.6,
            expected_value=0.85,
            deviation_score=2.5,
            confidence=0.75,
            algorithm_version="1.0.0",
            time_window=time_window,
            metric_name="quality_score",
            source_id="trace_3",
        ),
    ]

    for record in records:
        store.append(record)

    return store


@pytest.fixture
def populated_client(populated_store: MemoryAnomalyStore) -> TestClient:
    """Create a test client with a populated store."""
    set_store(populated_store)
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "genai-anomaly-service"

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness endpoint."""
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client: TestClient) -> None:
        """Test root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "genai-anomaly-service"
        assert "advisory only" in data["note"]


class TestAnomalyEndpoints:
    """Tests for anomaly API endpoints."""

    def test_list_anomalies_empty(self, client: TestClient) -> None:
        """Test listing anomalies returns empty list when store is empty."""
        response = client.get("/api/v1/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert data["anomalies"] == []
        assert data["total_count"] == 0

    def test_list_anomalies_with_data(self, populated_client: TestClient) -> None:
        """Test listing anomalies returns records."""
        response = populated_client.get("/api/v1/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert len(data["anomalies"]) == 3
        assert data["total_count"] == 3

    def test_list_anomalies_filter_by_type(self, populated_client: TestClient) -> None:
        """Test filtering anomalies by type."""
        response = populated_client.get("/api/v1/anomalies?anomaly_type=cost")
        assert response.status_code == 200
        data = response.json()
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["anomaly_type"] == "cost"

    def test_list_anomalies_filter_by_confidence(
        self, populated_client: TestClient
    ) -> None:
        """Test filtering anomalies by minimum confidence."""
        response = populated_client.get("/api/v1/anomalies?min_confidence=0.8")
        assert response.status_code == 200
        data = response.json()
        # Should return cost (0.85) and latency (0.9), not quality (0.75)
        assert len(data["anomalies"]) == 2
        assert all(a["confidence"] >= 0.8 for a in data["anomalies"])

    def test_get_anomaly_by_id(self, populated_client: TestClient) -> None:
        """Test getting a specific anomaly by ID."""
        # First get the list to find an ID
        list_response = populated_client.get("/api/v1/anomalies")
        record_id = list_response.json()["anomalies"][0]["record_id"]

        # Now get by ID
        response = populated_client.get(f"/api/v1/anomalies/{record_id}")
        assert response.status_code == 200
        assert response.json()["record_id"] == record_id

    def test_get_anomaly_not_found(self, client: TestClient) -> None:
        """Test getting non-existent anomaly returns 404."""
        fake_id = "12345678-1234-5678-1234-567812345678"
        response = client.get(f"/api/v1/anomalies/{fake_id}")
        assert response.status_code == 404

    def test_get_anomaly_invalid_id(self, client: TestClient) -> None:
        """Test getting anomaly with invalid ID returns 400."""
        response = client.get("/api/v1/anomalies/invalid-id")
        assert response.status_code == 400


class TestReplayEndpoint:
    """Tests for replay analysis endpoint."""

    def test_replay_analysis(self, populated_client: TestClient) -> None:
        """Test replay analysis returns records in time window."""
        now = datetime.utcnow()
        start = (now - timedelta(hours=2)).isoformat()
        end = now.isoformat()

        response = populated_client.post(
            "/api/v1/anomalies/analyze/replay",
            json={"time_window": {"start": start, "end": end}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert "cost" in data["summary"]

    def test_replay_invalid_time_window(self, client: TestClient) -> None:
        """Test replay with invalid time window returns error."""
        now = datetime.utcnow()
        response = client.post(
            "/api/v1/anomalies/analyze/replay",
            json={
                "time_window": {
                    "start": now.isoformat(),
                    "end": (now - timedelta(hours=1)).isoformat(),  # End before start
                }
            },
        )
        assert response.status_code == 400


class TestTrustSignalEndpoint:
    """Tests for trust signal endpoint."""

    def test_get_trust_signal_empty(self, client: TestClient) -> None:
        """Test trust signal with no anomalies shows high trust."""
        response = client.get("/api/v1/anomalies/trust-signals/current")
        assert response.status_code == 200
        data = response.json()
        assert data["trust_level"] == "high"
        assert data["has_anomalies"] is False

    def test_get_trust_signal_with_anomalies(
        self, populated_client: TestClient
    ) -> None:
        """Test trust signal with anomalies shows lower trust."""
        response = populated_client.get("/api/v1/anomalies/trust-signals/current")
        assert response.status_code == 200
        data = response.json()
        assert data["has_anomalies"] is True
        assert data["anomaly_counts"]["total"] == 3


class TestStatisticsEndpoint:
    """Tests for statistics endpoint."""

    def test_get_statistics(self, populated_client: TestClient) -> None:
        """Test getting storage statistics."""
        response = populated_client.get("/api/v1/anomalies/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 3
        assert data["records_by_type"]["cost"] == 1
        assert data["storage_type"] == "memory"
