"""
LLMOps Data Reader

Read-only access to LLMOps data for anomaly detection.

DESIGN RULES (NON-NEGOTIABLE):
- READ-ONLY access only (HTTP GET)
- Pull-based: Anomaly service pulls from LLMOps
- Fail-open: return empty on ANY error
- Timeout ≤1s, no retries, no backoff
- Never writes back to LLMOps

This service builds trust, not control.
Anomaly detection is observational intelligence —
not operational authority.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
import json
from pathlib import Path

import requests

from anomaly.config.settings import get_settings


class DataWindow:
    """Time window for data queries."""
    
    def __init__(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.limit = limit


class LLMOpsReader(ABC):
    """
    Abstract base class for LLMOps data readers.
    
    All readers are READ-ONLY.
    No writes, no callbacks, no side effects.
    """
    
    @abstractmethod
    def read_traces(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read trace records from LLMOps for latency anomalies."""
        ...
    
    @abstractmethod
    def read_costs(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read cost records from LLMOps for cost spike detection."""
        ...
    
    @abstractmethod
    def read_evaluations(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read evaluation records from LLMOps for quality drift."""
        ...
    
    @abstractmethod
    def read_policies(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read policy outcome records from LLMOps for policy instability."""
        ...
    
    @abstractmethod
    def read_slas(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read SLA records from LLMOps for tier-based baselines."""
        ...


class LLMOpsAPIReader(LLMOpsReader):
    """
    Read LLMOps data via HTTP API.
    
    Pulls data from LLMOps query endpoints.
    
    DESIGN RULES (ENFORCED):
    - HTTP GET only (read-only)
    - Short timeout (≤1s) for fail-fast
    - Fail-open: return empty on any error
    - Respects LLMOPS_ENABLED setting
    - NO POST/PUT/PATCH/DELETE
    - NO retries, NO backoff
    """
    
    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = base_url or settings.llmops_base_url
    
    def _build_params(
        self, window: DataWindow | None, limit: int = 1000
    ) -> dict[str, str]:
        """Build query parameters from DataWindow."""
        params: dict[str, str] = {"limit": str(limit)}
        if window:
            if window.start_time:
                params["start_time"] = window.start_time.isoformat()
            if window.end_time:
                params["end_time"] = window.end_time.isoformat()
            if window.limit:
                params["limit"] = str(window.limit)
        return params
    
    def _fetch(
        self, endpoint: str, window: DataWindow | None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """
        Fetch data from LLMOps API endpoint.
        
        Returns empty list on any error (fail-open).
        This is MANDATORY behavior - no exceptions may propagate.
        """
        settings = get_settings()
        
        # Check if LLMOps integration is disabled
        if not settings.llmops_enabled:
            return []
        
        url = f"{self.base_url}{endpoint}"
        params = self._build_params(window, limit)
        
        try:
            # Use configurable timeout (convert ms to seconds)
            # No retries - fail fast, fail open
            timeout_seconds = settings.llmops_timeout_ms / 1000.0
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            
            # Handle wrapped response format from LLMOps API
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
        except requests.RequestException:
            # Fail-open: return empty on any network/HTTP error
            # This includes: ConnectionError, Timeout, HTTPError, etc.
            return []
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fail-open: return empty on invalid payload
            return []
    
    def read_traces(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read traces from LLMOps API for latency anomalies."""
        return self._fetch("/query/traces", window)
    
    def read_costs(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read costs from LLMOps API for cost spike detection."""
        return self._fetch("/query/costs", window)
    
    def read_evaluations(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read evaluations from LLMOps API for quality drift."""
        return self._fetch("/query/evaluations", window)
    
    def read_policies(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read policy outcomes from LLMOps API for policy instability."""
        return self._fetch("/query/policies", window)
    
    def read_slas(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read SLAs from LLMOps API for tier-based baselines."""
        return self._fetch("/query/slas", window)


class LLMOpsFileReader(LLMOpsReader):
    """
    Read LLMOps data from exported JSONL files.
    
    Reads from local filesystem for offline processing.
    This is useful for replay analysis and testing.
    """
    
    def __init__(self, data_dir: str | None = None):
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.llmops_data_dir or "data")
    
    def _read_jsonl(
        self, filename: str, window: DataWindow | None
    ) -> list[dict[str, Any]]:
        """Read records from a JSONL file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return []
        
        records: list[dict[str, Any]] = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        records.append(record)
        except (OSError, json.JSONDecodeError):
            # Fail-open: return empty on any file/parse error
            return []
        
        # Apply simple time filtering if window specified
        if window and window.start_time:
            records = [
                r for r in records
                if r.get("ingested_at") and r["ingested_at"] >= window.start_time.isoformat()
            ]
        if window and window.end_time:
            records = [
                r for r in records
                if r.get("ingested_at") and r["ingested_at"] <= window.end_time.isoformat()
            ]
        
        # Apply record count limit
        if window and window.limit:
            records = records[:window.limit]
        
        return records
    
    def read_traces(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read traces from JSONL file."""
        return self._read_jsonl("traces.jsonl", window)
    
    def read_costs(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read costs from JSONL file."""
        return self._read_jsonl("costs.jsonl", window)
    
    def read_evaluations(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read evaluations from JSONL file."""
        return self._read_jsonl("evaluations.jsonl", window)
    
    def read_policies(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read policy outcomes from JSONL file."""
        return self._read_jsonl("policies.jsonl", window)
    
    def read_slas(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        """Read SLAs from JSONL file."""
        return self._read_jsonl("slas.jsonl", window)


class InMemoryReader(LLMOpsReader):
    """
    In-memory reader for testing.
    
    Holds data directly in memory.
    Returns deep copies to ensure immutability.
    """
    
    def __init__(
        self,
        traces: list[dict[str, Any]] | None = None,
        costs: list[dict[str, Any]] | None = None,
        evaluations: list[dict[str, Any]] | None = None,
        policies: list[dict[str, Any]] | None = None,
        slas: list[dict[str, Any]] | None = None,
    ):
        import copy
        # Store deep copies to prevent external modification
        self._traces = copy.deepcopy(traces) if traces else []
        self._costs = copy.deepcopy(costs) if costs else []
        self._evaluations = copy.deepcopy(evaluations) if evaluations else []
        self._policies = copy.deepcopy(policies) if policies else []
        self._slas = copy.deepcopy(slas) if slas else []
    
    def _deep_copy(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a deep copy of the data."""
        import copy
        return copy.deepcopy(data)
    
    def read_traces(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        return self._deep_copy(self._traces)
    
    def read_costs(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        return self._deep_copy(self._costs)
    
    def read_evaluations(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        return self._deep_copy(self._evaluations)
    
    def read_policies(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        return self._deep_copy(self._policies)
    
    def read_slas(self, window: DataWindow | None = None) -> list[dict[str, Any]]:
        return self._deep_copy(self._slas)


def get_reader() -> LLMOpsReader:
    """
    Get the appropriate reader based on configuration.
    
    Prefers file-based if LLMOPS_DATA_DIR is set, otherwise uses API.
    """
    settings = get_settings()
    
    if settings.llmops_data_dir:
        return LLMOpsFileReader(settings.llmops_data_dir)
    else:
        return LLMOpsAPIReader(settings.llmops_base_url)
