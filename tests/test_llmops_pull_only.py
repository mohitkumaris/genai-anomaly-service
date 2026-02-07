"""
LLMOps Integration Compliance Tests

Tests proving architectural invariants:
1. Pull-only: No HTTP methods other than GET
2. Fail-open: LLMOps unavailable → empty result
3. Determinism: Same input → same output
4. No execution coupling: No orchestrator/agent imports
5. Read-only: No mutation methods

CORE INVARIANT: The Anomaly Service observes the platform.
It NEVER influences execution.
"""

import ast
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from anomaly.features.llmops_reader import (
    LLMOpsReader,
    LLMOpsAPIReader,
    LLMOpsFileReader,
    InMemoryReader,
    DataWindow,
    get_reader,
)
from anomaly.config import settings as settings_module


class TestPullOnlyCompliance:
    """Verify Anomaly service only uses HTTP GET (read-only)."""
    
    def test_llmops_reader_uses_get_only(self):
        """Verify LLMOpsAPIReader only uses requests.get."""
        reader_path = Path(__file__).parent.parent / "anomaly" / "features" / "llmops_reader.py"
        source = reader_path.read_text()
        
        # Parse AST to find all method calls
        tree = ast.parse(source)
        
        forbidden_methods = ["post", "put", "patch", "delete"]
        found_forbidden = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_methods:
                    found_forbidden.append(node.attr)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in forbidden_methods:
                        found_forbidden.append(node.func.attr)
        
        assert len(found_forbidden) == 0, (
            f"Found forbidden HTTP methods in llmops_reader.py: {found_forbidden}. "
            "Only GET is allowed for read-only access."
        )
    
    def test_no_write_endpoints_in_reader(self):
        """Verify no POST/PUT/PATCH/DELETE URLs or methods defined."""
        reader_path = Path(__file__).parent.parent / "anomaly" / "features" / "llmops_reader.py"
        source = reader_path.read_text()
        
        # Check for any indication of write operations
        forbidden_patterns = [
            "requests.post",
            "requests.put",
            "requests.patch",
            "requests.delete",
            "/ingest",  # LLMOps ingest endpoints
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source.lower(), (
                f"Found forbidden pattern '{pattern}' in llmops_reader.py. "
                "Anomaly service must be read-only."
            )
    
    def test_no_callback_or_webhook_patterns(self):
        """Verify no callback or webhook patterns exist."""
        reader_path = Path(__file__).parent.parent / "anomaly" / "features" / "llmops_reader.py"
        source = reader_path.read_text().lower()
        
        forbidden_patterns = ["webhook", "callback", "notify", "push"]
        
        for pattern in forbidden_patterns:
            # Allow in comments/docstrings but not as function names
            lines_with_pattern = [
                line for line in source.split("\n")
                if pattern in line and "def " in line
            ]
            assert len(lines_with_pattern) == 0, (
                f"Found forbidden pattern '{pattern}' as function name. "
                "Anomaly service must be pull-only."
            )


class TestFailOpenCompliance:
    """Verify Anomaly service fails open when LLMOps is unavailable."""
    
    def test_api_reader_returns_empty_on_connection_error(self):
        """Test that connection errors return empty list, not crash."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()
            
            reader = LLMOpsAPIReader("http://nonexistent:9999")
            
            # All methods should return empty list, not raise
            assert reader.read_traces() == []
            assert reader.read_costs() == []
            assert reader.read_evaluations() == []
            assert reader.read_policies() == []
            assert reader.read_slas() == []
    
    def test_api_reader_returns_empty_on_timeout(self):
        """Test that timeout returns empty list, not crash."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout()
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
            assert reader.read_costs() == []
    
    def test_api_reader_returns_empty_on_http_error(self):
        """Test that HTTP errors (4xx/5xx) return empty list, not crash."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            import requests
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
    
    def test_api_reader_returns_empty_on_invalid_json(self):
        """Test that invalid JSON returns empty list, not crash."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            import json
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
    
    def test_file_reader_returns_empty_on_missing_file(self):
        """Test that missing files return empty list."""
        reader = LLMOpsFileReader("/nonexistent/directory")
        
        assert reader.read_traces() == []
        assert reader.read_costs() == []


class TestLLMOpsDisabledMode:
    """Verify Anomaly service works correctly when LLMOPS_ENABLED=false."""
    
    def setup_method(self):
        """Save original settings."""
        self.original_settings = settings_module._settings
    
    def teardown_method(self):
        """Restore original settings."""
        settings_module._settings = self.original_settings
        os.environ.pop("LLMOPS_ENABLED", None)
    
    def test_api_reader_returns_empty_when_disabled(self):
        """Test that disabled LLMOps returns empty without making requests."""
        settings_module._settings = None
        os.environ["LLMOPS_ENABLED"] = "false"
        
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            result = reader.read_traces()
            
            # Should return empty without calling requests
            assert result == []
            mock_get.assert_not_called()
    
    def test_all_methods_empty_when_disabled(self):
        """Test all read methods return empty when disabled."""
        settings_module._settings = None
        os.environ["LLMOPS_ENABLED"] = "false"
        
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            assert reader.read_traces() == []
            assert reader.read_costs() == []
            assert reader.read_evaluations() == []
            assert reader.read_policies() == []
            assert reader.read_slas() == []
            
            mock_get.assert_not_called()


class TestTimeoutConfiguration:
    """Verify timeout is configurable and respects settings."""
    
    def setup_method(self):
        """Save original settings."""
        self.original_settings = settings_module._settings
    
    def teardown_method(self):
        """Restore original settings."""
        settings_module._settings = self.original_settings
        os.environ.pop("LLMOPS_TIMEOUT_MS", None)
    
    def test_uses_configured_timeout(self):
        """Test that reader uses LLMOPS_TIMEOUT_MS setting."""
        settings_module._settings = None
        os.environ["LLMOPS_TIMEOUT_MS"] = "500"  # 500ms
        
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            reader.read_traces()
            
            # Verify timeout was set to 0.5 seconds
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["timeout"] == 0.5
    
    def test_default_timeout_is_one_second(self):
        """Test default timeout is 1 second (1000ms)."""
        settings_module._settings = None
        os.environ.pop("LLMOPS_TIMEOUT_MS", None)
        
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            reader.read_traces()
            
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["timeout"] == 1.0


class TestDeterminismCompliance:
    """Verify anomaly detection outputs are deterministic."""
    
    def test_same_input_same_output(self):
        """Test that identical input produces identical output."""
        traces = [
            {"trace_id": "123", "latency_ms": 150, "agent_name": "general"},
            {"trace_id": "456", "latency_ms": 200, "agent_name": "retrieval"},
        ]
        
        reader = InMemoryReader(traces=traces)
        
        # Run read twice
        result1 = reader.read_traces()
        result2 = reader.read_traces()
        
        # Must be identical
        assert result1 == result2
    
    def test_all_readers_deterministic(self):
        """Test all data types return deterministic results."""
        test_data = {"key": "value", "number": 42}
        
        reader = InMemoryReader(
            traces=[test_data],
            costs=[test_data],
            evaluations=[test_data],
            policies=[test_data],
            slas=[test_data],
        )
        
        # Each method should return identical results on repeated calls
        assert reader.read_traces() == reader.read_traces()
        assert reader.read_costs() == reader.read_costs()
        assert reader.read_evaluations() == reader.read_evaluations()
        assert reader.read_policies() == reader.read_policies()
        assert reader.read_slas() == reader.read_slas()


class TestNoExecutionCoupling:
    """Verify Anomaly service has no coupling to execution/orchestrator."""
    
    def test_no_orchestrator_imports(self):
        """Verify no imports from orchestrator in anomaly codebase."""
        anomaly_dir = Path(__file__).parent.parent / "anomaly"
        
        forbidden_imports = [
            "orchestrator",
            "langchain",
            "openai",
            "agent_orchestrator",
            "genai_agent_orchestrator",
            "planner",
            "enforcement",
        ]
        
        violations = []
        
        for py_file in anomaly_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text()
            
            for forbidden in forbidden_imports:
                if f"import {forbidden}" in source or f"from {forbidden}" in source:
                    violations.append(f"{py_file.name}: imports {forbidden}")
        
        assert len(violations) == 0, (
            f"Found forbidden imports in anomaly codebase: {violations}. "
            "Anomaly service must not import orchestrator, planner, or LLM SDKs."
        )
    
    def test_no_llm_sdk_dependencies(self):
        """Verify anomaly features don't depend on LLM SDKs."""
        features_dir = Path(__file__).parent.parent / "anomaly" / "features"
        
        if not features_dir.exists():
            pytest.skip("Features directory not found")
        
        forbidden = ["langchain", "openai", "anthropic", "azure.openai"]
        
        for py_file in features_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text()
            
            for lib in forbidden:
                assert lib not in source, (
                    f"Found forbidden library '{lib}' in {py_file.name}. "
                    "Anomaly features must not use LLM SDKs."
                )
    
    def test_no_enforcement_or_remediation(self):
        """Verify no enforcement or remediation logic exists."""
        features_dir = Path(__file__).parent.parent / "anomaly" / "features"
        
        if not features_dir.exists():
            pytest.skip("Features directory not found")
        
        forbidden_patterns = [
            "enforce",
            "remediate",
            "block",
            "throttle",
            "route",
            "redirect",
        ]
        
        for py_file in features_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            source = py_file.read_text().lower()
            
            for pattern in forbidden_patterns:
                # Allow in comments, check for function definitions
                lines_with_pattern = [
                    line for line in source.split("\n")
                    if pattern in line and "def " in line
                ]
                assert len(lines_with_pattern) == 0, (
                    f"Found forbidden pattern '{pattern}' as function in {py_file.name}. "
                    "Anomaly service must not contain enforcement logic."
                )


class TestReadOnlyCompliance:
    """Verify all data access is read-only."""
    
    def test_in_memory_reader_returns_copies(self):
        """Test InMemoryReader returns copies, not references."""
        original = [{"key": "value"}]
        reader = InMemoryReader(traces=original)
        
        result = reader.read_traces()
        
        # Modifying result should not affect original
        result.append({"new": "data"})
        
        assert len(original) == 1
        assert len(reader.read_traces()) == 1
    
    def test_modifications_dont_affect_source(self):
        """Test that modifying returned data doesn't affect source."""
        original_data = [{"id": 1, "value": "original"}]
        reader = InMemoryReader(
            traces=original_data,
            costs=original_data.copy(),
        )
        
        # Get and modify
        traces = reader.read_traces()
        traces[0]["value"] = "modified"
        
        # Original should be unchanged
        fresh_read = reader.read_traces()
        assert fresh_read[0]["value"] == "original"


class TestWrappedResponseHandling:
    """Test handling of LLMOps wrapped response format."""
    
    def test_handles_wrapped_response(self):
        """Test reader extracts 'data' from wrapped response."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "meta": {"count": 1, "limit": 100, "has_more": False},
                "data": [{"trace_id": "123", "agent_name": "general"}]
            }
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            result = reader.read_traces()
            
            assert len(result) == 1
            assert result[0]["trace_id"] == "123"
    
    def test_handles_raw_list_response(self):
        """Test reader handles raw list response (backward compatibility)."""
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {"trace_id": "123", "agent_name": "general"}
            ]
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            result = reader.read_traces()
            
            assert len(result) == 1
            assert result[0]["trace_id"] == "123"


class TestDataWindowSupport:
    """Test time window and pagination support."""
    
    def test_time_window_params_passed_to_api(self):
        """Test that time window parameters are passed to API."""
        from datetime import datetime
        
        with patch("anomaly.features.llmops_reader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_get.return_value = mock_response
            
            reader = LLMOpsAPIReader("http://localhost:8100")
            
            start = datetime(2026, 2, 1, 0, 0, 0)
            end = datetime(2026, 2, 7, 23, 59, 59)
            window = DataWindow(start_time=start, end_time=end, limit=500)
            
            reader.read_traces(window)
            
            call_args = mock_get.call_args
            params = call_args[1]["params"]
            
            assert "start_time" in params
            assert "end_time" in params
            assert params["limit"] == "500"
