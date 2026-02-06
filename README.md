# GenAI Anomaly Service

Trust and anomaly detection layer for the governed GenAI platform. This service detects unexpected deviations between predicted and actual system behavior, producing trust signals and anomaly reports.

> **⚠️ IMPORTANT**: This service is **advisory only** — it detects anomalies but does **NOT** control, enforce, or remediate anything.

## Overview

The GenAI Anomaly Service is a foundational infrastructure component that:

- Consumes historical actual outcomes from LLMOps
- Consumes historical predictions from genai-ml-service
- Compares predicted vs actual signals across time
- Detects anomalies in cost, quality, latency, and policy outcomes
- Computes drift metrics and deviation scores
- Emits anomaly records and trust signals
- Supports replay-based anomaly analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GenAI Anomaly Service                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │   LLMOps Data    │     │   ML Predictions │                  │
│  │  (Read-Only)     │     │   (Read-Only)    │                  │
│  └────────┬─────────┘     └────────┬─────────┘                  │
│           │                        │                             │
│           └──────────┬─────────────┘                             │
│                      ▼                                           │
│           ┌──────────────────────┐                              │
│           │  Baseline Calculator │                              │
│           │  (Statistical)       │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                           │
│  ┌────────────────────────────────────────────────┐             │
│  │              Anomaly Detectors                  │             │
│  │  ┌──────┐ ┌─────────┐ ┌─────────┐ ┌────────┐  │             │
│  │  │ Cost │ │ Quality │ │ Latency │ │ Policy │  │             │
│  │  └──────┘ └─────────┘ └─────────┘ └────────┘  │             │
│  └────────────────────────┬───────────────────────┘             │
│                           ▼                                      │
│           ┌────────────────────────┐                            │
│           │  Append-Only Store     │                            │
│           │  (Immutable Records)   │                            │
│           └────────────────────────┘                            │
│                           │                                      │
│                           ▼                                      │
│           ┌────────────────────────┐                            │
│           │      FastAPI API       │                            │
│           │  (Query & Replay)      │                            │
│           └────────────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
genai-anomaly-service/
├── anomaly/
│   ├── config/              # Configuration settings
│   │   └── settings.py      # Algorithm versions, thresholds
│   ├── models/              # Domain models
│   │   ├── anomaly_record.py
│   │   ├── baseline.py
│   │   ├── input_data.py
│   │   └── trust_signal.py
│   ├── baselines/           # Baseline computation
│   │   ├── interface.py
│   │   └── statistical.py
│   ├── detectors/           # Anomaly detection algorithms
│   │   ├── interface.py
│   │   ├── cost_detector.py
│   │   ├── quality_detector.py
│   │   ├── latency_detector.py
│   │   ├── policy_detector.py
│   │   └── registry.py
│   └── store/               # Persistence layer
│       ├── interface.py
│       ├── memory_store.py
│       └── file_store.py
├── app/
│   ├── api/                 # API endpoints
│   │   ├── health.py
│   │   └── routes.py
│   └── main.py              # FastAPI application
├── tests/                   # Test suite
│   ├── test_detectors.py
│   ├── test_baselines.py
│   ├── test_store.py
│   └── test_api.py
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Clone the repository
cd genai-anomaly-service

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Usage

### Running the Service

```bash
# Start the service
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/live` | GET | Liveness probe |
| `/api/v1/anomalies` | GET | List anomalies with filters |
| `/api/v1/anomalies/{id}` | GET | Get specific anomaly |
| `/api/v1/anomalies/analyze/replay` | POST | Replay analysis |
| `/api/v1/anomalies/trust-signals/current` | GET | Current trust signal |
| `/api/v1/anomalies/stats` | GET | Storage statistics |

### Query Examples

```bash
# Health check
curl http://localhost:8000/health

# List all anomalies
curl http://localhost:8000/api/v1/anomalies

# Filter by type
curl "http://localhost:8000/api/v1/anomalies?anomaly_type=cost"

# Filter by confidence
curl "http://localhost:8000/api/v1/anomalies?min_confidence=0.8"

# Replay analysis
curl -X POST http://localhost:8000/api/v1/anomalies/analyze/replay \
  -H "Content-Type: application/json" \
  -d '{"time_window": {"start": "2026-02-01T00:00:00Z", "end": "2026-02-06T23:59:59Z"}}'

# Get trust signal
curl http://localhost:8000/api/v1/anomalies/trust-signals/current
```

## Anomaly Types

| Type | Description | Detection Method |
|------|-------------|------------------|
| **Cost** | Token/API cost deviations | Z-score based (default: 2 std) |
| **Quality** | Response quality anomalies | Z-score + percentile threshold |
| **Latency** | Response time anomalies | P99 violation + z-score |
| **Policy** | Policy outcome deviations | Binary mismatch detection |

## Anomaly Record Format

Each anomaly record includes:

```json
{
  "record_id": "uuid",
  "anomaly_type": "cost|quality|latency|policy",
  "observed_value": 150.0,
  "expected_value": 100.0,
  "deviation_score": 2.5,
  "confidence": 0.85,
  "algorithm_version": "1.0.0",
  "time_window": {
    "start": "2026-02-06T00:00:00Z",
    "end": "2026-02-06T23:59:59Z"
  },
  "timestamp": "2026-02-06T12:00:00Z",
  "metric_name": "estimated_cost_usd",
  "source_id": "trace_12345"
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_detectors.py -v
pytest tests/test_baselines.py -v
pytest tests/test_store.py -v
pytest tests/test_api.py -v

# Run determinism tests
pytest tests/ -v -k "deterministic"
```

## Architectural Constraints

### This Service MUST:
- ✅ Be deterministic (same inputs → same outputs)
- ✅ Use append-only storage (immutable records)
- ✅ Version all detection algorithms
- ✅ Be explainable (baseline vs deviation)
- ✅ Support replay-based analysis

### This Service MUST NOT:
- ❌ Invoke agents or LLMs
- ❌ Route requests
- ❌ Enforce policies
- ❌ Block or throttle traffic
- ❌ Influence the planner
- ❌ Provide real-time control
- ❌ Auto-remediate anomalies
- ❌ Trigger retraining

## Configuration

Key settings in `anomaly/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `algorithm_version` | `1.0.0` | Algorithm version for traceability |
| `z_score_threshold` | `2.0` | Standard deviations for anomaly |
| `min_confidence` | `0.5` | Minimum confidence to report |
| `storage_path` | `./data/anomalies` | File storage location |

## License

MIT
