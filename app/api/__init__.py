"""API module."""

from app.api.health import router as health_router
from app.api.routes import router as anomaly_router

__all__ = ["anomaly_router", "health_router"]
