"""GenAI Anomaly Service - FastAPI application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from anomaly.config import get_settings
from anomaly.store import FileAnomalyStore, MemoryAnomalyStore
from app.api import anomaly_router, health_router
from app.api.routes import set_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.
    
    Initializes resources on startup and cleans up on shutdown.
    """
    settings = get_settings()

    # Initialize storage
    if settings.enable_file_storage:
        store = FileAnomalyStore(settings.storage_path)
    else:
        store = MemoryAnomalyStore()

    set_store(store)

    yield

    # Cleanup (if needed)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="GenAI Anomaly Service",
        description=(
            "Trust and anomaly detection layer for governed GenAI platform. "
            "This service detects unexpected deviations between predicted and actual "
            "system behavior, producing trust signals and anomaly reports. "
            "**IMPORTANT: This service is advisory only - it does NOT control, "
            "enforce, or remediate anything.**"
        ),
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST"],  # Limited methods - read-heavy API
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(anomaly_router, prefix=settings.api_prefix)

    return app


# Application instance
app = create_app()


@app.get("/")
async def root() -> dict:
    """Root endpoint with service information."""
    settings = get_settings()
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "description": "Trust and anomaly detection layer",
        "status": "operational",
        "note": "This service is advisory only - it does NOT control or enforce anything",
    }
