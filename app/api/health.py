"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "genai-anomaly-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness probe endpoint.
    
    Used by orchestration systems to determine if the service
    is ready to receive traffic.
    
    Returns:
        Readiness status
    """
    return {
        "status": "ready",
        "service": "genai-anomaly-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/live")
async def liveness_check() -> dict:
    """Liveness probe endpoint.
    
    Used by orchestration systems to determine if the service
    is alive and should not be restarted.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "service": "genai-anomaly-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
