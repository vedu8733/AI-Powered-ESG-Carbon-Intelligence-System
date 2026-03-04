"""
Health check endpoints for production monitoring.
Used by load balancers, Kubernetes probes, and monitoring systems.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, status

from app import __version__

router = APIRouter()


@router.get(
    "",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns service health status. Use for liveness probes.",
)
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": __version__,
        "service": "esg-carbon-intelligence",
    }


@router.get(
    "/ready",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Returns readiness status. Use for readiness probes.",
)
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - extend with DB/vector store checks when needed."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
