"""Health check route for the template agent API.

This module provides health check endpoints to monitor the status
and availability of the template agent service.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


def _health_payload() -> dict[str, str]:
    """Return a stable health payload used by all health routes."""
    return {"status": "healthy", "service": "Snowflake Agent", "version": "1.0.0"}


@router.get("/health")
async def health_check() -> JSONResponse:
    """Perform a health check on the template agent service.

    This endpoint is used to verify that the service is running and
    responding to requests. It returns a simple JSON response indicating
    the service status.

    Returns:
        A JSONResponse containing the service status and name.
    """
    return JSONResponse(content=_health_payload())
