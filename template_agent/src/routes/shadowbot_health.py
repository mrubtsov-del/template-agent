"""Shadowbot health endpoint.

# /v1/shadowbot-agent/health

Required by Red Hat AI Studio / Shadowbot platform validation. Exposes the
same payload as the generic /health endpoint but at the path the validator
expects.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


# /v1/shadowbot-agent/health
@router.get("/v1/shadowbot-agent/health")
async def shadowbot_health() -> JSONResponse:
    """Return the agent health payload for the Shadowbot validator."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Snowflake Agent",
            "version": "1.0.0",
        }
    )
