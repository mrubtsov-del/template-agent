"""Exception handlers for converting AgentException to HTTP responses.

This module provides FastAPI exception handlers that transform custom
AgentException instances into properly formatted HTTP responses with
detailed error information.

Separation of concerns:
- exceptions.py: Exception class definitions (domain logic)
- exception_handlers.py: HTTP response transformation (HTTP layer)
"""

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

from .exceptions import AgentException
from shadowbot_agent_api.logger import get_python_logger
from .constants import Constants

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)


async def agent_exception_handler(request: Request, exc: AgentException) -> JSONResponse:
    """
    Handle AgentException by converting it to a structured JSON response.

    This handler preserves all error details (error_code, error_data, etc.)
    instead of converting to a generic HTTPException, providing better
    debugging and client-side error handling.

    Args:
        request: The FastAPI request object
        exc: The AgentException instance to handle

    Returns:
        JSONResponse with structured error information
    """
    logger.warning(
        f"AgentException handled: {exc.error_code or 'NO_CODE'} - {exc.detail} "
        f"(status={exc.status_code}, path={request.url.path})"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers=exc.headers
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all custom exception handlers on a FastAPI application.

    This function should be called on any FastAPI app instance that uses
    the shadowbot-agent-api to ensure proper error handling.

    Args:
        app: FastAPI application instance to register handlers on

    Example:
        ```python
        from fastapi import FastAPI
        from shadowbot_agent_api.exception_handlers import register_exception_handlers

        my_app = FastAPI()
        register_exception_handlers(my_app)
        ```

    Note:
        The default `app` exported by shadowbot-agent-api already has
        handlers registered. Only call this if creating a custom app.
    """
    app.add_exception_handler(AgentException, agent_exception_handler)
    logger.info("Registered AgentException handler for detailed error responses")

    # Future: Can add more custom exception handlers here
    # app.add_exception_handler(OtherException, other_handler)