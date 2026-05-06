"""FastAPI application initialization and shared configuration.

This module contains the FastAPI app factory and shared configuration
that applies to both V1 and V2 APIs.

Separation of concerns:
- app.py: FastAPI app initialization, CORS, middleware, router inclusion
- api.py: V1 endpoint definitions
- api_v2.py: V2 endpoint definitions
- exception_handlers.py: Exception handling logic
"""

from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .constants import Constants
from .exception_handlers import register_exception_handlers
from .auth import setup_auth_from_env, configure_auth
from shadowbot_agent_api.logger import get_python_logger

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)


def create_app(
    title: str = "Shadowbot Agent API",
    description: str = "A pluggable agent API built with FastAPI, allowing custom business logic integration.",
    version: str = "1.0.0",
    enable_cors: bool = True,
    cors_origins: Optional[List[str]] = None,
    enable_auth: bool = True
) -> FastAPI:
    """
    Factory function to create and configure a FastAPI application.

    This function creates a FastAPI app with:
    - CORS middleware (optional)
    - Custom exception handlers for AgentException
    - V1 and V2 API routers
    - Authentication configuration (optional)

    Args:
        title: API title for documentation
        description: API description for documentation
        version: API version string
        enable_cors: Whether to enable CORS middleware
        cors_origins: List of allowed CORS origins. Defaults to localhost origins.
        enable_auth: Whether to initialize authentication from environment

    Returns:
        Configured FastAPI application instance

    Example:
        ```python
        # Create custom app with specific configuration
        from shadowbot_agent_api.app import create_app

        app = create_app(
            title="My Custom Agent API",
            cors_origins=["https://myapp.com"],
            enable_auth=False
        )
        ```
    """
    # Create FastAPI app instance
    app = FastAPI(
        title=title,
        description=description,
        version=version,
    )

    # Configure CORS if enabled
    if enable_cors:
        default_origins = [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000"
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or default_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {cors_origins or default_origins}")

    # Register custom exception handlers (applies to both V1 and V2 APIs)
    register_exception_handlers(app)

    # Initialize authentication if enabled
    if enable_auth:
        auth_config = setup_auth_from_env()
        if auth_config:
            configure_auth(auth_config)
            logger.info(f"Authentication initialized for issuer: {auth_config.issuer}")

    # Import and include routers
    # Note: Imports are here to avoid circular dependencies
    from .api import chat_api_router
    from .api_v2 import chat_api_router_v2

    app.include_router(chat_api_router)
    app.include_router(chat_api_router_v2)

    logger.info("FastAPI app created with V1 and V2 routers")

    return app


def configure_cors(
    app: FastAPI,
    allow_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None
) -> None:
    """
    Configure CORS for a FastAPI app with custom settings.

    This is a utility function for adding CORS to custom apps.
    The default app already has CORS configured.

    Args:
        app: FastAPI application instance
        allow_origins: List of allowed origins. Defaults to common development origins.
        allow_credentials: Whether to allow credentials. Defaults to True.
        allow_methods: List of allowed HTTP methods. Defaults to standard methods.
        allow_headers: List of allowed headers. Defaults to all headers.

    Example:
        ```python
        from fastapi import FastAPI
        from shadowbot_agent_api.app import configure_cors

        my_app = FastAPI()
        configure_cors(my_app, allow_origins=["https://myapp.com"])
        ```
    """
    default_origins = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
    default_methods = ["GET", "POST", "PUT", "DELETE"]
    default_headers = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or default_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods or default_methods,
        allow_headers=allow_headers or default_headers,
    )
    logger.info(f"CORS configured for app: {app.title}")


# Create the default app instance
# This is the main app exported by the package
# NOTE: This is only created when explicitly requested, not at import time
# to avoid conflicts when users have their own FastAPI app
def get_default_app():
    """Get or create the default chat app instance."""
    return create_app()

# For backward compatibility, create default app lazily
default_chat_app = None