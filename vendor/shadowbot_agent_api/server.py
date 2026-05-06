from typing import Optional

from fastapi import FastAPI

from .constants import Constants
from shadowbot_agent_api.logger import get_python_logger
import uvicorn

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)


def run_server(
        target_app: Optional[FastAPI] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs
):
    """
    Runs the FastAPI web server using Uvicorn.
    This function blocks until the server is stopped.

    Args:
        target_app (Optional[FastAPI]): An existing FastAPI application instance
                                        to which the package's routes should be added.
                                        If None, the package's default app will be used.
        host (str): The host to bind to. Defaults to "0.0.0.0".
        port (int): The port to listen on. Defaults to 8000.
        **kwargs: Additional keyword arguments to pass to uvicorn.run().
                  Common args include 'reload=True' for development.
    """
    from .api import default_chat_app, chat_api_router
    
    app_to_run = default_chat_app

    if target_app:
        # If an existing app is provided, include our router into it.
        # Note: We assume the user will then run `target_app` themselves using `uvicorn.run()`
        # outside this function, or they might integrate it into a larger system.
        # This function will NOT start a Uvicorn server if target_app is provided.
        logger.info(f"Integrating chat API routes into provided FastAPI app: {target_app.title}")
        target_app.include_router(chat_api_router, prefix="/conversations")
        logger.info("Chat API routes added. You must run your FastAPI application manually.")
        logger.info(f"Example: uvicorn your_main_app_file:your_app_instance --host {host} --port {port}")
        return

    # If no target_app is provided, run the package's default app
    logger.info(f"Starting Uvicorn server for the package's default app on http://{host}:{port}")
    try:
        uvicorn.run(app_to_run, host=host, port=port, **kwargs)
    except Exception as e:
        logger.error(f"Failed to start server on {host}:{port}. It might be already in use or another error occurred: {e}")
        logger.info("If the port is in use, try changing the 'port' argument or ensure no other server is running.")