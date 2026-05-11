"""Shared helpers for Shadowbot V1 handlers.

Keeps a single source of truth for the logger and user-id resolution so each
endpoint module stays minimal and focused on its own route.
"""

from typing import Optional

from shadowbot_agent_api import ConversationRequest, UserContext

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def resolve_user_id(
    request: ConversationRequest, user: Optional[UserContext]
) -> str:
    """Pick the best identifier we have for the caller.

    Preference order: authenticated email > user-supplied email > "anonymous".
    """
    if user and user.email:
        return user.email
    if request.userInfo and request.userInfo.userEmail:
        return request.userInfo.userEmail
    return "anonymous"
