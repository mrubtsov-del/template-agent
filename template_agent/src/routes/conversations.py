"""Shadowbot V1 conversation list handler.

# GET /api/v1/conversations

DUMMY IMPLEMENTATION: returns [] until thread metadata persistence is wired.
"""

from typing import List, Optional

from shadowbot_agent_api import (
    Conversation,
    UserContext,
    get_conversations_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.routes.common import logger, snowflake_auth_present


@get_conversations_handler()
@require_auth
async def handle_get_conversations(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> List[Conversation]:
    """Return the user's conversations (stub: empty list)."""
    user_id = (user.email if user else None) or (user.sub if user else None) or "anonymous"
    logger.warning(
        "[V1] DUMMY IMPLEMENTATION - List conversations returns empty list. "
        "Wire up persistence to enable history.",
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )
    return []
