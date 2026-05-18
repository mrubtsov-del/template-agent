"""Shadowbot V1 conversation messages handler.

# GET /api/v1/conversations/{conversation_id}/messages

DUMMY IMPLEMENTATION: returns [] until message history persistence is wired.
"""

from typing import List, Optional

from shadowbot_agent_api import (
    ConversationMessage,
    UserContext,
    get_messages_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.routes.common import logger, snowflake_auth_present


@get_messages_handler()
@require_auth
async def handle_get_messages(
    conversation_id: str,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> List[ConversationMessage]:
    """Return messages for a conversation (stub: empty list)."""
    user_id = (user.email if user else None) or (user.sub if user else None) or "anonymous"
    logger.warning(
        "[V1] DUMMY IMPLEMENTATION - Get messages returns empty list. "
        "Wire up persistence to enable message history.",
        conversation_id=conversation_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )
    return []
