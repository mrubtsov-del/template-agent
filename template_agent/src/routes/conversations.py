"""Shadowbot V1 conversation list handler.

# /api/v1/conversations

Returns the list of conversations owned by the authenticated user.
A persistence layer is not yet implemented, so the stub returns [].
"""

from typing import List

from shadowbot_agent_api import (
    Conversation,
    UserContext,
    get_conversations_handler,
    require_auth,
)

from template_agent.src.routes.common import logger


# /api/v1/conversations
@get_conversations_handler()
@require_auth
async def handle_get_conversations(user: UserContext) -> List[Conversation]:
    """Return the user's conversations (stub: empty list)."""
    logger.info("shadowbot.conversations.list", user=user.email or user.sub)
    return []
