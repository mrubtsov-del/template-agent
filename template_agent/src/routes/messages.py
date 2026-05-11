"""Shadowbot V1 conversation messages handler.

# /api/v1/conversations/{conversation_id}/messages

Returns the message history for a given conversation. Stubbed until the
persistence layer is wired up.
"""

from typing import List

from shadowbot_agent_api import (
    ConversationMessage,
    UserContext,
    get_messages_handler,
    require_auth,
)

from template_agent.src.routes.common import logger


# /api/v1/conversations/{conversation_id}/messages
@get_messages_handler()
@require_auth
async def handle_get_messages(
    conversation_id: str, user: UserContext
) -> List[ConversationMessage]:
    """Return messages for a conversation (stub: empty list)."""
    logger.info(
        "shadowbot.messages.list",
        conversation_id=conversation_id,
        user=user.email or user.sub,
    )
    return []
