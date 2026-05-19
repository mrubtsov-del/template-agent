"""Shadowbot V1 conversation messages handler.

# GET /api/v1/conversations/{conversation_id}/messages
"""

from typing import List, Optional

from shadowbot_agent_api import (
    ConversationMessage,
    UserContext,
    get_messages_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.core.conversation_history import list_messages_for_conversation
from template_agent.src.routes.common import logger, resolve_user_label, snowflake_auth_present


@get_messages_handler()
@require_auth
async def handle_get_messages(
    conversation_id: str,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> List[ConversationMessage]:
    """Return messages for a conversation (V1 shape)."""
    user_id = resolve_user_label(user)
    logger.info(
        "[V1] Get messages",
        conversation_id=conversation_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    result = list_messages_for_conversation(
        user_id,
        conversation_id,
        page=1,
        page_size=500,
    )
    return [
        ConversationMessage(
            conversationID=conversation_id,
            type=msg.type,
            content=msg.content,
            comment=None,
            reaction=None,
            images=msg.images,
        )
        for msg in result.items
    ]
