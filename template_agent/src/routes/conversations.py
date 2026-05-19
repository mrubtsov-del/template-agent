"""Shadowbot V1 conversation list handler.

# GET /api/v1/conversations
"""

from typing import List, Optional

from shadowbot_agent_api import (
    Conversation,
    UserContext,
    get_conversations_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.core.conversation_history import list_conversations_for_user
from template_agent.src.routes.common import logger, resolve_user_label, snowflake_auth_present


@get_conversations_handler()
@require_auth
async def handle_get_conversations(
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> List[Conversation]:
    """Return the user's conversations for V1 clients."""
    user_id = resolve_user_label(user)
    logger.info(
        "[V1] Get conversations",
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    result = list_conversations_for_user(user_id, page=1, page_size=100, user=user)
    return [
        Conversation(
            conversationID=conv.conversation_id,
            title=conv.title,
            platform=conv.platform,
            threadTs=conv.custom_fields.get("threadTs"),
            channelId=conv.custom_fields.get("channelId"),
        )
        for conv in result.items
    ]
