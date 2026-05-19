"""Shadowbot V2 conversation messages handler.

# GET /api/v2/conversations/{conversation_id}/messages
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_messages_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import MessageHistoryResponseV2

from template_agent.src.core.conversation_history import list_messages_for_conversation
from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
)


@get_messages_handler_v2()
@require_auth
async def handle_get_messages_v2(
    conversation_id: str,
    page: int = 1,
    page_size: int = 50,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> MessageHistoryResponseV2:
    """Return paginated message history for a conversation."""
    user_id = resolve_user_label(user)
    logger.info(
        "[V2] Get messages",
        conversation_id=conversation_id,
        user_id=user_id,
        page=page,
        page_size=page_size,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    result = list_messages_for_conversation(
        user_id,
        conversation_id,
        page=page,
        page_size=page_size,
    )
    return MessageHistoryResponseV2(
        messages=result.items,
        conversation_id=conversation_id,
        session_id=result.session_id,
        total_count=result.total_count,
        page=page,
        page_size=page_size,
    )
