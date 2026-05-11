"""Shadowbot V2 conversation messages handler.

# /api/v2/conversations/{conversation_id}/messages

Returns paginated message history (List[ChatMessageV2]) for a conversation.

DUMMY IMPLEMENTATION: stubbed until the persistence layer is wired up. The
expected backing store is the `rag_metadata` table from the V2 migration
guide; once present, replace the empty list with a real query.
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_messages_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import MessageHistoryResponseV2

from template_agent.src.routes.common import logger


# /api/v2/conversations/{conversation_id}/messages
@get_messages_handler_v2()
@require_auth
async def handle_get_messages_v2(
    conversation_id: str,
    page: int = 1,
    page_size: int = 50,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> MessageHistoryResponseV2:
    """Return messages for a conversation (stub: empty list)."""
    user_email = (user.email if user else None) or "anonymous"
    logger.warning(
        "[V2] DUMMY IMPLEMENTATION - Get messages returns empty list. "
        "Wire up persistence to enable message history.",
        conversation_id=conversation_id,
        user_id=user_email,
        page=page,
        page_size=page_size,
    )
    return MessageHistoryResponseV2(
        messages=[],
        conversation_id=conversation_id,
        session_id=conversation_id,
        total_count=0,
        page=page,
        page_size=page_size,
    )
