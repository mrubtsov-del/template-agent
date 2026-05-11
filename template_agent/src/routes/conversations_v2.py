"""Shadowbot V2 conversation list handler.

# /api/v2/conversations

Returns paginated list of conversations for the authenticated user.

DUMMY IMPLEMENTATION: persistence is not wired yet, so this returns an empty
list with the correct response envelope. When the metadata store
(`thread_metadata` table from the Shadowbot V2 migration guide) is added,
replace the stub body with a real DB lookup keyed on `user.email`.
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_conversations_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ConversationListResponseV2

from template_agent.src.routes.common import logger


# /api/v2/conversations
@get_conversations_handler_v2()
@require_auth
async def handle_get_conversations_v2(
    page: int = 1,
    page_size: int = 20,
    platform: Optional[str] = None,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> ConversationListResponseV2:
    """Return paginated conversations (stub: empty list until DB is wired)."""
    user_email = (user.email if user else None) or "anonymous"
    logger.warning(
        "[V2] DUMMY IMPLEMENTATION - List conversations returns empty list. "
        "Wire up persistence to enable history.",
        user_id=user_email,
        page=page,
        page_size=page_size,
        platform=platform,
    )
    return ConversationListResponseV2(
        conversations=[],
        total_count=0,
        page=page,
        page_size=page_size,
    )
