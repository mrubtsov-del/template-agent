"""Shadowbot V2 conversation list handler.

# GET /api/v2/conversations
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_conversations_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ConversationListResponseV2

from template_agent.src.core.conversation_history import list_conversations_for_user
from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
)


@get_conversations_handler_v2()
@require_auth
async def handle_get_conversations_v2(
    page: int = 1,
    page_size: int = 20,
    platform: Optional[str] = None,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> ConversationListResponseV2:
    """Return paginated conversations for the authenticated user."""
    user_id = resolve_user_label(user)
    logger.info(
        "[V2] Get conversations",
        user_id=user_id,
        page=page,
        page_size=page_size,
        platform=platform,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    result = list_conversations_for_user(
        user_id,
        page=page,
        page_size=page_size,
        platform=platform,
        user=user,
    )
    return ConversationListResponseV2(
        conversations=result.items,
        total_count=result.total_count,
        page=page,
        page_size=page_size,
    )
