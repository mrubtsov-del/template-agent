"""Shadowbot V2 conversation list handler.

# GET /api/v2/conversations

DUMMY IMPLEMENTATION: empty paginated list until thread metadata is wired.
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    get_conversations_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ConversationListResponseV2

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
    """Return paginated conversations (stub: empty list until DB is wired)."""
    user_id = resolve_user_label(user)
    logger.warning(
        "[V2] DUMMY IMPLEMENTATION - List conversations returns empty list. "
        "Wire up persistence to enable history.",
        user_id=user_id,
        page=page,
        page_size=page_size,
        platform=platform,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )
    return ConversationListResponseV2(
        conversations=[],
        total_count=0,
        page=page,
        page_size=page_size,
    )
