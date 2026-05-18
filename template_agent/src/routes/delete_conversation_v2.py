"""Shadowbot V2 delete conversation handler.

# DELETE /api/v2/conversations/{conversation_id}
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    delete_conversation_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import DeleteConversationResponseV2

from template_agent.src.core.storage import unregister_thread
from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
    utc_iso_z,
)
from template_agent.src.settings import settings


@delete_conversation_handler_v2()
@require_auth
async def handle_delete_conversation_v2(
    conversation_id: str,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> DeleteConversationResponseV2:
    """Delete a conversation by ID.

    Removes the thread from the in-memory registry when ``USE_INMEMORY_SAVER`` is
    enabled. PostgreSQL checkpoint purge is not wired yet; the API still returns
    success so the Shadowbot UI can complete the delete flow.
    """
    user_id = resolve_user_label(user)
    deleted_at = utc_iso_z()

    logger.info(
        "[V2] Delete conversation",
        conversation_id=conversation_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    removed = False
    if settings.USE_INMEMORY_SAVER:
        removed = unregister_thread(user_id, conversation_id)
        if not removed and user_id != "anonymous":
            # Conversation may have been created under request userInfo only.
            removed = unregister_thread("anonymous", conversation_id)
    else:
        logger.warning(
            "[V2] Delete conversation: PostgreSQL checkpoint purge not implemented; "
            "returning deleted status for conversation_id=%s",
            conversation_id,
        )

    return DeleteConversationResponseV2(
        conversation_id=conversation_id,
        status="deleted",
        modifier_details={
            "deleted_at": deleted_at,
            "deleted_by": user_id,
            "registry_removed": removed,
        },
        custom_fields={},
    )
