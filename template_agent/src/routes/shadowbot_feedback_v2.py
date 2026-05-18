"""Shadowbot V2 feedback handler.

# POST /api/v2/conversations/{conversation_id}/messages/{message_id}/feedback
"""

from typing import Optional

from shadowbot_agent_api import UserContext, feedback_handler_v2, require_auth
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import FeedbackRequestV2, FeedbackResponseV2

from template_agent.src.routes.common import (
    logger,
    resolve_user_label,
    snowflake_auth_present,
    utc_iso_z,
)


@feedback_handler_v2()
@require_auth
async def handle_feedback_v2(
    conversation_id: str,
    message_id: str,
    feedback_data: FeedbackRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> FeedbackResponseV2:
    """Log user feedback on a message (storage backend can be added later).

    Body parameter is ``feedback_data`` (not ``request``) per skill convention.
    """
    user_id = resolve_user_label(user)
    now = utc_iso_z()

    logger.info(
        "[V2] Feedback received",
        conversation_id=conversation_id,
        message_id=message_id,
        type=feedback_data.type,
        score=feedback_data.score,
        category=feedback_data.category,
        has_comment=bool(feedback_data.comment),
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    return FeedbackResponseV2(
        message_id=message_id,
        modifier_details={
            "created_at": now,
            "created_by": user_id,
            "updated_at": now,
        },
        additional_references=[],
        custom_fields={},
    )
