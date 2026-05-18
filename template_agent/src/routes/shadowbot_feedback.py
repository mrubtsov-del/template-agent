"""Shadowbot V1 feedback handler.

# POST /api/v1/conversations/{conversation_id}/messages/{message_id}/feedback
"""

from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    feedback_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders, Feedback, FeedbackResponse

from template_agent.src.routes.common import logger, snowflake_auth_present


@feedback_handler()
@require_auth
async def handle_feedback(
    conversation_id: str,
    message_id: str,
    request: Feedback,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> FeedbackResponse:
    """Log user feedback on a message (storage backend can be added later)."""
    user_id = (user.email if user else None) or (user.sub if user else None) or "anonymous"

    logger.info(
        "[V1] Feedback received",
        conversation_id=conversation_id,
        message_id=message_id,
        option=request.option,
        has_comment=bool(request.comment),
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    return FeedbackResponse(
        conversationID=conversation_id,
        messageID=message_id,
        otherLinks={},
        informationSaved="Feedback logged successfully",
    )
