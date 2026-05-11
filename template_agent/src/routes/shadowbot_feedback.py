"""Shadowbot V1 feedback handler.

# /api/v1/conversations/{conversation_id}/messages/{message_id}/feedback

Accepts a thumbs-up/down style feedback payload for a specific message.
Currently logs and acknowledges; storage backend can be added later.
"""

from shadowbot_agent_api import (
    UserContext,
    feedback_handler,
    require_auth,
)
from shadowbot_agent_api.models import Feedback, FeedbackResponse

from template_agent.src.routes.common import logger


# /api/v1/conversations/{conversation_id}/messages/{message_id}/feedback
@feedback_handler()
@require_auth
async def handle_feedback(
    conversation_id: str,
    message_id: str,
    request: Feedback,
    user: UserContext,
) -> FeedbackResponse:
    """Persist (currently: log) user feedback on a message."""
    logger.info(
        "shadowbot.feedback",
        conversation_id=conversation_id,
        message_id=message_id,
        option=request.option,
        comment=request.comment,
        user=user.email or user.sub,
    )
    return FeedbackResponse(
        conversationID=conversation_id,
        messageID=message_id,
        otherLinks={},
        informationSaved="Feedback logged successfully",
    )
