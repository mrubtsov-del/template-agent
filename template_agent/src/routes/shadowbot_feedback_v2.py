"""Shadowbot V2 feedback handler.

# /api/v2/conversations/{conversation_id}/messages/{message_id}/feedback

Accepts a granular feedback payload (type + score + optional category/comment)
for a specific message. Currently logs and acknowledges; storage backend can
be added later.
"""

from datetime import datetime, timezone
from typing import Optional

from shadowbot_agent_api import (
    UserContext,
    feedback_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import FeedbackRequestV2, FeedbackResponseV2

from template_agent.src.routes.common import logger


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# /api/v2/conversations/{conversation_id}/messages/{message_id}/feedback
@feedback_handler_v2()
@require_auth
async def handle_feedback_v2(
    conversation_id: str,
    message_id: str,
    feedback_data: FeedbackRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> FeedbackResponseV2:
    """Persist (currently: log) user feedback on a message.

    Body parameter is `feedback_data` (not `request`) so vendor's V2
    dispatcher doesn't clash when injecting a FastAPI `Request` kwarg.
    Naming matches the shadowbot-agent-api skill convention.
    """
    user_email = (user.email if user else None) or "anonymous"
    now = _utc_iso_z()

    logger.info(
        "[V2] Feedback received",
        conversation_id=conversation_id,
        message_id=message_id,
        type=feedback_data.type,
        score=feedback_data.score,
        category=feedback_data.category,
        has_comment=bool(feedback_data.comment),
        user=user_email,
    )

    return FeedbackResponseV2(
        message_id=message_id,
        modifier_details={
            "created_at": now,
            "created_by": user_email,
            "updated_at": now,
        },
        additional_references=[],
        custom_fields={},
    )
