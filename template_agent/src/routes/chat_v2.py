"""Shadowbot V2 synchronous chat handler.

# /api/v2/conversations/chat

V2 differences from V1:
- Request field is `message` (not `question`).
- `session_id` is part of the model (auto-generated if missing).
- Returns a single `ChatMessageV2` instead of `ConversationResponse`.
- Sources are `List[Source]`, not a `Dict[str, str]`.
- `custom_auth: CustomAuthHeaders` exposes `X-Authorization-*` third-party tokens.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from shadowbot_agent_api import (
    UserContext,
    chat_handler_v2,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ChatMessageV2, ConversationRequestV2

from template_agent.src.core.manager import AgentManager
from template_agent.src.routes.common import logger
from template_agent.src.schema import StreamRequest


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# /api/v2/conversations/chat
@chat_handler_v2()
@require_auth
async def handle_chat_request_v2(
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> ChatMessageV2:
    """Run the agent to completion and return a single ChatMessageV2.

    Note: the first parameter is named `request_body` (not `request`) to avoid
    a name clash with the FastAPI `Request` object that vendor's V2 dispatcher
    injects via kwargs when a parameter named `request` exists. Naming matches
    the shadowbot-agent-api skill convention.
    """
    conv_id = request_body.conversation_id or str(uuid4())
    session_id = request_body.session_id or conv_id
    msg_id = str(uuid4())
    user_email = (
        (user.email if user else None)
        or (request_body.user_info.userEmail if request_body.user_info else None)
        or "anonymous"
    )

    logger.info(
        "[V2] Chat called",
        conversation_id=conv_id,
        session_id=session_id,
        user_id=user_email,
    )

    try:
        manager = AgentManager()
        stream_req = StreamRequest(
            message=request_body.message,
            thread_id=conv_id,
            session_id=session_id,
            user_id=user_email,
            stream_tokens=False,
        )

        final_text = ""
        async for event in manager.stream_response(stream_req):
            if event.get("type") != "message":
                continue
            content = event.get("content", {})
            if content.get("type") == "ai" and content.get("content"):
                final_text = content["content"]

        logger.info(
            "[V2] Chat completed",
            conversation_id=conv_id,
            message_id=msg_id,
            chars=len(final_text),
        )

        return ChatMessageV2(
            type="ai",
            content=final_text or "No response generated.",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
            tool_calls=[],
            response_metadata={"finish_reason": "stop"},
            references=[],
            images=[],
            custom_fields=request_body.custom_fields or {},
            timestamp=_utc_iso_z(),
        )
    except Exception as exc:
        logger.error(
            "[V2] Chat failed",
            conversation_id=conv_id,
            user_id=user_email,
            error=str(exc),
            exc_info=True,
        )
        return ChatMessageV2(
            type="ai",
            content=f"Internal error: {exc}",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
            tool_calls=[],
            response_metadata={"finish_reason": "error", "error": str(exc)},
            references=[],
            images=[],
            custom_fields={},
            timestamp=_utc_iso_z(),
        )
