"""Shadowbot V2 streaming chat handler.

# /api/v2/conversations/chat/stream

V2 streaming uses a single `StreamEventV2` envelope:
- Token events: `{type: "token", content: "<text>", message_id, conversation_id, session_id}`
- Final message: `{type: "message", content: ChatMessageV2}`

Same message_id across tokens = client patches/appends the same message.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from uuid import uuid4

from shadowbot_agent_api import (
    UserContext,
    require_auth,
    stream_chat_handler_v2,
)
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import (
    ChatMessageV2,
    ConversationRequestV2,
    StreamEventV2,
)

from template_agent.src.core.manager import AgentManager
from template_agent.src.routes.common import logger
from template_agent.src.schema import StreamRequest


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# /api/v2/conversations/chat/stream
@stream_chat_handler_v2()
@require_auth
async def handle_stream_chat_v2(
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> AsyncGenerator[StreamEventV2, None]:
    """Yield token StreamEventV2 events and a final message event.

    The first parameter is `request_body` (not `request`) to avoid a name
    clash with the FastAPI `Request` object that vendor's V2 dispatcher
    injects via kwargs. Naming matches the shadowbot-agent-api skill
    convention.
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
        "[V2] Stream called",
        conversation_id=conv_id,
        session_id=session_id,
        user_id=user_email,
    )

    manager = AgentManager(
        custom_auth=custom_auth,
        snowflake_login=user_email if user_email != "anonymous" else None,
    )
    stream_req = StreamRequest(
        message=request_body.message,
        thread_id=conv_id,
        session_id=session_id,
        user_id=user_email,
        stream_tokens=True,
    )

    accumulated = ""
    token_count = 0
    try:
        async for event in manager.stream_response(stream_req):
            event_type = event.get("type") if isinstance(event, dict) else None
            content = event.get("content", {}) if isinstance(event, dict) else {}

            if event_type == "token":
                token_text = event.get("content", "")
                if not token_text:
                    continue
                accumulated += token_text
                token_count += 1
                yield StreamEventV2(
                    type="token",
                    content=token_text,
                    message_id=msg_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                )
            elif event_type == "message":
                # Fallback when AgentManager did not stream tokens at all.
                if (
                    content.get("type") == "ai"
                    and content.get("content")
                    and not accumulated
                ):
                    text = content["content"]
                    accumulated += text
                    yield StreamEventV2(
                        type="token",
                        content=text,
                        message_id=msg_id,
                        conversation_id=conv_id,
                        session_id=session_id,
                    )
    except Exception as exc:
        logger.error(
            "[V2] Stream failed",
            conversation_id=conv_id,
            user_id=user_email,
            error=str(exc),
            exc_info=True,
        )
        yield StreamEventV2(
            type="token",
            content=f"\n[error] {exc}",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
        )

    final_message = ChatMessageV2(
        type="ai",
        content=accumulated,
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
    yield StreamEventV2(type="message", content=final_message)

    logger.info(
        "[V2] Stream completed",
        conversation_id=conv_id,
        message_id=msg_id,
        tokens=token_count,
        chars=len(accumulated),
    )
