"""Shadowbot V1 streaming chat handler.

# /api/v1/conversations/chat/stream

The decorator @stream_chat_handler() registers the async generator in the
shadowbot_agent_api handler registry. The actual FastAPI route lives in
vendor/shadowbot_agent_api/api.py and is mounted via app.include_router.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator, Union
from uuid import uuid4

from shadowbot_agent_api import (
    ConversationRequest,
    StreamChunk,
    StreamEnd,
    UserContext,
    require_auth,
    stream_chat_handler,
)

from template_agent.src.core.manager import AgentManager
from template_agent.src.routes.common import logger
from template_agent.src.schema import StreamRequest


def _utc_iso_z() -> str:
    """Return current UTC time in ISO 8601 with trailing 'Z' (spec format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# /api/v1/conversations/chat/stream
@stream_chat_handler()
@require_auth
async def handle_stream_chat(
    request: ConversationRequest,
    user: UserContext,
) -> AsyncGenerator[Union[StreamChunk, StreamEnd], None]:
    """Yield StreamChunk for each token and a terminating StreamEnd."""
    conv_id = request.conversationId or str(uuid4())
    msg_id = str(uuid4())
    user_identifier = user.email or user.sub or user.preferred_username

    logger.info(
        "[V1] Stream called",
        conversation_id=conv_id,
        user_id=user_identifier,
    )

    manager = AgentManager()
    stream_req = StreamRequest(
        message=request.question,
        thread_id=conv_id,
        session_id=conv_id,
        user_id=user_identifier,
        stream_tokens=True,
    )

    full_text = ""
    message_content_type = "ai"
    try:
        async for event in manager.stream_response(stream_req):
            event_type = event.get("type") if isinstance(event, dict) else None
            content = event.get("content", {}) if isinstance(event, dict) else {}

            if event_type == "token":
                token_text = event.get("content", "")
                if not token_text:
                    continue
                full_text += token_text
                yield StreamChunk(
                    type="token", contentType=message_content_type, text=token_text
                )
            elif event_type == "message":
                # Fallback: if the agent didn't stream tokens, emit the full
                # message as a single chunk so the client still gets the answer.
                if (
                    content.get("type") == "ai"
                    and content.get("content")
                    and not full_text
                ):
                    text = content["content"]
                    full_text += text
                    yield StreamChunk(type="token", contentType="ai", text=text)
    except Exception as exc:
        logger.error(
            "[V1] Stream failed",
            conversation_id=conv_id,
            user_id=user_identifier,
            error=str(exc),
            exc_info=True,
        )
        yield StreamChunk(type="token", contentType="ai", text=f"Error: {exc}")

    yield StreamEnd(
        type="message",
        contentType=message_content_type,
        conversationID=conv_id,
        messageID=msg_id,
        finalText=full_text,
        timestamp=_utc_iso_z(),
        messageReferenceList=[],
        imageReferenceList=None,
    )

    logger.info(
        "[V1] Stream completed",
        conversation_id=conv_id,
        message_id=msg_id,
        chars=len(full_text),
    )
