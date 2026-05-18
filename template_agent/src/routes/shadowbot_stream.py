"""Shadowbot V1 streaming chat handler.

# POST /api/v1/conversations/chat/stream
"""

from typing import AsyncGenerator, Optional, Union
from uuid import uuid4

from shadowbot_agent_api import (
    ConversationRequest,
    StreamChunk,
    StreamEnd,
    UserContext,
    require_auth,
    stream_chat_handler,
)
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.routes.common import (
    build_agent_manager,
    iter_agent_stream_events,
    logger,
    resolve_snowflake_login,
    resolve_user_id,
    snowflake_auth_present,
    utc_iso_z,
)
from template_agent.src.schema import StreamRequest


@stream_chat_handler()
@require_auth
async def handle_stream_chat(
    request: ConversationRequest,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> AsyncGenerator[Union[StreamChunk, StreamEnd], None]:
    """Yield ``StreamChunk`` per token and a terminating ``StreamEnd``."""
    conv_id = request.conversationId or str(uuid4())
    msg_id = str(uuid4())
    user_id = resolve_user_id(request, user)
    snowflake_login = resolve_snowflake_login(user, request)
    content_type = "ai"

    logger.info(
        "[V1] Stream called",
        conversation_id=conv_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    manager = build_agent_manager(custom_auth, snowflake_login)
    stream_req = StreamRequest(
        message=request.question,
        thread_id=conv_id,
        session_id=conv_id,
        user_id=user_id,
        stream_tokens=True,
    )

    full_text = ""
    token_count = 0
    try:
        async for event in iter_agent_stream_events(manager, stream_req):
            event_type = event.get("type")
            content = event.get("content", {})

            if event_type == "token":
                token_text = event.get("content", "")
                if not token_text:
                    continue
                full_text += token_text
                token_count += 1
                yield StreamChunk(
                    type="token", contentType=content_type, text=token_text
                )
            elif event_type == "message":
                if (
                    content.get("type") == "ai"
                    and content.get("content")
                    and not full_text
                ):
                    text = content["content"]
                    full_text += text
                    token_count += 1
                    yield StreamChunk(type="token", contentType=content_type, text=text)
    except Exception as exc:
        logger.error(
            "[V1] Stream failed",
            conversation_id=conv_id,
            user_id=user_id,
            error=str(exc),
            exc_info=True,
        )
        yield StreamChunk(type="token", contentType=content_type, text=f"Error: {exc}")

    yield StreamEnd(
        type="message",
        contentType=content_type,
        conversationID=conv_id,
        messageID=msg_id,
        finalText=full_text,
        timestamp=utc_iso_z(),
        messageReferenceList=[],
        imageReferenceList=None,
    )

    logger.info(
        "[V1] Stream completed",
        conversation_id=conv_id,
        message_id=msg_id,
        tokens=token_count,
        chars=len(full_text),
    )
