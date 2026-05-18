"""Shadowbot V2 streaming chat handler.

# POST /api/v2/conversations/chat/stream
"""

from typing import AsyncGenerator, Optional
from uuid import uuid4

from shadowbot_agent_api import UserContext, require_auth, stream_chat_handler_v2
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import (
    ConversationRequestV2,
    StreamEventV2,
)

from template_agent.src.routes.common import (
    build_agent_manager,
    build_chat_message_v2,
    iter_agent_stream_events,
    logger,
    resolve_snowflake_login,
    resolve_user_id_v2,
    resolve_v2_conversation_ids,
    snowflake_auth_present,
)
from template_agent.src.schema import StreamRequest


@stream_chat_handler_v2()
@require_auth
async def handle_stream_chat_v2(
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> AsyncGenerator[StreamEventV2, None]:
    """Yield token ``StreamEventV2`` events and a final message event."""
    conv_id, session_id = resolve_v2_conversation_ids(request_body)
    msg_id = str(uuid4())
    user_id = resolve_user_id_v2(request_body, user)
    snowflake_login = resolve_snowflake_login(user, request_body=request_body)

    logger.info(
        "[V2] Stream called",
        conversation_id=conv_id,
        session_id=session_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    manager = build_agent_manager(custom_auth, snowflake_login)
    stream_req = StreamRequest(
        message=request_body.message,
        thread_id=conv_id,
        session_id=session_id,
        user_id=user_id,
        stream_tokens=True,
    )

    accumulated = ""
    token_count = 0
    try:
        async for event in iter_agent_stream_events(manager, stream_req):
            event_type = event.get("type")
            content = event.get("content", {})

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
                if (
                    content.get("type") == "ai"
                    and content.get("content")
                    and not accumulated
                ):
                    text = content["content"]
                    accumulated += text
                    token_count += 1
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
            user_id=user_id,
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

    final_message = build_chat_message_v2(
        content=accumulated,
        message_id=msg_id,
        conversation_id=conv_id,
        session_id=session_id,
        custom_fields=request_body.custom_fields or {},
    )
    yield StreamEventV2(type="message", content=final_message)

    logger.info(
        "[V2] Stream completed",
        conversation_id=conv_id,
        message_id=msg_id,
        tokens=token_count,
        chars=len(accumulated),
    )
