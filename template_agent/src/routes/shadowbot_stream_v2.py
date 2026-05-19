"""Shadowbot V2 streaming chat handler.

# POST /api/v2/conversations/chat/stream

SSE contract (shadowbot-agent-api skill):
  1. Zero or more ``StreamEventV2`` with ``type="token"`` (same ``message_id``).
  2. Exactly one terminal ``StreamEventV2`` with ``type="message"`` and ``content=ChatMessageV2``.
"""

from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from shadowbot_agent_api import UserContext, require_auth, stream_chat_handler_v2
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ConversationRequestV2, StreamEventV2

from template_agent.src.routes.common import (
    build_agent_manager,
    build_chat_message_v2,
    iter_agent_stream_events,
    logger,
    resolve_snowflake_login,
    resolve_user_id_v2,
    resolve_v2_conversation_ids,
    shadowbot_plot_image_references,
    snowflake_auth_present,
)
from template_agent.src.schema import StreamRequest


def _token_event(
    text: str,
    *,
    message_id: str,
    conversation_id: str,
    session_id: str,
) -> StreamEventV2:
    return StreamEventV2(
        type="token",
        content=text,
        message_id=message_id,
        conversation_id=conversation_id,
        session_id=session_id,
    )


@stream_chat_handler_v2()
@require_auth
async def handle_stream_chat_v2(
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> AsyncGenerator[StreamEventV2, None]:
    """Yield token ``StreamEventV2`` events and a terminal message event."""
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
        platform=request_body.platform,
        stream_tokens=True,
    )

    accumulated = ""
    token_count = 0
    finish_reason = "stop"
    stream_error: Optional[str] = None
    # Last AI message from AgentManager "updates" (covers non-token LLM paths).
    last_ai: dict[str, Any] = {}

    try:
        async for event in iter_agent_stream_events(manager, stream_req):
            event_type = event.get("type")
            content = event.get("content", {})

            if event_type == "error":
                err = content if isinstance(content, dict) else {}
                stream_error = err.get("message") or "Agent error"
                finish_reason = "error"
                logger.warning(
                    "[V2] Stream agent error event",
                    conversation_id=conv_id,
                    error=stream_error,
                )
                yield _token_event(
                    f"\n[error] {stream_error}",
                    message_id=msg_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                )
                continue

            if event_type == "token":
                token_text = event.get("content", "")
                if not token_text:
                    continue
                accumulated += token_text
                token_count += 1
                yield _token_event(
                    token_text,
                    message_id=msg_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                )
                continue

            if event_type != "message" or not isinstance(content, dict):
                continue

            msg_type = content.get("type")
            if msg_type == "ai":
                ai_text = content.get("content") or ""
                if ai_text:
                    last_ai = {
                        "content": ai_text,
                        "tool_calls": content.get("tool_calls") or [],
                    }
                # Gemini often emits the full answer only via "updates", not "messages".
                if ai_text and not accumulated:
                    accumulated = ai_text
                    token_count += 1
                    yield _token_event(
                        ai_text,
                        message_id=msg_id,
                        conversation_id=conv_id,
                        session_id=session_id,
                    )
            elif msg_type == "tool" and not accumulated:
                # Visible heartbeat while Snowflake tools run (avoids "hung" UI).
                tool_name = content.get("name") or "tool"
                hint = f"\n[running {tool_name}...]\n"
                token_count += 1
                yield _token_event(
                    hint,
                    message_id=msg_id,
                    conversation_id=conv_id,
                    session_id=session_id,
                )

    except Exception as exc:
        stream_error = str(exc)
        finish_reason = "error"
        logger.error(
            "[V2] Stream failed",
            conversation_id=conv_id,
            user_id=user_id,
            error=stream_error,
            exc_info=True,
        )
        yield _token_event(
            f"\n[error] {exc}",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
        )

    if not accumulated and last_ai.get("content"):
        accumulated = last_ai["content"]

    plot_images = shadowbot_plot_image_references()
    custom_fields = dict(request_body.custom_fields or {})
    if plot_images:
        custom_fields["plots"] = [
            {"plot_id": img.article_id, "url": img.url, "title": img.article_title}
            for img in plot_images
        ]

    final_message = build_chat_message_v2(
        content=accumulated or (stream_error and f"Error: {stream_error}") or "",
        message_id=msg_id,
        conversation_id=conv_id,
        session_id=session_id,
        custom_fields=custom_fields,
        finish_reason=finish_reason,
        error=stream_error,
    )
    if last_ai.get("tool_calls"):
        final_message.tool_calls = last_ai["tool_calls"]
    if plot_images:
        final_message.images = plot_images

    yield StreamEventV2(type="message", content=final_message)

    logger.info(
        "[V2] Stream completed",
        conversation_id=conv_id,
        message_id=msg_id,
        tokens=token_count,
        chars=len(accumulated),
        finish_reason=finish_reason,
    )
