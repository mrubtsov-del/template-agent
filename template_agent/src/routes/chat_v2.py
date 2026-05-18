"""Shadowbot V2 synchronous chat handler.

# POST /api/v2/conversations/chat
"""

from typing import Optional
from uuid import uuid4

from shadowbot_agent_api import UserContext, chat_handler_v2, require_auth
from shadowbot_agent_api.models import CustomAuthHeaders
from shadowbot_agent_api.models_v2 import ChatMessageV2, ConversationRequestV2

from template_agent.src.routes.common import (
    build_agent_manager,
    build_chat_message_v2,
    collect_final_ai_text,
    logger,
    resolve_snowflake_login,
    resolve_user_id_v2,
    resolve_v2_conversation_ids,
    snowflake_auth_present,
)
from template_agent.src.schema import StreamRequest


@chat_handler_v2()
@require_auth
async def handle_chat_request_v2(
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> ChatMessageV2:
    """Run the agent to completion and return a single ``ChatMessageV2``.

    Parameter is ``request_body`` (not ``request``) so vendor's V2 dispatcher
    does not inject a FastAPI ``Request`` kwarg with the same name.
    """
    conv_id, session_id = resolve_v2_conversation_ids(request_body)
    msg_id = str(uuid4())
    user_id = resolve_user_id_v2(request_body, user)
    snowflake_login = resolve_snowflake_login(user, request_body=request_body)

    logger.info(
        "[V2] Chat called",
        conversation_id=conv_id,
        session_id=session_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    try:
        manager = build_agent_manager(custom_auth, snowflake_login)
        stream_req = StreamRequest(
            message=request_body.message,
            thread_id=conv_id,
            session_id=session_id,
            user_id=user_id,
            stream_tokens=False,
        )
        final_text = await collect_final_ai_text(manager, stream_req)

        logger.info(
            "[V2] Chat completed",
            conversation_id=conv_id,
            message_id=msg_id,
            chars=len(final_text),
        )

        return build_chat_message_v2(
            content=final_text or "No response generated.",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
            custom_fields=request_body.custom_fields or {},
        )
    except Exception as exc:
        logger.error(
            "[V2] Chat failed",
            conversation_id=conv_id,
            user_id=user_id,
            error=str(exc),
            exc_info=True,
        )
        return build_chat_message_v2(
            content=f"Internal error: {exc}",
            message_id=msg_id,
            conversation_id=conv_id,
            session_id=session_id,
            custom_fields={},
            finish_reason="error",
            error=str(exc),
        )
