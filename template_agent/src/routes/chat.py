"""Shadowbot V1 synchronous chat handler.

# POST /api/v1/conversations/chat
"""

from typing import Optional
from uuid import uuid4

from shadowbot_agent_api import (
    ConversationRequest,
    ConversationResponse,
    UserContext,
    chat_handler,
    require_auth,
)
from shadowbot_agent_api.models import CustomAuthHeaders, Response

from template_agent.src.routes.common import (
    build_agent_manager,
    collect_final_ai_text,
    logger,
    resolve_snowflake_login,
    resolve_user_id,
    snowflake_auth_present,
)
from template_agent.src.schema import StreamRequest


@chat_handler()
@require_auth
async def handle_chat_request(
    request: ConversationRequest,
    user: Optional[UserContext] = None,
    custom_auth: Optional[CustomAuthHeaders] = None,
) -> ConversationResponse:
    """Run the agent to completion and return the final text answer."""
    conv_id = request.conversationId or str(uuid4())
    msg_id = str(uuid4())
    user_id = resolve_user_id(request, user)
    snowflake_login = resolve_snowflake_login(user, request)

    logger.info(
        "[V1] Chat called",
        conversation_id=conv_id,
        user_id=user_id,
        snowflake_auth=snowflake_auth_present(custom_auth),
    )

    try:
        manager = build_agent_manager(custom_auth, snowflake_login)
        stream_req = StreamRequest(
            message=request.question,
            thread_id=conv_id,
            session_id=conv_id,
            user_id=user_id,
            stream_tokens=False,
        )
        final_text = await collect_final_ai_text(manager, stream_req)

        logger.info(
            "[V1] Chat completed",
            conversation_id=conv_id,
            message_id=msg_id,
            chars=len(final_text),
        )

        return ConversationResponse(
            conversationID=conv_id,
            messageID=msg_id,
            response=Response(answer=final_text or "No response generated."),
            informationSaved="question saved",
            chunkId=0,
            streamEnded=True,
            otherLinks={},
        )
    except Exception as exc:
        logger.error(
            "[V1] Chat failed",
            conversation_id=conv_id,
            user_id=user_id,
            error=str(exc),
            exc_info=True,
        )
        return ConversationResponse(
            conversationID=conv_id,
            messageID=msg_id,
            response=Response(answer=f"Internal error: {exc}"),
            informationSaved="error occurred",
            chunkId=0,
            streamEnded=True,
            otherLinks={},
        )
