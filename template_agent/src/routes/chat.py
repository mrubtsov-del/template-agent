"""Shadowbot V1 synchronous chat handler.

# /api/v1/conversations/chat

The decorator @chat_handler() registers the function in the
shadowbot_agent_api handler registry. The actual FastAPI route lives in
vendor/shadowbot_agent_api/api.py and is mounted via app.include_router.
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
from shadowbot_agent_api.models import Response

from template_agent.src.core.manager import AgentManager
from template_agent.src.routes.common import logger, resolve_user_id
from template_agent.src.schema import StreamRequest


# /api/v1/conversations/chat
@chat_handler()
@require_auth
async def handle_chat_request(
    request: ConversationRequest,
    user: Optional[UserContext] = None,
) -> ConversationResponse:
    """Run the agent to completion and return the final text answer."""
    conv_id = request.conversationId or str(uuid4())
    user_id = resolve_user_id(request, user)

    logger.info(
        "[V1] Chat called",
        conversation_id=conv_id,
        user_id=user_id,
    )

    try:
        manager = AgentManager()
        stream_req = StreamRequest(
            message=request.question,
            thread_id=conv_id,
            session_id=conv_id,
            user_id=user_id,
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
            "[V1] Chat completed",
            conversation_id=conv_id,
            chars=len(final_text),
        )

        return ConversationResponse(
            conversationID=conv_id,
            messageID=str(uuid4()),
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
            messageID=str(uuid4()),
            response=Response(answer=f"Internal error: {exc}"),
            informationSaved="error occurred",
            chunkId=0,
            streamEnded=True,
            otherLinks={},
        )
