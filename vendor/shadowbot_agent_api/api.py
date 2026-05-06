"""V1 API Endpoints - Original API endpoints for backward compatibility.

This module contains all V1 API endpoints. The FastAPI app instance and
shared configuration (CORS, exception handlers, etc.) are in app.py.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import HTTPException, Response, APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
import json

from .constants import Constants
from .core import get_handler
from shadowbot_agent_api.logger import get_python_logger
from .models import (
    # V1 models
    ConversationRequest, ConversationResponse,
    StreamChunk, StreamEnd,
    Conversation, ConversationMessage, FeedbackResponse, UserContext, Feedback,
    CustomAuthHeaders
)
from .auth import get_optional_user, get_custom_auth_headers

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)


def _handler_requires_auth(handler_func) -> bool:
    """Check if a handler function requires authentication."""
    return getattr(handler_func, '_auth_required', False)


def _handler_supports_auth(handler_func) -> bool:
    """Check if a handler function supports optional authentication."""
    return getattr(handler_func, '_auth_enabled', False) or _handler_requires_auth(handler_func)


# Define APIRouter for V1 endpoints
# Note: The main FastAPI app and shared configuration are in app.py
chat_api_router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations V1"],
)

@chat_api_router.post("/chat", response_model=ConversationResponse, summary="Send a chat message (synchronous)")
async def chat_endpoint(
    request_body: ConversationRequest,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Handles synchronous chat requests.
    Calls the developer's registered `chat_handler`.
    """
    try:
        handler = get_handler("chat")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params and _handler_supports_auth(handler):
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth

        if kwargs:
            response = await handler(request_body, **kwargs)
        else:
            response = await handler(request_body)

        return response
    except Exception as e:
        logger.exception(f"shadowbot_agents_api Error in chat handler: {e}")
        raise

@chat_api_router.post("/chat/stream", summary="Send a chat message (streaming)")
async def stream_chat_endpoint(
    request_body: ConversationRequest,
    response: Response,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Handles streaming chat requests.
    Calls the developer's registered `stream_chat_handler` and streams responses as SSE.
    """
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"

    async def event_generator():
        try:
            handler = get_handler("stream_chat")

            # Check if handler requires auth and user is not authenticated
            if _handler_requires_auth(handler) and user is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Authentication required'})}\n\n"
                return

            # Inspect handler signature to determine parameters
            import inspect
            sig = inspect.signature(handler)
            params = sig.parameters

            # Build kwargs based on what handler accepts
            kwargs = {}
            if 'user' in params and _handler_supports_auth(handler):
                kwargs['user'] = user
            if 'custom_auth' in params:
                kwargs['custom_auth'] = custom_auth

            if kwargs:
                async for item in handler(request_body, **kwargs):
                    if isinstance(item, (StreamChunk, StreamEnd)):
                        yield f"data: {json.dumps(jsonable_encoder(item))}\n\n"
                    else:
                        logger.warning(f"Unexpected item type yielded by stream handler: {type(item)}")
            else:
                async for item in handler(request_body):
                    if isinstance(item, (StreamChunk, StreamEnd)):
                        yield f"data: {json.dumps(jsonable_encoder(item))}\n\n"
                    else:
                        logger.warning(f"Unexpected item type yielded by stream handler: {type(item)}")
        except NotImplementedError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            logger.error(f"shadowbot_agents_api Streaming chat handler not implemented: {e}")
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            logger.error(f"shadowbot_agents_api Streaming chat handler ValueError: {e}")
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error during streaming chat.'})}\n\n"
            logger.exception(f"Error in streaming chat handler: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@chat_api_router.get("/", response_model=List[Conversation], summary="Get all conversations")
async def get_conversations_endpoint(
    user: Optional[UserContext] = Depends(get_optional_user)
):
    """
    Retrieves all conversations.
    Calls the developer's registered `get_conversations_handler`.
    """
    try:
        handler = get_handler("get_conversations")
        
        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Call handler with or without user context based on auth support
        if _handler_supports_auth(handler):
            conversations = await handler(user)
        else:
            conversations = await handler()
        
        return conversations
    except NotImplementedError:
        logger.info("No handler registered for 'get_conversations'. Returning empty list.")
        return []
    except Exception as e:
        logger.exception(f"shadowbot_agents_api Error in get_conversations handler: {e}")
        raise

@chat_api_router.get("/{conversation_id}/messages", response_model=List[ConversationMessage], summary="Get messages for a specific conversation")
async def get_messages_endpoint(
    conversation_id: str,
    user: Optional[UserContext] = Depends(get_optional_user)
):
    """
    Retrieves messages for a specific conversation.
    Calls the developer's registered `get_messages_handler`.
    """
    try:
        handler = get_handler("get_messages")
        
        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Call handler with or without user context based on auth support
        if _handler_supports_auth(handler):
            messages = await handler(conversation_id, user)
        else:
            messages = await handler(conversation_id)
        
        return messages
    except NotImplementedError as e:
        logger.info(f"shadowbot_agents_api No handler registered for 'get_messages'. Returning empty list.")
        return []
    except Exception as e:
        logger.exception(f"shadowbot_agents_api Error in get_messages handler for conv_id {conversation_id}: {e}")
        raise

@chat_api_router.post("/{conversation_id}/messages/{message_id}/feedback", response_model=FeedbackResponse, summary="Provide feedback on a message")
async def feedback_endpoint(
        conversation_id: str,
        message_id: str,
        request_body: Feedback,
        user: Optional[UserContext] = Depends(get_optional_user)
):
    """
    Provides feedback on a specific message.
    Calls the developer's registered `feedback_handler`.
    """
    try:
        handler = get_handler("feedback")
        
        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Call handler with or without user context based on auth support
        if _handler_supports_auth(handler):
            response = await handler(conversation_id, message_id, request_body, user)
        else:
            response = await handler(conversation_id, message_id, request_body)
        
        return response
    except NotImplementedError:
        # Default implementation: log the feedback
        logger.info(
            f"Feedback received for conversation {conversation_id}, message {message_id}: {request_body.option}")
        if request_body.comment:
            logger.info(f"Feedback comment: {request_body.comment}")
            logger.info(f"Feedback option: {request_body.option}")
        if user:
            logger.info(f"Feedback from user: {user.email or user.sub}")

        return FeedbackResponse(
            conversationID=conversation_id,
            messageID=message_id,
            otherLinks={},
            informationSaved="Feedback logged successfully"
        )
    except ValueError as e: # Example for business logic raising validation/not found error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"shadowbot_agents_api Error in feedback handler for conv_id {conversation_id}, msg_id {message_id}: {e}")
        raise



@chat_api_router.delete("/{conversation_id}", summary="Delete a conversation (V1 - Not Supported)")
async def delete_conversation_endpoint(
    conversation_id: str,
    user: Optional[UserContext] = Depends(get_optional_user)
):
    """
    Delete a conversation by ID.
    
    V1 API does not support delete functionality.
    This endpoint returns a 501 Not Implemented error with a helpful message.
    """
    logger.info(f"shadowbot_agents_api This API is not implemented, hence skipping deletion for conversationID={conversation_id} by user={user.email or user.sub}")



# Add health endpoint to the chat API router
@chat_api_router.get("/v1/shadowbot-agent/health", summary="Health check endpoint")
async def health_endpoint():
    """
    Basic health check endpoint.
    Returns the service status and version information.
    """
    return {
        "status": "healthy",
        "service": "Shadowbot Agent API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
