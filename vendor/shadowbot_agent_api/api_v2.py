"""V2 API Endpoints - Enhanced endpoints with better error handling and request support.

This module contains all V2 API endpoints that provide:
- Better exception handling with detailed error information
- HTTP Request parameter for all endpoints
- CustomAuthHeaders for third-party service auth
- Structured tool call support in responses
- Enhanced metadata and extensibility
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Response, Depends, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
import json

from .constants import Constants
from .exceptions import AgentException
from .core import get_handler
from shadowbot_agent_api.logger import get_python_logger
from .models import UserContext, CustomAuthHeaders
from .models_v2 import (
    ChatMessageV2,
    ConversationRequestV2,
    MessageHistoryResponseV2,
    FeedbackRequestV2,
    FeedbackResponseV2,
    ConversationListResponseV2,
    StreamEventV2,
    DeleteConversationResponseV2,
    DataSourcesResponseV2,
    FeedbackCategoriesResponseV2,
)
from .auth import get_optional_user, get_custom_auth_headers

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)


def _handler_requires_auth(handler_func) -> bool:
    """Check if a handler function requires authentication (via @require_auth decorator)."""
    return getattr(handler_func, '_auth_required', False)


def _handler_supports_auth(handler_func) -> bool:
    """Check if a handler function supports optional authentication (via @with_auth decorator)."""
    return getattr(handler_func, '_auth_enabled', False) or _handler_requires_auth(handler_func)


# Define APIRouter for V2 endpoints
chat_api_router_v2 = APIRouter(
    prefix="/api/v2/conversations",
    tags=["Conversations V2"],
) 

@chat_api_router_v2.post("/chat", response_model=ChatMessageV2, summary="Send a chat message V2 (non-streaming)")
async def chat_endpoint_v2(
    request: Request,
    request_body: ConversationRequestV2,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Handles V2 non-streaming chat requests.

    V2 Enhancements:
    - Returns unified ChatMessageV2 model with tool_calls, metadata, references
    - Supports session_id for multi-conversation tracking
    - Supports custom_fields for platform extensibility
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details from AgentException

    Calls the developer's registered `chat_handler_v2`.
    """
    try:
        handler = get_handler("chat_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            message = await handler(request_body, **kwargs)
        else:
            message = await handler(request_body)

        return message
    except NotImplementedError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=501, detail=str(e))
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        # Log unexpected errors and raise as AgentProcessingException
        logger.exception(f"Unexpected error in chat_v2 handler: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error during chat processing: {str(e)}")


@chat_api_router_v2.post("/chat/stream", summary="Send a chat message V2 (streaming)")
async def stream_chat_endpoint_v2(
    request: Request,
    request_body: ConversationRequestV2,
    response: Response,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Handles V2 streaming chat requests.

    V2 Streaming Pattern:
    - Yields ChatMessageV2 objects progressively
    - Same message_id = update existing message
    - New message_id = create new message
    - Tool calls included in ChatMessageV2 when AI uses tools
    - Tool results are separate ChatMessageV2 with tool_call_id
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details

    Calls the developer's registered `stream_chat_handler_v2` and streams as SSE.
    """
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"

    async def event_generator():
        try:
            handler = get_handler("stream_chat_v2")

            # Check if handler requires auth and user is not authenticated
            if _handler_requires_auth(handler) and user is None:
                from .exceptions import AuthenticationException
                exc = AuthenticationException()
                yield f"data: {json.dumps(exc.to_dict())}\n\n"
                return

            # Inspect handler signature to determine parameters
            import inspect
            sig = inspect.signature(handler)
            params = sig.parameters

            # Build kwargs based on what handler accepts
            kwargs = {}
            if 'user' in params:
                kwargs['user'] = user
            if 'custom_auth' in params:
                kwargs['custom_auth'] = custom_auth
            if 'request' in params:
                kwargs['request'] = request

            if kwargs:
                async for message in handler(request_body, **kwargs):
                    if isinstance(message, (ChatMessageV2, StreamEventV2)):
                        yield f"data: {json.dumps(jsonable_encoder(message))}\n\n"
                    else:
                        logger.warning(f"Unexpected item type yielded by stream_v2 handler: {type(message)}")
            else:
                async for message in handler(request_body):
                    if isinstance(message, (ChatMessageV2, StreamEventV2)):
                        yield f"data: {json.dumps(jsonable_encoder(message))}\n\n"
                    else:
                        logger.warning(f"Unexpected item type yielded by stream_v2 handler: {type(message)}")
        except NotImplementedError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'error_code': 'NOT_IMPLEMENTED'})}\n\n"
            logger.error(f"Streaming chat_v2 handler not implemented: {e}")
        except AgentException as e:
            # Stream AgentException details
            yield f"data: {json.dumps(e.to_dict())}\n\n"
            logger.error(f"AgentException in streaming chat_v2: {e.detail}")
        except Exception as e:
            # Log and stream unexpected errors
            logger.exception(f"Unexpected error in streaming chat_v2 handler: {e}")
            from .exceptions import AgentProcessingException
            exc = AgentProcessingException(detail=f"Internal server error during streaming chat: {str(e)}")
            yield f"data: {json.dumps(exc.to_dict())}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@chat_api_router_v2.get("/{conversation_id}/messages", response_model=MessageHistoryResponseV2, summary="Get message history V2")
async def get_messages_endpoint_v2(
    request: Request,
    conversation_id: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=50, ge=1, le=100, description="Number of messages per page (max 100)"),
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Retrieves V2 message history for a conversation with pagination support.

    Query Parameters:
    - page: Page number (default: 1, minimum: 1)
    - page_size: Messages per page (default: 50, minimum: 1, maximum: 100)

    V2 Enhancements:
    - Returns MessageHistoryResponseV2 with List[ChatMessageV2]
    - Each ChatMessageV2 includes tool_calls, metadata, references
    - Supports pagination with page, page_size, and total_count
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details

    Calls the developer's registered `get_messages_handler_v2`.
    """
    try:
        handler = get_handler("get_messages_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'page' in params:
            kwargs['page'] = page
        if 'page_size' in params:
            kwargs['page_size'] = page_size
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            history = await handler(conversation_id, **kwargs)
        else:
            history = await handler(conversation_id, page, page_size)

        return history
    except NotImplementedError as e:
        logger.warning(f"No handler registered for 'get_messages_v2'. Returning empty history.")
        # Return empty history response
        return MessageHistoryResponseV2(
            messages=[],
            conversationID=conversation_id,
            sessionID="",
            totalCount=0
        )
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_messages_v2 handler for conv_id {conversation_id}: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error fetching messages: {str(e)}")


@chat_api_router_v2.post("/{conversation_id}/messages/{message_id}/feedback", response_model=FeedbackResponseV2, summary="Submit feedback V2")
async def feedback_endpoint_v2(
    request: Request,
    conversation_id: str,
    message_id: str,
    request_body: FeedbackRequestV2,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Submits V2 feedback on a message.

    V2 Enhancements:
    - type: Categorize feedback (positive, negative, bug-report)
    - score: Numeric rating (0.0-1.0)
    - custom_fields: Platform-specific feedback data
    - Returns modifier_details with audit trail
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details

    Calls the developer's registered `feedback_handler_v2`.
    """
    try:
        handler = get_handler("feedback_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            response = await handler(conversation_id, message_id, request_body, **kwargs)
        else:
            response = await handler(conversation_id, message_id, request_body)

        return response
    except NotImplementedError:
        # Default implementation: log the feedback
        logger.info(f"Feedback V2 received for conversation {conversation_id}, message {message_id}")
        logger.info(f"  Type: {request_body.type}, Score: {request_body.score}")
        if request_body.comment:
            logger.info(f"  Comment: {request_body.comment}")
        if user:
            logger.info(f"  From user: {user.email or user.sub}")

        return FeedbackResponseV2(
            messageID=message_id,
            modifierDetails={
                "created_at": datetime.utcnow().isoformat() + "Z",
                "created_by": user.email if user else "anonymous"
            },
            additionalReferences=[],
            customFields={}
        )
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in feedback_v2 handler for conv_id {conversation_id}, msg_id {message_id}: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error processing feedback: {str(e)}")


@chat_api_router_v2.get("", response_model=ConversationListResponseV2, summary="Get conversation list V2")
async def get_conversations_endpoint_v2(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Number of items per page (max 100)"),
    platform: Optional[str] = Query(default=None, description="Optional platform filter (e.g., 'web', 'slack', or comma-separated 'web,mobile')"),
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Retrieves V2 conversation list with pagination support.

    Query Parameters:
    - page: Page number (default: 1, minimum: 1)
    - page_size: Items per page (default: 20, minimum: 1, maximum: 100)
    - platform: Optional platform filter (e.g., 'web', 'slack', or comma-separated 'web,mobile')

    V2 Enhancements:
    - Returns ConversationListResponseV2 with List[ConversationV2]
    - Each ConversationV2 includes metadata only (no messages)
    - Supports pagination (total_count, page, page_size)
    - modifier_details tracks created/updated timestamps
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details
    - Supports platform filtering

    Calls the developer's registered `get_conversations_handler_v2`.
    """
    try:
        handler = get_handler("get_conversations_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'page' in params:
            kwargs['page'] = page
        if 'page_size' in params:
            kwargs['page_size'] = page_size
        if 'platform' in params:
            kwargs['platform'] = platform
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            conv_list = await handler(**kwargs)
        else:
            conv_list = await handler()

        return conv_list
    except NotImplementedError:
        logger.warning("No handler registered for 'get_conversations_v2'. Returning empty list.")
        return ConversationListResponseV2(
            conversations=[],
            totalCount=0
        )
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_conversations_v2 handler: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error fetching conversations: {str(e)}")


@chat_api_router_v2.delete("/{conversation_id}", response_model=DeleteConversationResponseV2, summary="Delete conversation V2")
async def delete_conversation_endpoint_v2(
    request: Request,
    conversation_id: str,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Deletes a conversation by ID.

    V2 Enhancements:
    - Returns DeleteConversationResponseV2 with deletion confirmation
    - Includes modifier_details with audit trail (deleted_at, deleted_by)
    - Supports custom_fields for platform-specific metadata
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details

    Calls the developer's registered `delete_conversation_handler_v2`.
    """
    try:
        handler = get_handler("delete_conversation_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            response = await handler(conversation_id, **kwargs)
        else:
            response = await handler(conversation_id)

        return response
    except NotImplementedError:
        # Default implementation: log the deletion and return success
        logger.info(f"Delete conversation V2 called for conversation_id: {conversation_id}")
        if user:
            logger.info(f"  Deleted by user: {user.email or user.sub}")

        return DeleteConversationResponseV2(
            conversationID=conversation_id,
            status="deleted",
            modifierDetails={
                "deleted_at": datetime.utcnow().isoformat() + "Z",
                "deleted_by": user.email if user else "anonymous"
            },
            customFields={}
        )
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in delete_conversation_v2 handler for conv_id {conversation_id}: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error deleting conversation: {str(e)}")


@chat_api_router_v2.get("/feedback/categories", response_model=FeedbackCategoriesResponseV2, summary="Get feedback categories V2")
async def get_feedback_categories_endpoint_v2(
    request: Request,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Retrieves V2 feedback categories supported by the agent.

    Returns the list of feedback categories that the UI should render
    when a user wants to provide feedback on a message. Each category
    includes a code, display label, and whether a comment is required.

    Calls the developer's registered `get_feedback_categories_handler_v2`.
    """
    try:
        handler = get_handler("get_feedback_categories_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            response = await handler(**kwargs)
        else:
            response = await handler()

        return response
    except NotImplementedError:
        logger.warning("No handler registered for 'get_feedback_categories_v2'. Returning empty list.")
        return FeedbackCategoriesResponseV2(feedback_categories=[])
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_feedback_categories_v2 handler: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error fetching feedback categories: {str(e)}")


@chat_api_router_v2.get("/data/sources", response_model=DataSourcesResponseV2, response_model_exclude_none=True, summary="Get data sources V2")
async def get_data_sources_endpoint_v2(
    request: Request,
    user: Optional[UserContext] = Depends(get_optional_user),
    custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
):
    """
    Retrieves V2 data sources information.

    V2 Data Sources API:
    - Returns DataSourcesResponseV2 with live and upcoming collections
    - Live collections may include optional last_updated timestamps (human-readable format)
    - Upcoming collections are planned data sources not yet live
    - Request parameter available for accessing request details
    - Better exception handling that preserves error details

    This endpoint is typically used by agent management UIs to display
    what data sources an agent is using and when they were last refreshed.

    Calls the developer's registered `get_data_sources_handler_v2`.
    """
    try:
        handler = get_handler("get_data_sources_v2")

        # Check if handler requires auth and user is not authenticated
        if _handler_requires_auth(handler) and user is None:
            from .exceptions import AuthenticationException
            raise AuthenticationException()

        # Inspect handler signature to determine parameters
        import inspect
        sig = inspect.signature(handler)
        params = sig.parameters

        # Build kwargs based on what handler accepts
        kwargs = {}
        if 'user' in params:
            kwargs['user'] = user
        if 'custom_auth' in params:
            kwargs['custom_auth'] = custom_auth
        if 'request' in params:
            kwargs['request'] = request

        if kwargs:
            response = await handler(**kwargs)
        else:
            response = await handler()

        return response
    except NotImplementedError:
        logger.warning("No handler registered for 'get_data_sources_v2'. Returning empty data sources.")
        # Return empty data sources response
        return DataSourcesResponseV2(
            live_collections=[],
            upcoming_collections=[]
        )
    except AgentException:
        # Re-raise AgentException to be handled by custom exception handler
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_data_sources_v2 handler: {e}")
        from .exceptions import AgentProcessingException
        raise AgentProcessingException(detail=f"Internal server error fetching data sources: {str(e)}")