from typing import Callable, Any, Dict, AsyncGenerator, Union, Coroutine, List

from .constants import Constants
from shadowbot_agent_api.logger import get_python_logger
from .models import (
    # V1 models
    ConversationRequest, ConversationResponse, Conversation, FeedbackRequest, FeedbackResponse,
    StreamChunk, StreamEnd, ConversationMessage
)
from .models_v2 import (
    # V2 models
    ConversationRequestV2, ChatMessageV2, MessageHistoryResponseV2,
    FeedbackRequestV2, FeedbackResponseV2, ConversationListResponseV2,
    StreamEventV2, DataSourcesResponseV2, FeedbackCategoriesResponseV2
)

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)

# This dictionary will store the user-defined handler functions.
_handler_registry: Dict[str, Callable[..., Any]] = {}


def _register_handler(api_name: str, handler_func: Callable[..., Any]):
    """Internal function to register a handler."""
    if api_name in _handler_registry:
        logger.warning(f"Handler for '{api_name}' is being overwritten.")
    _handler_registry[api_name] = handler_func
    logger.info(f"Registered handler for '{api_name}': {handler_func.__name__}")


def get_handler(api_name: str) -> Callable[..., Any]:
    """Retrieves a registered handler."""
    handler = _handler_registry.get(api_name)
    if not handler:
        logger.warn(f"No handler registered for API: '{api_name}'")
        raise NotImplementedError(f"Business logic for '{api_name}' is not implemented. "
                                  "Please decorate a function with the corresponding handler.")
    return handler


def chat_handler() -> Callable[[Callable[[ConversationRequest], Coroutine[Any, Any, ConversationResponse]]], Callable[
    [ConversationRequest], Coroutine[Any, Any, ConversationResponse]]]:
    """
    Decorator for the synchronous chat API (`POST /api/v1/conversations/chat`).
    The decorated function should accept a ConversationRequest and return a ConversationResponse.
    It must be an async function.
    """

    def decorator(func: Callable[[ConversationRequest], Coroutine[Any, Any, ConversationResponse]]) -> Callable[
        [ConversationRequest], Coroutine[Any, Any, ConversationResponse]]:
        _register_handler("chat", func)
        return func

    return decorator


def stream_chat_handler() -> Callable[
    [Callable[[ConversationRequest], AsyncGenerator[Union[StreamChunk, StreamEnd], None]]], Callable[
        [ConversationRequest], AsyncGenerator[Union[StreamChunk, StreamEnd], None]]]:
    """
    Decorator for the streaming chat API (`POST /api/v1/conversations/chat/stream`).
    The decorated function should accept a ConversationRequest and yield StreamChunk/StreamEnd objects.
    It must be an async generator function.
    """

    def decorator(func: Callable[[ConversationRequest], AsyncGenerator[Union[StreamChunk, StreamEnd], None]]) -> \
    Callable[[ConversationRequest], AsyncGenerator[Union[StreamChunk, StreamEnd], None]]:
        _register_handler("stream_chat", func)
        return func

    return decorator


def get_conversations_handler() -> Callable[
    [Callable[[], Coroutine[Any, Any, List[Conversation]]]], Callable[[], Coroutine[Any, Any, List[Conversation]]]]:
    """
    Decorator for the get conversations API (`GET /conversations`).
    The decorated function should accept no arguments and return a list of Conversation objects.
    It must be an async function.
    """

    def decorator(func: Callable[[], Coroutine[Any, Any, List[Conversation]]]) -> Callable[
        [], Coroutine[Any, Any, List[Conversation]]]:
        _register_handler("get_conversations", func)
        return func

    return decorator


def get_messages_handler() -> Callable[
    [Callable[[str], Coroutine[Any, Any, List[ConversationMessage]]]], Callable[[str], Coroutine[Any, Any, List[ConversationMessage]]]]:
    """
    Decorator for the get messages API (`GET /conversations/{conversation_id}/messages`).
    The decorated function should accept `conversation_id: str` and return a list of ConversationMessage objects.
    It must be an async function.
    """

    def decorator(func: Callable[[str], Coroutine[Any, Any, List[ConversationMessage]]]) -> Callable[
        [str], Coroutine[Any, Any, List[ConversationMessage]]]:
        _register_handler("get_messages", func)
        return func

    return decorator


def feedback_handler() -> Callable[
    [Callable[[str, str, FeedbackRequest], Coroutine[Any, Any, FeedbackResponse]]], Callable[
        [str, str, FeedbackRequest], Coroutine[Any, Any, FeedbackResponse]]]:
    """
    Decorator for the feedback API (`POST /conversations/{conversation_id}/messages/{message_id}/feedback`).
    The decorated function should accept `conversation_id: str`, `message_id: str`,
    and `request: FeedbackRequest` and return a FeedbackResponse.
    It must be an async function.
    """

    def decorator(func: Callable[[str, str, FeedbackRequest], Coroutine[Any, Any, FeedbackResponse]]) -> Callable[
        [str, str, FeedbackRequest], Coroutine[Any, Any, FeedbackResponse]]:
        _register_handler("feedback", func)
        return func

    return decorator


# Auth enablement decorators
def with_auth(func: Callable) -> Callable:
    """
    Decorator to enable authentication for any handler.
    When applied, the handler will receive an additional UserContext parameter.
    
    Usage:
        @chat_handler()
        @with_auth
        async def my_chat_logic(request: ConversationRequest, user: Optional[UserContext]) -> ConversationResponse:
            # Access user.email, user.sub, etc.
            pass
    """
    # Mark the function as auth-enabled
    func._auth_enabled = True
    return func


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for any handler.
    When applied, the handler will receive a UserContext parameter (never None).
    If no valid token is provided, the endpoint will return 401 Unauthorized.

    Usage:
        @chat_handler()
        @require_auth
        async def my_chat_logic(request: ConversationRequest, user: UserContext) -> ConversationResponse:
            # user is guaranteed to be present
            pass
    """
    # Mark the function as requiring auth
    func._auth_required = True
    return func


# ============================================
# V2 Handler Decorators
# ============================================

def chat_handler_v2() -> Callable[[Callable[[ConversationRequestV2], Coroutine[Any, Any, ChatMessageV2]]], Callable[
    [ConversationRequestV2], Coroutine[Any, Any, ChatMessageV2]]]:
    """
    Decorator for the V2 non-streaming chat API (`POST /api/v2/conversations/chat`).

    The decorated function should accept a ConversationRequestV2 and return a ChatMessageV2.
    It must be an async function.

    V2 Enhancements:
    - Uses unified ChatMessageV2 model (with tool_calls, metadata, references)
    - Supports session_id for multi-conversation tracking
    - Supports custom_fields for platform extensibility
    - Better exception handling - raise AgentException subclasses for detailed errors

    Optional Parameters:
    - user: Optional[UserContext] - Available when using @with_auth or @require_auth
    - custom_auth: CustomAuthHeaders - Available for custom auth headers
    - request: Request - FastAPI Request object for accessing request details

    Usage:
        @chat_handler_v2()
        async def my_chat_logic(request_body: ConversationRequestV2) -> ChatMessageV2:
            return ChatMessageV2(
                type="ai",
                content="Hello!",
                message_id="msg_1",
                conversation_id=request_body.conversation_id or "new_conv",
                session_id=request_body.session_id,
                tool_calls=[...],  # Optional tool calls
                references=[...]   # Optional RAG sources
            )

    With auth and request:
        @chat_handler_v2()
        @with_auth
        async def my_chat_logic(
            request_body: ConversationRequestV2,
            user: Optional[UserContext],
            request: Request
        ) -> ChatMessageV2:
            # Access user context and request details
            # Raise AgentException subclasses for detailed error handling
            from shadowbot_agent_api.exceptions import ConversationNotFoundException
            if not request_body.conversation_id:
                raise ConversationNotFoundException(conversation_id="")
            pass
    """
    def decorator(func: Callable[[ConversationRequestV2], Coroutine[Any, Any, ChatMessageV2]]) -> Callable[
        [ConversationRequestV2], Coroutine[Any, Any, ChatMessageV2]]:
        _register_handler("chat_v2", func)
        return func

    return decorator


def stream_chat_handler_v2() -> Callable[
    [Callable[[ConversationRequestV2], AsyncGenerator[Union[ChatMessageV2, StreamEventV2], None]]], Callable[
        [ConversationRequestV2], AsyncGenerator[Union[ChatMessageV2, StreamEventV2], None]]]:
    """
    Decorator for the V2 streaming chat API (`POST /api/v2/conversations/chat/stream`).

    The decorated function should accept a ConversationRequestV2 and yield ChatMessageV2 or StreamEventV2 objects.
    It must be an async generator function.

    V2 Streaming Pattern:
    - Yield progressive ChatMessageV2 updates with the same message_id
    - Client updates message when same message_id is seen
    - Client creates new message when new message_id is seen
    - Tool calls are included in ChatMessageV2 when AI decides to use tools
    - Tool results are separate ChatMessageV2 objects with tool_call_id
    - Better exception handling - raise AgentException subclasses for detailed errors

    Optional Parameters:
    - user: Optional[UserContext] - Available when using @with_auth or @require_auth
    - custom_auth: CustomAuthHeaders - Available for custom auth headers
    - request: Request - FastAPI Request object for accessing request details

    Usage:
        @stream_chat_handler_v2()
        async def my_stream_logic(request_body: ConversationRequestV2):
            # Stream AI response
            yield ChatMessageV2(type="ai", content="Hello", message_id="msg_1", ...)
            yield ChatMessageV2(type="ai", content="Hello, how", message_id="msg_1", ...)
            yield ChatMessageV2(type="ai", content="Hello, how can I help?",
                            tool_calls=[...], message_id="msg_1", ...)

            # Stream tool execution
            yield ChatMessageV2(type="tool", content="Tool result",
                            tool_call_id="call_123", message_id="msg_2", ...)

            # Stream final response
            yield ChatMessageV2(type="ai", content="Based on the results...",
                            message_id="msg_3", references=[...], ...)

    With auth and request:
        @stream_chat_handler_v2()
        @with_auth
        async def my_stream_logic(
            request_body: ConversationRequestV2,
            user: Optional[UserContext],
            request: Request
        ):
            # Access user context and request details
            # Raise AgentException for detailed error handling
            pass
    """
    def decorator(func: Callable[[ConversationRequestV2], AsyncGenerator[ChatMessageV2, None]]) -> \
        Callable[[ConversationRequestV2], AsyncGenerator[ChatMessageV2, None]]:
        _register_handler("stream_chat_v2", func)
        return func

    return decorator


def get_messages_handler_v2() -> Callable[
    [Callable[[str], Coroutine[Any, Any, MessageHistoryResponseV2]]], Callable[[str], Coroutine[Any, Any, MessageHistoryResponseV2]]]:
    """
    Decorator for the V2 message history API (`GET /api/v2/conversations/{conversation_id}/messages`).

    The decorated function should accept `conversation_id: str` and return a MessageHistoryResponseV2.
    It must be an async function.

    V2 Enhancements:
    - Returns MessageHistoryResponseV2 with List[ChatMessageV2]
    - Each ChatMessageV2 includes tool_calls, metadata, references
    - Supports pagination via total_count

    Usage:
        @get_messages_handler_v2()
        async def my_messages_logic(conversation_id: str) -> MessageHistoryResponseV2:
            messages = [
                ChatMessageV2(type="human", content="Hello", ...),
                ChatMessageV2(type="ai", content="Hi", tool_calls=[...], ...),
                ChatMessageV2(type="tool", content="Result", tool_call_id="...", ...),
            ]
            return MessageHistoryResponseV2(
                messages=messages,
                conversation_id=conversation_id,
                session_id="sess_123",
                total_count=len(messages)
            )

    With auth:
        @get_messages_handler_v2()
        @with_auth
        async def my_messages_logic(conversation_id: str, user: Optional[UserContext]) -> MessageHistoryResponseV2:
            # Access user context if provided
            pass
    """
    def decorator(func: Callable[[str], Coroutine[Any, Any, MessageHistoryResponseV2]]) -> Callable[
        [str], Coroutine[Any, Any, MessageHistoryResponseV2]]:
        _register_handler("get_messages_v2", func)
        return func

    return decorator


def feedback_handler_v2() -> Callable[
    [Callable[[str, str, FeedbackRequestV2], Coroutine[Any, Any, FeedbackResponseV2]]], Callable[
        [str, str, FeedbackRequestV2], Coroutine[Any, Any, FeedbackResponseV2]]]:
    """
    Decorator for the V2 feedback API (`POST /api/v2/conversations/{conversation_id}/messages/{message_id}/feedback`).

    The decorated function should accept `conversation_id: str`, `message_id: str`,
    and `request: FeedbackRequestV2` and return a FeedbackResponseV2.
    It must be an async function.

    V2 Enhancements:
    - type: Categorize feedback (positive, negative, bug-report)
    - score: Numeric rating (0.0-1.0 scale)
    - custom_fields: Platform-specific feedback data
    - modifier_details: Audit trail in response

    Usage:
        @feedback_handler_v2()
        async def my_feedback_logic(conversation_id: str, message_id: str,
                                    request: FeedbackRequestV2) -> FeedbackResponseV2:
            # request.type: "positive" | "negative" | "bug-report"
            # request.score: Optional[float] (0.0 - 1.0)
            # request.comment: Optional[str]
            # request.custom_fields: Dict[str, Any]

            return FeedbackResponseV2(
                message_id=message_id,
                modifier_details={
                    "created_at": "2025-10-31T10:00:00Z",
                    "created_by": "user@example.com"
                },
                additional_references=[...],
                custom_fields={}
            )

    With auth:
        @feedback_handler_v2()
        @with_auth
        async def my_feedback_logic(conversation_id: str, message_id: str,
                                    request: FeedbackRequestV2,
                                    user: Optional[UserContext]) -> FeedbackResponseV2:
            # Access user context if provided
            pass
    """
    def decorator(func: Callable[[str, str, FeedbackRequestV2], Coroutine[Any, Any, FeedbackResponseV2]]) -> Callable[
        [str, str, FeedbackRequestV2], Coroutine[Any, Any, FeedbackResponseV2]]:
        _register_handler("feedback_v2", func)
        return func

    return decorator


def get_conversations_handler_v2() -> Callable[
    [Callable[[], Coroutine[Any, Any, ConversationListResponseV2]]], Callable[[], Coroutine[Any, Any, ConversationListResponseV2]]]:
    """
    Decorator for the V2 conversations list API (`GET /api/v2/conversations`).

    The decorated function should accept no arguments and return a ConversationListResponseV2.
    It must be an async function.

    V2 Enhancements:
    - Returns ConversationListResponseV2 with List[ConversationV2]
    - Each ConversationV2 includes metadata only (no messages)
    - Supports pagination (total_count, page, page_size)
    - modifier_details tracks created/updated timestamps
    - custom_fields for platform-specific data

    Usage:
        @get_conversations_handler_v2()
        async def my_conversations_logic() -> ConversationListResponseV2:
            conversations = [
                ConversationV2(
                    conversation_id="conv_1",
                    session_id="sess_1",
                    title="Weather in Tokyo",
                    platform="slack",
                    modifier_details={
                        "created_at": "2025-10-31T10:00:00Z",
                        "last_message_at": "2025-10-31T10:30:00Z"
                    },
                    custom_fields={"last_message_preview": "The weather is..."}
                )
            ]
            return ConversationListResponseV2(
                conversations=conversations,
                total_count=42,
                page=1,
                page_size=20
            )

    With auth:
        @get_conversations_handler_v2()
        @with_auth
        async def my_conversations_logic(user: Optional[UserContext]) -> ConversationListResponseV2:
            # Access user context if provided
            # Filter conversations by user
            pass
    """
    def decorator(func: Callable[[], Coroutine[Any, Any, ConversationListResponseV2]]) -> Callable[
        [], Coroutine[Any, Any, ConversationListResponseV2]]:
        _register_handler("get_conversations_v2", func)
        return func

    return decorator


def get_data_sources_handler_v2() -> Callable[
    [Callable[[], Coroutine[Any, Any, DataSourcesResponseV2]]], Callable[[], Coroutine[Any, Any, DataSourcesResponseV2]]]:
    """
    Decorator for the V2 data sources API (`GET /api/v2/conversations/data/sources`).

    The decorated function should accept no arguments and return a DataSourcesResponseV2.
    It must be an async function.

    V2 Data Sources:
    - Returns DataSourcesResponseV2 with live and upcoming collections
    - Live collections include last_updated timestamps
    - Upcoming collections are planned data sources not yet live
    - Used by agent management UIs to display data source information

    Optional Parameters:
    - user: Optional[UserContext] - Available when using @with_auth or @require_auth
    - custom_auth: CustomAuthHeaders - Available for custom auth headers
    - request: Request - FastAPI Request object for accessing request details

    Usage:
        @get_data_sources_handler_v2()
        async def my_data_sources_logic() -> DataSourcesResponseV2:
            from shadowbot_agent_api.models_v2 import LiveDataCollection, UpcomingDataCollection
            
            live = [
                LiveDataCollection(
                    collection_name="product_docs",
                    last_updated="2024-12-01T02:15:00Z"
                ),
                LiveDataCollection(
                    collection_name="servicenow_kb",
                    last_updated="2024-11-28T18:45:00Z"
                )
            ]
            
            upcoming = [
                UpcomingDataCollection(collection_name="redhat_docs"),
                UpcomingDataCollection(collection_name="jira_tickets")
            ]
            
            return DataSourcesResponseV2(
                live_collections=live,
                upcoming_collections=upcoming
            )

    With auth:
        @get_data_sources_handler_v2()
        @with_auth
        async def my_data_sources_logic(user: Optional[UserContext]) -> DataSourcesResponseV2:
            # Access user context if provided
            # Return user-specific data sources
            pass
    """
    def decorator(func: Callable[[], Coroutine[Any, Any, DataSourcesResponseV2]]) -> Callable[
        [], Coroutine[Any, Any, DataSourcesResponseV2]]:
        _register_handler("get_data_sources_v2", func)
        return func

    return decorator


def get_feedback_categories_handler_v2() -> Callable[
    [Callable[[], Coroutine[Any, Any, FeedbackCategoriesResponseV2]]], Callable[[], Coroutine[Any, Any, FeedbackCategoriesResponseV2]]]:
    """
    Decorator for the V2 feedback categories API (`GET /api/v2/conversations/feedback/categories`).

    The decorated function should return a FeedbackCategoriesResponseV2 containing the list
    of feedback categories the agent supports. The UI uses this to dynamically render
    feedback category options.

    Optional Parameters:
    - user: Optional[UserContext] - Available when using @with_auth or @require_auth
    - custom_auth: CustomAuthHeaders - Available for custom auth headers
    - request: Request - FastAPI Request object for accessing request details

    Usage:
        @get_feedback_categories_handler_v2()
        async def my_feedback_categories() -> FeedbackCategoriesResponseV2:
            from shadowbot_agent_api.models_v2 import FeedbackCategoryItem

            return FeedbackCategoriesResponseV2(
                feedback_categories=[
                    FeedbackCategoryItem(code="incorrect_or_outdated_info", label="Incorrect or outdated info", comment_required=False),
                    FeedbackCategoryItem(code="error_or_outage", label="Error or outage", comment_required=True),
                    FeedbackCategoryItem(code="other", label="Other", comment_required=True),
                ]
            )

    With auth:
        @get_feedback_categories_handler_v2()
        @require_auth
        async def my_feedback_categories(user: UserContext) -> FeedbackCategoriesResponseV2:
            # Access user context
            pass
    """
    def decorator(func: Callable[[], Coroutine[Any, Any, FeedbackCategoriesResponseV2]]) -> Callable[
        [], Coroutine[Any, Any, FeedbackCategoriesResponseV2]]:
        _register_handler("get_feedback_categories_v2", func)
        return func

    return decorator

