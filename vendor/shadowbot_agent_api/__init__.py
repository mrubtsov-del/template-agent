# Import and expose the core decorators for easy access
from .core import (
    # V1 decorators
    chat_handler,
    stream_chat_handler,
    get_conversations_handler,
    get_messages_handler,
    feedback_handler,
    # V2 decorators
    chat_handler_v2,
    stream_chat_handler_v2,
    get_conversations_handler_v2,
    get_messages_handler_v2,
    feedback_handler_v2,
    get_data_sources_handler_v2,
    get_feedback_categories_handler_v2,
    # Auth decorators (work with both V1 and V2)
    with_auth,
    require_auth
)

# Import and expose Pydantic models for type hinting in user's business logic
from .models import (
    # V1 models
    ConversationRequest, ConversationResponse,
    StreamChunk, StreamEnd,
    Conversation, ConversationMessage,
    FeedbackRequest, FeedbackResponse,
    # Shared models
    ToolCall, Source,
    # Auth models (used by both V1 and V2)
    AuthConfig, UserContext, CustomAuthHeaders
)
from .models_v2 import (
    # V2 models
    ChatMessageV2,
    ConversationRequestV2, MessageHistoryResponseV2,
    FeedbackRequestV2, FeedbackResponseV2,
    ConversationV2, ConversationListResponseV2,
    # Data sources models
    DataCollection, DataSourcesResponseV2,
    # Feedback categories models
    FeedbackCategoryItem, FeedbackCategoriesResponseV2,
)
# Import and expose auth utilities
from .auth import configure_auth, setup_auth_from_env, get_custom_auth_headers

# Import and expose logger
from .logger import get_python_logger

# Import and expose exceptions for V2 error handling
from .exceptions import (
    AgentException,
    ConversationNotFoundException,
    MessageNotFoundException,
    SessionNotFoundException,
    InvalidRequestException,
    AuthenticationException,
    AuthorizationException,
    AgentProcessingException,
    RateLimitException,
    ToolExecutionException
)

# Import and expose exception handlers
from .exception_handlers import register_exception_handlers

# Import and expose the server runner function
from .server import run_server

# Import and expose the FastAPI app utilities (but don't create app at import time)
from .app import create_app  # Factory function for creating custom apps
from .app import configure_cors  # CORS configuration utility
from .app import get_default_app  # Get default app instance
from .api import chat_api_router  # V1 APIRouter for modular integration
from .api import health_endpoint  # Health check endpoint function
from .api_v2 import chat_api_router_v2  # V2 APIRouter for modular integration

# Lazy app creation - only create when explicitly requested
app = None  # Users should use create_app() or get_default_app() instead

__all__ = [
    # V1 handler decorators
    "chat_handler",
    "stream_chat_handler",
    "get_conversations_handler",
    "get_messages_handler",
    "feedback_handler",
    # V2 handler decorators
    "chat_handler_v2",
    "stream_chat_handler_v2",
    "get_conversations_handler_v2",
    "get_messages_handler_v2",
    "feedback_handler_v2",
    "get_data_sources_handler_v2",
    "get_feedback_categories_handler_v2",
    # Auth decorators (work with both V1 and V2)
    "with_auth",
    "require_auth",
    # Server utilities
    "run_server",
    # V1 models
    "ConversationRequest", "ConversationResponse",
    "StreamChunk", "StreamEnd",
    "Conversation", "ConversationMessage",
    "FeedbackRequest", "FeedbackResponse",
    "CustomAuthHeaders",
    # V2 models
    "ChatMessageV2", "ToolCall", "Source",
    "ConversationRequestV2", "MessageHistoryResponseV2",
    "FeedbackRequestV2", "FeedbackResponseV2",
    "ConversationV2", "ConversationListResponseV2",
    # Data sources models
    "DataCollection", "DataSourcesResponseV2",
    # Feedback categories models
    "FeedbackCategoryItem", "FeedbackCategoriesResponseV2",
    # Auth models (used by both V1 and V2)
    "AuthConfig", "UserContext",
    # Auth utilities
    "configure_auth", "setup_auth_from_env", "get_custom_auth_headers",
    # Logger
    "get_python_logger",
    # V2 Exceptions for detailed error handling
    "AgentException",
    "ConversationNotFoundException",
    "MessageNotFoundException",
    "SessionNotFoundException",
    "InvalidRequestException",
    "AuthenticationException",
    "AuthorizationException",
    "AgentProcessingException",
    "RateLimitException",
    "ToolExecutionException",
    # Exception handler utilities
    "register_exception_handlers",
    # FastAPI components
    "app",  # The package's default FastAPI app (lazy, use get_default_app() instead)
    "create_app",  # Factory function for creating custom apps
    "get_default_app",  # Get default app instance (lazy creation)
    "chat_api_router",  # V1 APIRouter for modular integration
    "chat_api_router_v2",  # V2 APIRouter for modular integration
    "health_endpoint",  # The health endpoint function
    "configure_cors"  # CORS configuration utility
]
