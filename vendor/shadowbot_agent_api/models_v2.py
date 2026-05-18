"""V2 Models - Enhanced models with tool call support and improved structure.

This module contains V2 API models that provide:
- Structured tool call support
- Better metadata handling
- Cleaner field naming conventions
- Enhanced extensibility through custom_fields
- Authentication via headers (not request body)
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Union

from .models import UserInfo, ToolCall, Source, ImageReference


class ChatMessageV2(BaseModel):
    """
    Unified message model for V2 API.

    Used for:
    - Streaming responses (progressive updates by message_id)
    - Non-streaming responses (complete messages)
    - Message history (conversation replay)

    This model provides structured tool call support, metadata,
    and source citations that were missing in V1 models.
    Inspired by the ChatMessage pattern agents have implemented
    (see schema.py for reference), now officially provided by the package.
    """
    model_config = ConfigDict(populate_by_name=True)

    # Core fields
    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="The role or type of the message in the conversation"
    )
    content: str = Field(
        description="The text content of the message"
    )

    # Tool support (V2 enhancement)
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        validation_alias="toolCalls",
        description="Tool calls included in this message (for AI messages requesting tool execution)"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        validation_alias="toolCallID",
        description="ID of the tool call that this message is responding to (for tool result messages)"
    )

    # IDs
    message_id: str = Field(
        validation_alias="messageID",
        description="Unique identifier for this message"
    )
    conversation_id: str = Field(
        validation_alias="conversationID",
        description="ID of the conversation this message belongs to"
    )
    session_id: str = Field(
        validation_alias="sessionID",
        description="ID of the session (can span multiple conversations)"
    )

    # V2 enhancements
    response_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="responseMetadata",
        description="Additional metadata (model info, finish_reason, token counts, etc.)"
    )
    references: List[Source] = Field(
        default_factory=list,
        description="Source citations for RAG responses (replaces unstructured sources dict from V1)"
    )
    images: List[ImageReference] = Field(
        default_factory=list,
        description="List of images referenced in the response from RAG retrieval (ServiceNow KB articles)"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific data (slack_user_id, platform, etc.)"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of message creation"
    )


class StreamEventV2(BaseModel):
    """
    Streaming event model matching V1's two-level structure.

    Root level has only two event types:
    - type="token": Flat structure with content as string and IDs at root level
    - type="message": Nested structure with content as ChatMessageV2 object

    Examples:
        Token event:
        {
            "type": "token",
            "content": "Hello",
            "message_id": "abc123",
            "conversation_id": "conv456",
            "session_id": "session789"
        }

        Message event:
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "Hello, how can I help?",
                "message_id": "abc123",
                "conversation_id": "conv456",
                "session_id": "session789",
                "tool_calls": [],
                "references": [],
                "images": []
            }
        }
    """
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["message", "token"] = Field(
        description="Event type: 'message' for structured messages, 'token' for streaming tokens"
    )

    # For message events: content is a ChatMessageV2 object
    # For token events: content is a string
    content: Union[ChatMessageV2, str] = Field(
        description="Message content object (type='message') or token string (type='token')"
    )

    # These fields only appear for token events (flattened to root level)
    message_id: Optional[str] = Field(
        default=None,
        validation_alias="messageID",
        description="Message ID (only for token events)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        validation_alias="conversationID",
        description="Conversation ID (only for token events)"
    )
    session_id: Optional[str] = Field(
        default=None,
        validation_alias="sessionID",
        description="Session ID (only for token events)"
    )


class ConversationRequestV2(BaseModel):
    """
    V2 request model for chat endpoints.

    Enhancements over V1:
    - session_id: Track sessions across multiple conversations
    - custom_fields: Platform-specific data (replaces hardcoded slackInfo)
    - Cleaner field naming (message instead of question)
    - Authentication tokens extracted from headers (not request body)

    Note: Authentication credentials (JWT tokens, API keys, etc.) should be
    passed via HTTP headers (e.g., Authorization: Bearer <token>) and are
    extracted by the authentication middleware, not from the request body.
    """
    model_config = ConfigDict(populate_by_name=True)

    # Core fields
    conversation_id: Optional[str] = Field(
        default=None,

        validation_alias="conversationID",
        description="Optional ID of an existing conversation. If omitted, a new conversation is sta"
                    "rted."
    )
    message: str = Field(
        description="The user's message/question"
    )
    session_id: Optional[str] = Field(
        default=None,
        validation_alias="sessionID",
        description="Optional Session ID to track user sessions across multiple conversations"
    )

    # Optional metadata
    user_info: Optional[UserInfo] = Field(
        default=None,
        validation_alias="userInfo",
        description="User name and email"
    )
    platform: Optional[str] = Field(
        default=None,
        description="Platform where the request originated (slack, web, mobile, etc.)"
    )
    mode: Optional[Literal["deep_research"]] = Field(
        default=None,
        description="Optional chat mode (for example: deep_research)"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific data (slack_user_id, slack_channel_id, etc.)"
    )

    # Additional fields
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of the request"
    )


class MessageHistoryResponseV2(BaseModel):
    """
    V2 response model for message history endpoint.

    Returns all messages in a conversation, with full support for
    tool calls, metadata, and structured references.

    Use case: GET /api/v2/conversations/{conversation_id}/messages
    """
    model_config = ConfigDict(populate_by_name=True)

    messages: List[ChatMessageV2] = Field(
        description="List of all messages in the conversation, in chronological order"
    )
    conversation_id: str = Field(
        validation_alias="conversationID",
        description="ID of the conversation these messages belong to"
    )
    session_id: str = Field(
        validation_alias="sessionID",
        description="ID of the session"
    )
    total_count: Optional[int] = Field(
        default=None,
        validation_alias="totalCount",
        serialization_alias="totalCount",
        description="Total number of messages (for pagination support)"
    )
    page: Optional[int] = Field(
        default=None,
        description="Current page number (1-indexed)"
    )
    page_size: Optional[int] = Field(
        default=None,
        validation_alias="pageSize",
        serialization_alias="pageSize",
        description="Number of messages per page"
    )


class FeedbackRequestV2(BaseModel):
    """
    V2 request model for feedback submission.

    Enhancements over V1:
    - type: Categorize feedback (positive, negative, bug-report)
    - score: Numeric rating (0.0-1.0 or 1-5)
    - category: Detailed feedback category for negative feedback
    - custom_fields: Platform-specific feedback data

    Use case: POST /api/v2/conversations/{conversation_id}/messages/{message_id}/feedback
    """
    model_config = ConfigDict(populate_by_name=True)

    # Feedback content
    comment: Optional[str] = Field(
        default=None,
        description="Optional text comment from the user"
    )
    type: Literal["positive", "negative", "bug-report"] = Field(
        description="Category of feedback"
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Numeric rating (0.0 to 1.0 scale). Use 0.0-0.2 for 1 star, 0.2-0.4 for 2 stars, etc."
    )
    category: Optional[str] = Field(
        default=None,
        description="Feedback category ID matching one of the agent's configured feedback categories"
    )

    # Extensibility
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific feedback metadata (e.g., slack_reaction, sentiment_score)"
    )


class FeedbackResponseV2(BaseModel):
    """
    V2 response model for feedback submission.

    Returns confirmation with metadata about who/when the feedback was recorded.
    """
    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(
        validation_alias="messageID",
        description="ID of the message that received feedback"
    )
    modifier_details: Dict[str, Any] = Field(
        validation_alias="modifierDetails",
        description="Metadata about feedback recording (created_at, created_by, updated_at, updated_by)"
    )
    additional_references: List[Source] = Field(
        default_factory=list,
        validation_alias="additionalReferences",
        description="Additional resources or documentation related to the feedback"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific response data"
    )


class ConversationV2(BaseModel):
    """
    V2 model for conversation metadata in list operations.

    This represents a single conversation item in the conversations list,
    containing metadata only (not the actual messages).

    Use case: Each item in GET /api/v2/conversations response
    """
    model_config = ConfigDict(populate_by_name=True)

    # IDs
    conversation_id: str = Field(
        validation_alias="conversationID",
        description="Unique identifier for the conversation"
    )
    session_id: str = Field(
        validation_alias="sessionID",
        description="Session ID this conversation belongs to"
    )

    # Metadata
    title: str = Field(
        description="Display title for the conversation (e.g., 'Weather in Tokyo')"
    )
    platform: Optional[str] = Field(
        default=None,
        description="Platform where the conversation originated (slack, web, mobile, etc.)"
    )
    user_info: Optional[UserInfo] = Field(
        default=None,
        validation_alias="userInfo",
        description="Information about the user who created the conversation"
    )

    # Audit trail
    modifier_details: Dict[str, Any] = Field(
        validation_alias="modifierDetails",
        description="Audit metadata (created_at, updated_at, created_by, updated_by, last_message_at)"
    )

    # Extensibility
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific metadata (e.g., slack_channel_id, last_message_preview, unread_count)"
    )


class ConversationListResponseV2(BaseModel):
    """
    V2 response model for conversation list endpoint.

    Returns a list of conversations with metadata (no messages).
    Messages are fetched separately using MessageHistoryResponseV2.

    Use case: GET /api/v2/conversations
    """
    model_config = ConfigDict(populate_by_name=True)

    conversations: List[ConversationV2] = Field(
        description="List of conversations with metadata"
    )
    total_count: Optional[int] = Field(
        default=None,
        validation_alias="totalCount",
        serialization_alias="totalCount",
        description="Total number of conversations available (for pagination)"
    )
    page: Optional[int] = Field(
        default=None,
        description="Current page number (for pagination)"
    )
    page_size: Optional[int] = Field(
        default=None,
        validation_alias="pageSize",
        serialization_alias="pageSize",
        description="Number of items per page (for pagination)"
    )


class DeleteConversationResponseV2(BaseModel):
    """
    V2 response model for conversation deletion endpoint.

    Returns confirmation that the conversation was deleted with metadata
    about when and by whom.

    Use case: DELETE /api/v2/conversations/{conversation_id}
    """
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str = Field(
        validation_alias="conversationID",
        description="ID of the conversation that was deleted"
    )
    status: Literal["deleted", "Failed"] = Field(
        default="deleted",
        description="Status of the deletion operation"
    )
    modifier_details: Dict[str, Any] = Field(
        validation_alias="modifierDetails",
        description="Metadata about the deletion (deleted_at, deleted_by)"
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="customFields",
        description="Platform-specific response data"
    )


# ============================================
# Data Sources API Models (V2 Only)
# ============================================


class DataCollection(BaseModel):
    """
    Model for data collection information.

    Used for both live collections (with optional lastUpdated timestamp) and
    upcoming collections (without timestamp). When serializing to JSON, use
    exclude_none=True to omit null lastUpdated values for cleaner output.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        description="Name of the data collection"
    )
    last_updated: Optional[str] = Field(
        default=None,
        validation_alias="lastUpdated",
        serialization_alias="lastUpdated",
        description="Human-readable date when the data collection was last updated (optional, omit for upcoming collections)"
    )


class DataSourcesResponseV2(BaseModel):
    """
    V2 response model for data sources endpoint.

    Returns information about data collections used by an agent,
    including both live collections (with optional refresh timestamps) and
    upcoming collections (planned but not yet live).

    Use case: GET /api/v2/conversations/data/sources
    """
    model_config = ConfigDict(populate_by_name=True)

    live_collections: List[DataCollection] = Field(
        default_factory=list,
        validation_alias="liveCollections",
        serialization_alias="liveCollections",
        description="List of currently active data collections with optional last updated timestamps"
    )
    upcoming_collections: List[DataCollection] = Field(
        default_factory=list,
        validation_alias="upcomingCollections",
        serialization_alias="upcomingCollections",
        description="List of planned data collections not yet live"
    )


# ============================================
# Feedback Categories API Models (V2 Only)
# ============================================


class FeedbackCategoryItem(BaseModel):
    """
    Model for a single feedback category option.

    Represents one selectable feedback category that the UI can render,
    including whether a comment is required when this category is selected.

    Example:
        {
            "code": "incorrect_or_outdated_info",
            "label": "Incorrect or outdated info",
            "commentRequired": false
        }
    """
    model_config = ConfigDict(populate_by_name=True)

    code: str = Field(
        description="Unique identifier for the feedback category (e.g., 'incorrect_or_outdated_info')"
    )
    label: str = Field(
        description="Human-readable display label for the category (e.g., 'Incorrect or outdated info')"
    )
    comment_required: bool = Field(
        default=False,
        validation_alias="commentRequired",
        serialization_alias="commentRequired",
        description="Whether a comment is required when this category is selected"
    )


class FeedbackCategoriesResponseV2(BaseModel):
    """
    V2 response model for feedback categories endpoint.

    Returns the list of feedback categories an agent supports,
    allowing the UI to dynamically render feedback options.

    Use case: GET /api/v2/conversations/feedback/categories
    """
    model_config = ConfigDict(populate_by_name=True)

    feedback_categories: List[FeedbackCategoryItem] = Field(
        default_factory=list,
        validation_alias="feedbackCategories",
        serialization_alias="feedbackCategories",
        description="List of feedback categories the agent supports"
    )