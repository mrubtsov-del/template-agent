"""Custom exceptions for the Shadowbot Agent API.

This module defines custom exception classes that preserve detailed error information
and provide better error handling than generic HTTPException.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException


class AgentException(HTTPException):
    """Base exception class for agent-related errors.

    This exception preserves detailed error information that can be
    returned to the client, unlike generic HTTPException which loses details.

    Args:
        status_code: HTTP status code
        detail: Human-readable error message
        error_code: Optional application-specific error code
        error_data: Optional additional error data/context
        headers: Optional HTTP headers to include in response
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.error_data = error_data or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary for JSON serialization."""
        result = {
            "detail": self.detail,
            "status_code": self.status_code,
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_data:
            result["error_data"] = self.error_data
        return result


class ConversationNotFoundException(AgentException):
    """Raised when a conversation is not found."""

    def __init__(
        self,
        conversation_id: str,
        detail: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=404,
            detail=detail or f"Conversation not found: {conversation_id}",
            error_code="CONVERSATION_NOT_FOUND",
            error_data={**(error_data or {}), "conversation_id": conversation_id}
        )


class MessageNotFoundException(AgentException):
    """Raised when a message is not found."""

    def __init__(
        self,
        message_id: str,
        conversation_id: Optional[str] = None,
        detail: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None
    ):
        data = {**(error_data or {}), "message_id": message_id}
        if conversation_id:
            data["conversation_id"] = conversation_id

        super().__init__(
            status_code=404,
            detail=detail or f"Message not found: {message_id}",
            error_code="MESSAGE_NOT_FOUND",
            error_data=data
        )


class SessionNotFoundException(AgentException):
    """Raised when a session is not found."""

    def __init__(
        self,
        session_id: str,
        detail: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=404,
            detail=detail or f"Session not found: {session_id}",
            error_code="SESSION_NOT_FOUND",
            error_data={**(error_data or {}), "session_id": session_id}
        )


class InvalidRequestException(AgentException):
    """Raised when request data is invalid."""

    def __init__(
        self,
        detail: str,
        error_code: str = "INVALID_REQUEST",
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code=error_code,
            error_data=error_data
        )


class AuthenticationException(AgentException):
    """Raised when authentication fails."""

    def __init__(
        self,
        detail: str = "Authentication required",
        error_code: str = "AUTHENTICATION_REQUIRED",
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=401,
            detail=detail,
            error_code=error_code,
            error_data=error_data,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationException(AgentException):
    """Raised when user is not authorized to perform an action."""

    def __init__(
        self,
        detail: str = "Not authorized to perform this action",
        error_code: str = "AUTHORIZATION_FAILED",
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=403,
            detail=detail,
            error_code=error_code,
            error_data=error_data
        )


class AgentProcessingException(AgentException):
    """Raised when agent processing fails."""

    def __init__(
        self,
        detail: str,
        error_code: str = "AGENT_PROCESSING_ERROR",
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code=error_code,
            error_data=error_data
        )


class RateLimitException(AgentException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        error_data: Optional[Dict[str, Any]] = None
    ):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        data = error_data or {}
        if retry_after:
            data["retry_after"] = retry_after

        super().__init__(
            status_code=429,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            error_data=data,
            headers=headers
        )


class ToolExecutionException(AgentException):
    """Raised when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        detail: str,
        error_code: str = "TOOL_EXECUTION_ERROR",
        error_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code=error_code,
            error_data={**(error_data or {}), "tool_name": tool_name}
        )
