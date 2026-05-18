"""Shared helpers for Shadowbot V1 and V2 handlers."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from shadowbot_agent_api import ConversationRequest, UserContext
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.schema import StreamRequest

if TYPE_CHECKING:
    from template_agent.src.core.manager import AgentManager
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def utc_iso_z() -> str:
    """UTC timestamp in ISO 8601 with trailing ``Z`` (Shadowbot spec)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def resolve_user_id(
    request: ConversationRequest, user: Optional[UserContext]
) -> str:
    """Best caller identifier for V1 (email > request userInfo > anonymous)."""
    if user and user.email:
        return user.email
    if request.userInfo and request.userInfo.userEmail:
        return request.userInfo.userEmail
    return "anonymous"


def resolve_snowflake_login(
    user: Optional[UserContext],
    request: Optional[ConversationRequest] = None,
) -> Optional[str]:
    """Snowflake username for per-user OAuth; None when unknown."""
    if user and user.email:
        return user.email
    if user and user.preferred_username:
        return user.preferred_username
    if user and user.sub:
        return user.sub
    if request is not None:
        uid = resolve_user_id(request, user)
        if uid != "anonymous":
            return uid
    return None


def snowflake_auth_present(custom_auth: Optional[CustomAuthHeaders]) -> bool:
    """Whether ``X-Authorization-Snowflake`` is present (never log the token)."""
    return resolve_snowflake_request_token(custom_auth) is not None


def resolve_snowflake_request_token(
    custom_auth: Optional[CustomAuthHeaders],
) -> Optional[str]:
    """Return trimmed ``X-Authorization-Snowflake`` value if present."""
    if not custom_auth:
        return None
    raw = custom_auth.get("Snowflake")
    if not raw or not str(raw).strip():
        return None
    return str(raw).strip()


def build_agent_manager(
    custom_auth: Optional[CustomAuthHeaders] = None,
    snowflake_login: Optional[str] = None,
) -> "AgentManager":
    """Construct ``AgentManager`` with request-scoped Snowflake auth."""
    from template_agent.src.core.manager import AgentManager

    return AgentManager(custom_auth=custom_auth, snowflake_login=snowflake_login)


async def collect_final_ai_text(
    manager: "AgentManager", stream_req: StreamRequest
) -> str:
    """Run the agent without token streaming; return the final AI message text."""
    final_text = ""
    async for event in manager.stream_response(stream_req):
        if event.get("type") != "message":
            continue
        content = event.get("content", {})
        if content.get("type") == "ai" and content.get("content"):
            final_text = content["content"]
    return final_text


async def iter_agent_stream_events(
    manager: "AgentManager", stream_req: StreamRequest
) -> AsyncIterator[dict[str, Any]]:
    """Yield raw AgentManager stream events (token / message / error)."""
    async for event in manager.stream_response(stream_req):
        if isinstance(event, dict):
            yield event
