"""Shadowbot conversation list and message history from checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from shadowbot_agent_api import UserContext
from shadowbot_agent_api.models_v2 import ChatMessageV2, ConversationV2

from template_agent.src.core.agent_utils import langchain_to_chat_message
from template_agent.src.core.storage import (
    ThreadMeta,
    get_thread_meta,
    get_user_thread_metas,
    thread_owned_by_user,
)
from template_agent.src.routes.common import utc_iso_z
from template_agent.src.schema import ChatMessage
from template_agent.src.settings import settings


def title_from_message(text: str, *, max_len: int = 80) -> str:
    """Build a sidebar title from the first user message."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _parse_platform_filter(platform: Optional[str]) -> Optional[set[str]]:
    if not platform or not str(platform).strip():
        return None
    return {p.strip().lower() for p in str(platform).split(",") if p.strip()}


def _matches_platform(meta: ThreadMeta, allowed: Optional[set[str]]) -> bool:
    if not allowed:
        return True
    if not meta.platform:
        return True
    return meta.platform.strip().lower() in allowed


def load_thread_chat_messages(thread_id: str) -> List[ChatMessage]:
    """Load messages for a thread from the active checkpointer."""
    if settings.USE_INMEMORY_SAVER:
        return _load_messages_inmemory(thread_id)
    return _load_messages_postgres(thread_id)


def _load_messages_inmemory(thread_id: str) -> List[ChatMessage]:
    from template_agent.src.core.storage import get_shared_checkpointer

    checkpointer = get_shared_checkpointer()
    config = RunnableConfig(configurable={"thread_id": thread_id, "checkpoint_ns": ""})
    state_history = list(checkpointer.list(config))
    if not state_history:
        return []

    chat_messages: List[ChatMessage] = []
    latest = state_history[-1]
    if latest.checkpoint and "channel_values" in latest.checkpoint:
        channel_values = latest.checkpoint["channel_values"]
        if "messages" in channel_values:
            for message in channel_values["messages"]:
                try:
                    chat_messages.append(langchain_to_chat_message(message))
                except (ValueError, TypeError):
                    continue

    if chat_messages:
        return chat_messages

    seen: set[tuple[str, str]] = set()
    for checkpoint_tuple in state_history:
        if not checkpoint_tuple.checkpoint:
            continue
        channel_values = checkpoint_tuple.checkpoint.get("channel_values") or {}
        for message in channel_values.get("messages") or []:
            try:
                chat_message = langchain_to_chat_message(message)
            except (ValueError, TypeError):
                continue
            key = (chat_message.type, chat_message.content)
            if key in seen:
                continue
            seen.add(key)
            chat_messages.append(chat_message)
    return chat_messages


def _load_messages_postgres(thread_id: str) -> List[ChatMessage]:
    import psycopg

    if not settings.DATABASE_URL:
        return []

    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT checkpoint
                    FROM checkpoints
                    WHERE thread_id = %s
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                row = cur.fetchone()
    except Exception:
        return []

    if not row or not row[0]:
        return []

    checkpoint_data = row[0]
    channel_values = checkpoint_data.get("channel_values") or {}
    messages = channel_values.get("messages") or []
    chat_messages: List[ChatMessage] = []
    for message in messages:
        try:
            chat_messages.append(langchain_to_chat_message(message))
        except (ValueError, TypeError):
            continue
    return chat_messages


def _postgres_thread_ids_for_user(user_id: str) -> List[str]:
    import psycopg

    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE metadata->>'user_id' = %s
                    ORDER BY thread_id
                    """,
                    (user_id,),
                )
                return [row[0] for row in cur.fetchall() if row[0]]
    except Exception:
        return []


@dataclass
class PaginatedConversations:
    items: List[ConversationV2]
    total_count: int


def list_conversations_for_user(
    user_id: str,
    *,
    page: int = 1,
    page_size: int = 20,
    platform: Optional[str] = None,
    user: Optional[UserContext] = None,
) -> PaginatedConversations:
    """Build paginated ``ConversationV2`` list for Shadowbot sidebar."""
    allowed = _parse_platform_filter(platform)
    metas = get_user_thread_metas(user_id)

    if not metas and not settings.USE_INMEMORY_SAVER:
        for thread_id in _postgres_thread_ids_for_user(user_id):
            metas.append(
                ThreadMeta(
                    thread_id=thread_id,
                    session_id=thread_id,
                    user_id=user_id,
                    title=f"Conversation {thread_id[:8]}",
                    platform=None,
                    created_at=utc_iso_z(),
                    updated_at=utc_iso_z(),
                )
            )

    filtered = [m for m in metas if _matches_platform(m, allowed)]
    filtered.sort(key=lambda m: m.updated_at, reverse=True)
    total = len(filtered)
    start = max(0, (page - 1) * page_size)
    page_items = filtered[start : start + page_size]

    user_email = user.email if user and user.email else user_id
    conversations: List[ConversationV2] = []
    for meta in page_items:
        preview = meta.last_message_preview
        conversations.append(
            ConversationV2(
                conversation_id=meta.thread_id,
                session_id=meta.session_id,
                title=meta.title,
                platform=meta.platform,
                user_info=None,
                modifier_details={
                    "created_at": meta.created_at,
                    "updated_at": meta.updated_at,
                    "created_by": user_email,
                    "updated_by": user_email,
                    "last_message_at": meta.updated_at,
                },
                custom_fields={"last_message_preview": preview} if preview else {},
            )
        )

    return PaginatedConversations(items=conversations, total_count=total)


def chat_message_to_v2(
    msg: ChatMessage,
    *,
    conversation_id: str,
    session_id: str,
    index: int,
) -> ChatMessageV2:
    """Map internal ``ChatMessage`` to Shadowbot ``ChatMessageV2``."""
    message_id = msg.run_id or f"{conversation_id}-{index}"
    metadata = dict(msg.response_metadata or {})
    if msg.type == "ai" and "finish_reason" not in metadata:
        metadata["finish_reason"] = "stop"

    return ChatMessageV2(
        type=msg.type,
        content=msg.content or "",
        message_id=message_id,
        conversation_id=conversation_id,
        session_id=session_id,
        tool_calls=list(msg.tool_calls or []),
        tool_call_id=msg.tool_call_id,
        response_metadata=metadata,
        references=[],
        images=[],
        custom_fields={},
        timestamp=utc_iso_z(),
    )


def history_message_types_for_ui() -> set[str]:
    """Message types shown in Shadowbot history replay."""
    return {"human", "ai", "tool"}


@dataclass
class PaginatedMessages:
    items: List[ChatMessageV2]
    session_id: str
    total_count: int


def list_messages_for_conversation(
    user_id: str,
    conversation_id: str,
    *,
    page: int = 1,
    page_size: int = 50,
) -> PaginatedMessages:
    """Return paginated V2 messages for a conversation the user owns."""
    if not thread_owned_by_user(user_id, conversation_id):
        if settings.USE_INMEMORY_SAVER:
            return PaginatedMessages(items=[], session_id=conversation_id, total_count=0)
        if conversation_id not in _postgres_thread_ids_for_user(user_id):
            return PaginatedMessages(items=[], session_id=conversation_id, total_count=0)

    meta = get_thread_meta(conversation_id)
    session_id = meta.session_id if meta else conversation_id
    raw_messages = load_thread_chat_messages(conversation_id)
    allowed = history_message_types_for_ui()
    filtered = [m for m in raw_messages if m.type in allowed and (m.content or m.type == "tool")]

    total = len(filtered)
    start = max(0, (page - 1) * page_size)
    page_slice = filtered[start : start + page_size]
    v2_messages = [
        chat_message_to_v2(
            msg,
            conversation_id=conversation_id,
            session_id=session_id,
            index=start + i,
        )
        for i, msg in enumerate(page_slice)
    ]
    return PaginatedMessages(
        items=v2_messages,
        session_id=session_id,
        total_count=total,
    )
