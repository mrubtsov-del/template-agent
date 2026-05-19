"""Global storage management for the template agent system.

This module provides a single global checkpoint instance that persists across
the entire application lifecycle when using in-memory storage mode, plus an
in-memory thread registry and metadata for Shadowbot conversation list APIs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# Global checkpoint instance - single instance for the entire application lifecycle
_global_checkpoint: Optional[InMemorySaver] = None

# user_id -> set of thread_ids
_thread_registry: dict[str, set[str]] = {}

# thread_id -> metadata for Shadowbot GET /conversations
_thread_meta: Dict[str, "ThreadMeta"] = {}


@dataclass
class ThreadMeta:
    """Sidebar metadata for a conversation thread."""

    thread_id: str
    session_id: str
    user_id: str
    title: str
    platform: Optional[str] = None
    created_at: str = field(default_factory=_utc_iso_z)
    updated_at: str = field(default_factory=_utc_iso_z)
    last_message_preview: Optional[str] = None


def get_global_checkpoint() -> InMemorySaver:
    """Get the global in-memory checkpoint instance."""
    global _global_checkpoint
    if _global_checkpoint is None:
        _global_checkpoint = InMemorySaver()
        logger.info("Created global InMemorySaver checkpoint instance")
    return _global_checkpoint


def register_thread(user_id: str, thread_id: str) -> None:
    """Register a thread for a user."""
    global _thread_registry
    if user_id not in _thread_registry:
        _thread_registry[user_id] = set()
    _thread_registry[user_id].add(thread_id)
    logger.info(f"Registered thread {thread_id} for user {user_id}")


def record_thread_activity(
    user_id: str,
    thread_id: str,
    *,
    session_id: Optional[str] = None,
    title_hint: Optional[str] = None,
    platform: Optional[str] = None,
    last_message_preview: Optional[str] = None,
) -> None:
    """Register thread ownership and upsert sidebar metadata."""
    register_thread(user_id, thread_id)
    now = _utc_iso_z()
    effective_session = session_id or thread_id
    global _thread_meta
    existing = _thread_meta.get(thread_id)
    if existing is None:
        title = (title_hint or "").strip() or "New conversation"
        _thread_meta[thread_id] = ThreadMeta(
            thread_id=thread_id,
            session_id=effective_session,
            user_id=user_id,
            title=title,
            platform=platform,
            created_at=now,
            updated_at=now,
            last_message_preview=last_message_preview,
        )
        return

    if title_hint and (existing.title == "New conversation" or not existing.title.strip()):
        existing.title = title_hint.strip()[:80] or existing.title
    if platform:
        existing.platform = platform
    if session_id:
        existing.session_id = session_id
    if last_message_preview:
        existing.last_message_preview = last_message_preview[:200]
    existing.updated_at = now


def get_thread_meta(thread_id: str) -> Optional[ThreadMeta]:
    return _thread_meta.get(thread_id)


def get_user_thread_metas(user_id: str) -> List[ThreadMeta]:
    """All conversation metadata for a user, newest first."""
    thread_ids = _thread_registry.get(user_id, set())
    metas = [_thread_meta[tid] for tid in thread_ids if tid in _thread_meta]
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    return metas


def thread_owned_by_user(user_id: str, thread_id: str) -> bool:
    return thread_id in _thread_registry.get(user_id, set())


def remove_thread_metadata(thread_id: str) -> None:
    global _thread_meta
    _thread_meta.pop(thread_id, None)


def unregister_thread(user_id: str, thread_id: str) -> bool:
    """Remove a thread from the in-memory registry."""
    global _thread_registry
    user_threads = _thread_registry.get(user_id)
    if not user_threads or thread_id not in user_threads:
        return False
    user_threads.discard(thread_id)
    if not user_threads:
        _thread_registry.pop(user_id, None)
    remove_thread_metadata(thread_id)
    logger.info(f"Unregistered thread {thread_id} for user {user_id}")
    return True


def get_user_threads(user_id: str) -> list[str]:
    """Get all threads for a user."""
    global _thread_registry
    threads = list(_thread_registry.get(user_id, set()))
    logger.info(f"Retrieved {len(threads)} threads for user {user_id}: {threads}")
    return threads


def reset_global_storage() -> None:
    """Reset the global checkpoint instance."""
    global _global_checkpoint, _thread_registry, _thread_meta
    _global_checkpoint = None
    _thread_registry = {}
    _thread_meta = {}
    logger.info("Reset global checkpoint instance and thread registry")


# Backward compatibility aliases
get_shared_checkpointer = get_global_checkpoint
get_shared_store = get_global_checkpoint
reset_shared_storage = reset_global_storage
