"""Tests for Shadowbot conversation list and message history."""

import pytest

from template_agent.src.core import storage
from template_agent.src.core.conversation_history import (
    list_conversations_for_user,
    list_messages_for_conversation,
    title_from_message,
)


@pytest.fixture(autouse=True)
def _reset_storage():
    storage.reset_global_storage()
    yield
    storage.reset_global_storage()


def test_title_from_message_truncates():
    assert title_from_message("Hello") == "Hello"
    long = "x" * 100
    assert title_from_message(long).endswith("...")
    assert len(title_from_message(long)) == 80


def test_list_conversations_paginated():
    storage.record_thread_activity(
        "user@example.com", "c1", title_hint="First", platform="web"
    )
    storage.record_thread_activity(
        "user@example.com", "c2", title_hint="Second", platform="mobile"
    )
    page1 = list_conversations_for_user(
        "user@example.com", page=1, page_size=1, platform="web,mobile"
    )
    assert page1.total_count == 2
    assert len(page1.items) == 1


def test_list_messages_maps_checkpoint_messages(monkeypatch):
    from template_agent.src.schema import ChatMessage

    storage.record_thread_activity("user@example.com", "thread-1", session_id="s1")
    monkeypatch.setattr(
        "template_agent.src.core.conversation_history.load_thread_chat_messages",
        lambda _thread_id: [
            ChatMessage(type="human", content="Hi"),
            ChatMessage(type="ai", content="Hello there"),
        ],
    )

    history = list_messages_for_conversation(
        "user@example.com", "thread-1", page=1, page_size=50
    )
    assert history.total_count == 2
    assert history.items[0].type == "human"
    assert history.items[1].content == "Hello there"
    assert history.session_id == "s1"


def test_messages_denied_for_other_user():
    storage.record_thread_activity("owner@example.com", "private-conv")
    result = list_messages_for_conversation(
        "other@example.com", "private-conv", page=1, page_size=50
    )
    assert result.total_count == 0
    assert result.items == []
