"""Tests for Shadowbot V1 contract handlers.

Two layers of testing:

1. Direct handler invocation (unit): we call the async functions registered
   by @chat_handler() / @stream_chat_handler() etc. while patching
   AgentManager. This validates business logic without touching FastAPI.

2. Endpoint-level (integration via TestClient): we mount the vendor's
   chat_api_router and verify the routing and @require_auth gating
   actually return 401 when no JWT is supplied.
"""

from typing import Any, AsyncGenerator, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shadowbot_agent_api import (
    ConversationRequest,
    UserContext,
    chat_api_router,
)
from shadowbot_agent_api.models import Feedback

# Importing these modules triggers @*_handler() decorators -> registers
# our V1 business logic in the shadowbot_agent_api handler registry.
from template_agent.src.routes import (  # noqa: F401
    chat as v1_chat,
    conversations as v1_conversations,
    messages as v1_messages,
    shadowbot_feedback as v1_feedback,
    shadowbot_stream as v1_stream,
)


# ---------- Fakes & helpers ----------------------------------------------------


class _FakeAgentManager:
    """Drop-in replacement for AgentManager used in tests.

    Emits two token events plus one full message event so both the sync chat
    path (which only reads the final 'message') and the stream path (which
    reads tokens) get realistic data to assert on.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    async def stream_response(
        self, _request: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "token", "content": "Hello "}
        yield {"type": "token", "content": "world"}
        yield {
            "type": "message",
            "content": {"type": "ai", "content": "Hello world"},
        }


_FAKE_USER = UserContext(sub="u-1", email="dev@example.com")


# ---------- 1) Direct handler invocation ---------------------------------------


class TestV1ChatHandler:
    @pytest.mark.asyncio
    async def test_returns_conversation_response_with_success_marker(
        self, monkeypatch
    ):
        monkeypatch.setattr(v1_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager())

        req = ConversationRequest(question="Hi")
        resp = await v1_chat.handle_chat_request(req, user=_FAKE_USER)

        assert resp.response.answer == "Hello world"
        assert resp.informationSaved == "question saved"
        assert resp.chunkId == 0
        assert resp.streamEnded is True
        assert resp.conversationId  # auto-generated UUID
        assert resp.messageId       # auto-generated UUID

    @pytest.mark.asyncio
    async def test_reuses_provided_conversation_id(self, monkeypatch):
        monkeypatch.setattr(v1_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager())

        req = ConversationRequest(question="Hi", conversationID="conv-123")
        resp = await v1_chat.handle_chat_request(req, user=_FAKE_USER)
        assert resp.conversationId == "conv-123"

    @pytest.mark.asyncio
    async def test_handler_returns_error_envelope_on_exception(self, monkeypatch):
        class _Boom:
            def __init__(self, *a, **kw): pass

            async def stream_response(self, _req):
                raise RuntimeError("boom")
                yield  # pragma: no cover - unreachable

        monkeypatch.setattr(v1_chat, "build_agent_manager", lambda *a, **k: _Boom())

        req = ConversationRequest(question="trigger error")
        resp = await v1_chat.handle_chat_request(req, user=_FAKE_USER)

        assert resp.informationSaved == "error occurred"
        assert "boom" in resp.response.answer


class TestV1StreamHandler:
    @pytest.mark.asyncio
    async def test_emits_stream_chunks_then_stream_end(self, monkeypatch):
        monkeypatch.setattr(v1_stream, "build_agent_manager", lambda *a, **k: _FakeAgentManager())

        req = ConversationRequest(question="Hi", conversationID="c-1")
        events = [
            event
            async for event in v1_stream.handle_stream_chat(req, user=_FAKE_USER)
        ]

        # Two token chunks + final StreamEnd message
        kinds = [e.type for e in events]
        assert kinds.count("token") == 2
        assert kinds[-1] == "message"

        last = events[-1]
        assert last.final_text == "Hello world"
        assert last.conversationId == "c-1"
        assert last.timestamp.endswith("Z")
        assert last.messageReferenceList == []
        assert last.imageReferenceList is None

    @pytest.mark.asyncio
    async def test_handlers_accept_custom_auth_param(self):
        import inspect

        for handler in (
            v1_chat.handle_chat_request,
            v1_stream.handle_stream_chat,
            v1_conversations.handle_get_conversations,
            v1_messages.handle_get_messages,
            v1_feedback.handle_feedback,
        ):
            assert "custom_auth" in inspect.signature(handler).parameters


class TestV1RetrievalHandlers:
    @pytest.mark.asyncio
    async def test_conversations_returns_empty_list(self):
        result = await v1_conversations.handle_get_conversations(user=_FAKE_USER)
        assert result == []

    @pytest.mark.asyncio
    async def test_messages_returns_empty_list(self):
        result = await v1_messages.handle_get_messages(
            conversation_id="conv-x", user=_FAKE_USER
        )
        assert result == []


class TestV1FeedbackHandler:
    @pytest.mark.asyncio
    async def test_feedback_acknowledged(self):
        fb = Feedback(option="thumbs_up", comment="good")
        resp = await v1_feedback.handle_feedback(
            conversation_id="conv-x",
            message_id="msg-y",
            request=fb,
            user=_FAKE_USER,
        )
        assert resp.conversationId == "conv-x"
        assert resp.messageId == "msg-y"
        assert resp.informationSaved == "Feedback logged successfully"


# ---------- 2) Endpoint-level auth gating -------------------------------------


def _make_v1_client() -> TestClient:
    """Mount the vendor V1 router so FastAPI dispatches through it."""
    app = FastAPI()
    app.include_router(chat_api_router)
    return TestClient(app)


class TestV1AuthGating:
    """All five @require_auth endpoints return 401 when no auth is configured."""

    def test_chat_returns_401(self):
        r = _make_v1_client().post(
            "/api/v1/conversations/chat", json={"question": "Hi"}
        )
        assert r.status_code == 401

    def test_get_conversations_returns_401(self):
        r = _make_v1_client().get("/api/v1/conversations/")
        # vendor returns [] when no handler is registered for some reason; we
        # registered ours via @require_auth so we expect 401.
        assert r.status_code == 401

    def test_get_messages_returns_401(self):
        r = _make_v1_client().get("/api/v1/conversations/conv-x/messages")
        assert r.status_code == 401

    def test_feedback_returns_401(self):
        r = _make_v1_client().post(
            "/api/v1/conversations/conv-x/messages/msg-y/feedback",
            json={"option": "thumbs_up"},
        )
        assert r.status_code == 401
