"""Tests for Shadowbot V2 contract handlers.

V2 handlers use @require_auth (per the shadowbot-agent-api skill guidance), so
endpoint-level tests must either:
  - bypass auth via FastAPI's dependency_overrides (happy paths), or
  - send unauthenticated requests and expect 401 (auth-gating tests).

Two layers like in test_shadowbot_v1.py:

1. Direct handler invocation (unit): call async functions with patched
   AgentManager, verify ChatMessageV2 / StreamEventV2 / etc. shape.

2. Endpoint-level (integration): mount vendor's chat_api_router_v2 and hit
   it via TestClient.
"""

from typing import Any, AsyncGenerator, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shadowbot_agent_api import UserContext, chat_api_router_v2
from shadowbot_agent_api.auth import get_optional_user
from shadowbot_agent_api.models_v2 import (
    ConversationRequestV2,
    FeedbackRequestV2,
)

# Importing these modules triggers @*_handler_v2() decorators -> registers
# V2 business logic in the shadowbot_agent_api handler registry.
from template_agent.src.routes import (  # noqa: F401
    chat_v2 as v2_chat,
    conversations_v2 as v2_conversations,
    data_sources_v2 as v2_data_sources,
    delete_conversation_v2 as v2_delete,
    feedback_categories_v2 as v2_feedback_categories,
    messages_v2 as v2_messages,
    shadowbot_feedback_v2 as v2_feedback,
    shadowbot_stream_v2 as v2_stream,
)


# ---------- Fakes & helpers ----------------------------------------------------


class _FakeAgentManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


# ---------- 1) Direct handler invocation ---------------------------------------


class TestV2ChatHandler:
    @pytest.mark.asyncio
    async def test_returns_chat_message_v2(self, monkeypatch):
        monkeypatch.setattr(
            v2_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager()
        )

        req = ConversationRequestV2(message="Hi", sessionID="sess-1")
        msg = await v2_chat.handle_chat_request_v2(req, user=None)

        assert msg.type == "ai"
        assert msg.content == "Hello world"
        assert msg.session_id == "sess-1"
        assert msg.conversation_id  # auto-generated
        assert msg.message_id
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.tool_calls == []
        assert msg.references == []
        assert msg.images == []
        assert msg.timestamp.endswith("Z")

    @pytest.mark.asyncio
    async def test_session_id_defaults_to_conversation_id(self, monkeypatch):
        monkeypatch.setattr(
            v2_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager()
        )

        req = ConversationRequestV2(message="Hi", conversationID="c-7")
        msg = await v2_chat.handle_chat_request_v2(req, user=None)
        assert msg.conversation_id == "c-7"
        assert msg.session_id == "c-7"

    @pytest.mark.asyncio
    async def test_handler_uses_kwarg_param_name(self, monkeypatch):
        """Regression: handler param must be `request_body`, not `request`.

        Vendor's V2 dispatcher injects the FastAPI `Request` object as kwarg
        named `request` whenever a parameter with that name exists in the
        handler signature, which collides with positional `ConversationRequestV2`.
        The skill convention is `request_body`.
        """
        monkeypatch.setattr(
            v2_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager()
        )
        import inspect

        params = inspect.signature(v2_chat.handle_chat_request_v2).parameters
        assert "request_body" in params
        assert "request" not in params

        stream_params = inspect.signature(
            v2_stream.handle_stream_chat_v2
        ).parameters
        assert "request_body" in stream_params
        assert "request" not in stream_params

        fb_params = inspect.signature(v2_feedback.handle_feedback_v2).parameters
        assert "feedback_data" in fb_params
        assert "request" not in fb_params

    @pytest.mark.asyncio
    async def test_handlers_accept_custom_auth_param(self):
        import inspect

        handlers = (
            v2_chat.handle_chat_request_v2,
            v2_stream.handle_stream_chat_v2,
            v2_conversations.handle_get_conversations_v2,
            v2_messages.handle_get_messages_v2,
            v2_feedback.handle_feedback_v2,
            v2_feedback_categories.handle_get_feedback_categories_v2,
            v2_data_sources.handle_get_data_sources_v2,
            v2_delete.handle_delete_conversation_v2,
        )
        for handler in handlers:
            assert "custom_auth" in inspect.signature(handler).parameters


class TestV2StreamHandler:
    @pytest.mark.asyncio
    async def test_emits_token_events_then_final_message(self, monkeypatch):
        monkeypatch.setattr(
            v2_stream, "build_agent_manager", lambda *a, **k: _FakeAgentManager()
        )

        req = ConversationRequestV2(message="Hi", sessionID="sess-1")
        events = [
            event
            async for event in v2_stream.handle_stream_chat_v2(req, user=None)
        ]

        assert events[0].type == "token"
        assert events[0].content == "Hello "
        assert events[-1].type == "message"

        final = events[-1].content  # ChatMessageV2
        assert final.content == "Hello world"
        assert final.session_id == "sess-1"
        assert final.timestamp.endswith("Z")

    @pytest.mark.asyncio
    async def test_message_only_ai_fallback_emits_token_and_final(self, monkeypatch):
        """When AgentManager yields no tokens, AI text from updates must still stream."""

        class _MessageOnlyManager:
            async def stream_response(self, _request):
                yield {
                    "type": "message",
                    "content": {"type": "ai", "content": "Answer without tokens"},
                }

        monkeypatch.setattr(
            v2_stream,
            "build_agent_manager",
            lambda *a, **k: _MessageOnlyManager(),
        )
        events = [
            e
            async for e in v2_stream.handle_stream_chat_v2(
                ConversationRequestV2(message="Hi"), user=None
            )
        ]
        assert [e.type for e in events] == ["token", "message"]
        assert events[0].content == "Answer without tokens"
        assert events[-1].content.content == "Answer without tokens"

    @pytest.mark.asyncio
    async def test_agent_error_event_surfaces_in_stream(self, monkeypatch):
        class _ErrorManager:
            async def stream_response(self, _request):
                yield {
                    "type": "error",
                    "content": {
                        "message": "Snowflake failed",
                        "error_type": "agent_error",
                    },
                }

        monkeypatch.setattr(
            v2_stream, "build_agent_manager", lambda *a, **k: _ErrorManager()
        )
        events = [
            e
            async for e in v2_stream.handle_stream_chat_v2(
                ConversationRequestV2(message="Hi"), user=None
            )
        ]
        assert events[0].type == "token"
        assert "Snowflake failed" in events[0].content
        assert events[-1].type == "message"
        assert events[-1].content.response_metadata["finish_reason"] == "error"


class TestV2RetrievalHandlers:
    @pytest.mark.asyncio
    async def test_conversations_returns_paginated_empty(self):
        resp = await v2_conversations.handle_get_conversations_v2(
            page=2, page_size=10
        )
        assert resp.conversations == []
        assert resp.total_count == 0
        assert resp.page == 2
        assert resp.page_size == 10

    @pytest.mark.asyncio
    async def test_messages_returns_paginated_empty(self):
        resp = await v2_messages.handle_get_messages_v2(
            conversation_id="conv-x", page=1, page_size=50
        )
        assert resp.conversation_id == "conv-x"
        assert resp.messages == []
        assert resp.total_count == 0


class TestV2FeedbackHandler:
    @pytest.mark.asyncio
    async def test_feedback_returns_modifier_details(self):
        fb = FeedbackRequestV2(type="positive", score=0.8, comment="nice")
        resp = await v2_feedback.handle_feedback_v2(
            conversation_id="conv-x",
            message_id="msg-y",
            feedback_data=fb,
            user=None,
        )
        assert resp.message_id == "msg-y"
        assert "created_at" in resp.modifier_details
        assert "created_by" in resp.modifier_details
        assert resp.additional_references == []


class TestV2FeedbackCategories:
    @pytest.mark.asyncio
    async def test_returns_predefined_categories(self):
        resp = await v2_feedback_categories.handle_get_feedback_categories_v2()
        codes = {item.code for item in resp.feedback_categories}

        # Standard Shadowbot categories must be present so the UI rendering
        # matches what other agents emit.
        assert {
            "missing_content",
            "incorrect_or_outdated_info",
            "error_or_outage",
            "unclear_or_low_quality_answer",
            "feature_request",
            "other",
        }.issubset(codes)

        # Snowflake-specific addition for SQL-generation feedback.
        assert "wrong_sql" in codes

        by_code = {item.code: item for item in resp.feedback_categories}
        # Free-text categories require a comment per Shadowbot convention.
        assert by_code["feature_request"].comment_required is True
        assert by_code["other"].comment_required is True
        assert by_code["wrong_sql"].comment_required is True
        # Discrete categories don't need a comment.
        assert by_code["incorrect_or_outdated_info"].comment_required is False
        assert by_code["missing_content"].comment_required is False


class TestV2DeleteConversation:
    @pytest.mark.asyncio
    async def test_delete_returns_deleted_status(self, monkeypatch):
        from template_agent.src import settings as settings_mod
        from template_agent.src.core import storage

        monkeypatch.setattr(settings_mod.settings, "USE_INMEMORY_SAVER", True)
        storage.register_thread("dev@example.com", "conv-del-1")

        user = UserContext(sub="u-1", email="dev@example.com")
        resp = await v2_delete.handle_delete_conversation_v2(
            "conv-del-1",
            user=user,
        )
        assert resp.conversation_id == "conv-del-1"
        assert resp.status == "deleted"
        assert resp.modifier_details["deleted_by"] == "dev@example.com"
        assert resp.modifier_details["registry_removed"] is True
        assert "conv-del-1" not in storage.get_user_threads("dev@example.com")

    def test_delete_endpoint_via_router(self):
        r = _make_v2_client().delete("/api/v2/conversations/conv-xyz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "deleted"
        assert data["conversation_id"] == "conv-xyz"


class TestV2DataSources:
    @pytest.mark.asyncio
    async def test_lists_snowflake_as_live_collection(self):
        resp = await v2_data_sources.handle_get_data_sources_v2()
        assert len(resp.live_collections) >= 1
        # The single collection name should mention Snowflake regardless of
        # whether DB/schema settings are filled in or empty.
        assert "Snowflake" in resp.live_collections[0].name
        # last_updated should be populated with today's date in YYYY-MM-DD form.
        assert resp.live_collections[0].last_updated is not None
        assert len(resp.live_collections[0].last_updated) == 10  # "YYYY-MM-DD"
        assert resp.upcoming_collections == []


# ---------- 2) Endpoint-level integration -------------------------------------


_FAKE_V2_USER = UserContext(sub="u-1", email="dev@example.com")


def _make_v2_client(authed: bool = True) -> TestClient:
    """Mount vendor's V2 router. When authed=True, bypass JWT via override."""
    app = FastAPI()
    app.include_router(chat_api_router_v2)
    if authed:
        app.dependency_overrides[get_optional_user] = lambda: _FAKE_V2_USER
    return TestClient(app)


class TestV2Endpoints:
    """Happy paths with auth bypassed by FastAPI dependency_overrides."""

    def test_chat_endpoint_returns_chat_message_v2(self, monkeypatch):
        """Endpoint accepts camelCase input but currently emits snake_case keys.

        Vendor's ChatMessageV2 declares `validation_alias` without
        `serialization_alias` on its id fields (`session_id`, `message_id`,
        `conversation_id`), so the response uses the Python attribute names.
        """
        monkeypatch.setattr(
            v2_chat, "build_agent_manager", lambda *a, **k: _FakeAgentManager()
        )
        r = _make_v2_client().post(
            "/api/v2/conversations/chat",
            json={"message": "Hi", "sessionID": "sess-1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "ai"
        assert data["content"] == "Hello world"
        assert data["session_id"] == "sess-1"
        assert "message_id" in data
        assert "conversation_id" in data

    def test_feedback_categories_endpoint(self):
        r = _make_v2_client().get(
            "/api/v2/conversations/feedback/categories"
        )
        assert r.status_code == 200
        data = r.json()
        codes = {item["code"] for item in data["feedbackCategories"]}
        # Both the standard Shadowbot set and our Snowflake-specific one.
        assert "missing_content" in codes
        assert "feature_request" in codes
        assert "wrong_sql" in codes

    def test_data_sources_endpoint(self):
        r = _make_v2_client().get("/api/v2/conversations/data/sources")
        assert r.status_code == 200
        data = r.json()
        assert "liveCollections" in data
        assert len(data["liveCollections"]) >= 1
        assert "Snowflake" in data["liveCollections"][0]["name"]
        assert data["liveCollections"][0]["lastUpdated"] is not None


class TestV2AuthGating:
    """All V2 endpoints under @require_auth must return 401 without a token."""

    def test_chat_returns_401(self):
        r = _make_v2_client(authed=False).post(
            "/api/v2/conversations/chat", json={"message": "Hi"}
        )
        assert r.status_code == 401

    def test_get_conversations_returns_401(self):
        r = _make_v2_client(authed=False).get("/api/v2/conversations")
        assert r.status_code == 401

    def test_get_messages_returns_401(self):
        r = _make_v2_client(authed=False).get(
            "/api/v2/conversations/conv-x/messages"
        )
        assert r.status_code == 401

    def test_feedback_returns_401(self):
        r = _make_v2_client(authed=False).post(
            "/api/v2/conversations/conv-x/messages/msg-y/feedback",
            json={"type": "positive", "score": 0.9},
        )
        assert r.status_code == 401

    def test_feedback_categories_returns_401(self):
        r = _make_v2_client(authed=False).get(
            "/api/v2/conversations/feedback/categories"
        )
        assert r.status_code == 401

    def test_data_sources_returns_401(self):
        r = _make_v2_client(authed=False).get(
            "/api/v2/conversations/data/sources"
        )
        assert r.status_code == 401

    def test_delete_returns_401(self):
        r = _make_v2_client(authed=False).delete(
            "/api/v2/conversations/conv-x"
        )
        assert r.status_code == 401
