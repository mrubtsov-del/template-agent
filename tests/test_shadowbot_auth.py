"""Tests for Shadowbot JWT bootstrap and RHAuth header forwarding."""

import pytest
from shadowbot_agent_api import AuthConfig, configure_auth
from shadowbot_agent_api.auth import get_optional_user
from shadowbot_agent_api.models import UserContext
from starlette.requests import Request

from template_agent.src.core import shadowbot_auth
from template_agent.src.settings import settings


def _request_with_header(name: str, value: str) -> Request:
    scope = {
        "type": "http",
        "headers": [(name.lower().encode(), value.encode())],
        "method": "POST",
        "path": "/api/v2/conversations/chat",
    }
    return Request(scope)


class TestConfigureShadowbotAuth:
    def test_disabled_when_auth_enabled_false(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTH_ENABLED", False)
        assert shadowbot_auth.configure_shadowbot_auth_from_settings() is False

    def test_validate_raises_when_enabled_but_incomplete(self, monkeypatch):
        from template_agent.src.core.exceptions.exceptions import AppException

        monkeypatch.setattr(settings, "AUTH_ENABLED", True)
        monkeypatch.setattr(settings, "AUTH_ISSUER", None)
        with pytest.raises(AppException, match="AUTH_ISSUER"):
            shadowbot_auth.validate_shadowbot_auth_settings(settings)

    def test_normalize_issuer_strips_trailing_slash(self):
        assert (
            shadowbot_auth.normalize_issuer(
                "https://auth.redhat.com/auth/realms/EmployeeIDP/"
            )
            == "https://auth.redhat.com/auth/realms/EmployeeIDP"
        )

    def test_enabled_when_settings_complete(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTH_ENABLED", True)
        monkeypatch.setattr(
            settings,
            "AUTH_ISSUER",
            "https://auth.redhat.com/auth/realms/EmployeeIDP",
        )
        monkeypatch.setattr(settings, "AUTH_AUDIENCE", "account")
        monkeypatch.setattr(
            settings,
            "AUTH_JWKS_URL",
            "https://auth.redhat.com/auth/realms/EmployeeIDP/protocol/openid-connect/certs",
        )
        assert shadowbot_auth.configure_shadowbot_auth_from_settings() is True


class TestRhAuthHeader:
    @pytest.mark.asyncio
    async def test_get_optional_user_accepts_raw_authorization_jwt(self, monkeypatch):
        from shadowbot_agent_api.auth import _jwt_string_from_request

        scope = {
            "type": "http",
            "headers": [(b"authorization", b"aaa.bbb.ccc")],
        }
        request = Request(scope)
        assert _jwt_string_from_request(request, None) == "aaa.bbb.ccc"

    @pytest.mark.asyncio
    async def test_get_optional_user_accepts_x_authorization_rhauth(self, monkeypatch):
        configure_auth(
            AuthConfig(
                enabled=True,
                issuer="https://auth.redhat.com/auth/realms/EmployeeIDP",
                audience="account",
                jwks_url="https://auth.redhat.com/auth/realms/EmployeeIDP/protocol/openid-connect/certs",
            )
        )

        async def _fake_verify(_token: str, _config: AuthConfig) -> UserContext:
            return UserContext(sub="user-1", email="analyst@example.com")

        monkeypatch.setattr(
            "shadowbot_agent_api.auth.verify_token",
            _fake_verify,
        )

        request = _request_with_header(
            "x-authorization-rhauth", "Bearer test-jwt-token"
        )
        user = await get_optional_user(request, None)
        assert user is not None
        assert user.email == "analyst@example.com"

    @pytest.mark.asyncio
    async def test_invalid_rhauth_token_returns_401_not_silent_anon(self, monkeypatch):
        from fastapi import HTTPException

        configure_auth(
            AuthConfig(
                enabled=True,
                issuer="https://auth.redhat.com/auth/realms/EmployeeIDP",
                audience="account",
                jwks_url="https://auth.redhat.com/auth/realms/EmployeeIDP/protocol/openid-connect/certs",
            )
        )

        async def _fail_verify(_token: str, _config: AuthConfig) -> UserContext:
            from shadowbot_agent_api.auth import AuthenticationError

            raise AuthenticationError("Invalid token: issuer mismatch")

        monkeypatch.setattr(
            "shadowbot_agent_api.auth.verify_token",
            _fail_verify,
        )

        request = _request_with_header("x-authorization-rhauth", "bad-token")
        with pytest.raises(HTTPException) as excinfo:
            await get_optional_user(request, None)
        assert excinfo.value.status_code == 401
        assert "issuer mismatch" in str(excinfo.value.detail)
