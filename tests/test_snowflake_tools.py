"""Tests for Snowflake tool validation and query execution helpers."""

from contextlib import contextmanager

import pytest
from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.core.exceptions.exceptions import AppException
from template_agent.src.core.tools import snowflake_tools


class _FakeDescriptionItem:
    def __init__(self, name: str):
        self.name = name


class _FakeCursor:
    def __init__(self):
        self.description = [_FakeDescriptionItem("ID"), _FakeDescriptionItem("NAME")]
        self._rows = [{"ID": 1, "NAME": "A"}, {"ID": 2, "NAME": "B"}]

    def execute(self, _sql: str):
        return None

    def fetchmany(self, max_rows: int):
        return self._rows[:max_rows]


@contextmanager
def _fake_cursor_ctx():
    yield _FakeCursor()


@pytest.fixture(autouse=True)
def _reset_allowed_tables(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_TABLES", None)


def test_is_read_only_allows_select():
    valid, error = snowflake_tools._is_read_only("SELECT * FROM CUSTOMERS LIMIT 5")
    assert valid is True
    assert error is None


def test_is_read_only_blocks_drop():
    valid, error = snowflake_tools._is_read_only("DROP TABLE CUSTOMERS")
    assert valid is False
    assert "Only read-only queries are allowed" in error


def test_is_read_only_blocks_multiple_statements():
    valid, error = snowflake_tools._is_read_only("SELECT 1; SELECT 2")
    assert valid is False
    assert "Multiple statements are not allowed" in error


def test_is_read_only_blocks_prompt_injection_multistatement():
    malicious = (
        "WITH seed AS (SELECT * FROM CUSTOMERS) "
        "SELECT * FROM seed LIMIT 1; DROP TABLE ORDERS"
    )
    valid, error = snowflake_tools._is_read_only(malicious)
    assert valid is False
    assert "Multiple statements are not allowed" in error


def test_is_read_only_blocks_non_allowlisted_table(monkeypatch):
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_TABLES",
        "CUSTOMERS,ORDERS",
    )
    valid, error = snowflake_tools._is_read_only("SELECT * FROM PAYMENTS LIMIT 1")
    assert valid is False
    assert "outside SNOWFLAKE_ALLOWED_TABLES" in error


def test_is_read_only_allows_allowlisted_table(monkeypatch):
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_TABLES",
        "CUSTOMERS,ORDERS",
    )
    valid, error = snowflake_tools._is_read_only("SELECT * FROM ORDERS LIMIT 1")
    assert valid is True
    assert error is None


def test_run_select_query_rejects_write():
    result = snowflake_tools.run_select_query.invoke({"sql": "DELETE FROM CUSTOMERS"})
    assert "error" in result
    assert "Only read-only queries are allowed" in result["error"]
    assert result["error_type"] == "validation_error"
    assert result["retryable"] is False


def test_run_select_query_success(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_MAX_ROWS", 10)
    monkeypatch.setattr(snowflake_tools, "_snowflake_cursor", _fake_cursor_ctx)

    result = snowflake_tools.run_select_query.invoke({"sql": "SELECT * FROM CUSTOMERS"})

    assert result["columns"] == ["ID", "NAME"]
    assert result["rows"] == [[1, "A"], [2, "B"]]
    assert result["row_count"] == 2
    assert result["truncated"] is False


def test_build_connect_kwargs_oauth_from_header_scope(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    auth = CustomAuthHeaders(auth_tokens={"Snowflake": "opaque-or-jwt-token"})
    with snowflake_tools.snowflake_request_auth_scope(auth, "analyst@example.com"):
        kwargs = snowflake_tools._build_connect_kwargs()
    assert kwargs["authenticator"] == "oauth"
    assert kwargs["token"] == "opaque-or-jwt-token"
    assert kwargs["user"] == "analyst@example.com"
    assert kwargs["account"] == "xy12345"
    assert "password" not in kwargs
    assert "private_key" not in kwargs


def test_build_connect_kwargs_oauth_requires_login(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    auth = CustomAuthHeaders(auth_tokens={"Snowflake": "token-only"})
    with pytest.raises(AppException) as excinfo:
        with snowflake_tools.snowflake_request_auth_scope(auth, None):
            snowflake_tools._build_connect_kwargs()
    assert "login" in str(excinfo.value).lower() or "user" in str(excinfo.value).lower()


def test_build_connect_kwargs_prefers_env_when_flag_set(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", "svc_user")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PASSWORD", "secret")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PRIVATE_KEY", None)
    monkeypatch.setattr(
        snowflake_tools.settings, "SNOWFLAKE_PREFER_ENV_CREDENTIALS", True
    )
    auth = CustomAuthHeaders(auth_tokens={"Snowflake": "bad-token"})
    with snowflake_tools.snowflake_request_auth_scope(auth, "analyst@example.com"):
        kwargs = snowflake_tools._build_connect_kwargs()
    assert kwargs.get("authenticator") != "oauth"
    assert kwargs["password"] == "secret"
    assert kwargs["user"] == "svc_user"


def test_connect_oauth_fallback_to_env(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", "svc_user")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PASSWORD", "secret")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PRIVATE_KEY", None)
    monkeypatch.setattr(
        snowflake_tools.settings, "SNOWFLAKE_PREFER_ENV_CREDENTIALS", False
    )
    monkeypatch.setattr(
        snowflake_tools.settings, "SNOWFLAKE_OAUTH_FALLBACK_TO_ENV", True
    )
    auth = CustomAuthHeaders(auth_tokens={"Snowflake": "bad-token"})
    calls: list[dict] = []

    def fake_connect(**kwargs):
        calls.append(kwargs)
        if kwargs.get("authenticator") == "oauth":
            raise snowflake_tools.snowflake.connector.Error(
                msg="Invalid OAuth access token"
            )
        return object()

    monkeypatch.setattr(snowflake_tools.snowflake.connector, "connect", fake_connect)
    with snowflake_tools.snowflake_request_auth_scope(auth, "analyst@example.com"):
        snowflake_tools._connect_snowflake()
    assert len(calls) == 2
    assert calls[0]["authenticator"] == "oauth"
    assert calls[1]["password"] == "secret"
    assert calls[1]["user"] == "svc_user"


def test_build_connect_kwargs_private_key_without_password(monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", "svc_user")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PASSWORD", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PRIVATE_KEY", pem)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", None)
    with snowflake_tools.snowflake_request_auth_scope(None, None):
        kwargs = snowflake_tools._build_connect_kwargs()
    assert "password" not in kwargs
    assert "private_key" in kwargs
    assert kwargs["user"] == "svc_user"


def test_build_connect_kwargs_env_password_when_no_request_token(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ACCOUNT", "xy12345")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER", "svc_user")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_USER_TEST", None)
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PASSWORD", "secret")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_PRIVATE_KEY", None)
    with snowflake_tools.snowflake_request_auth_scope(None, None):
        kwargs = snowflake_tools._build_connect_kwargs()
    assert kwargs.get("authenticator") != "oauth"
    assert kwargs["password"] == "secret"
    assert kwargs["user"] == "svc_user"


def test_tool_error_helper_shape():
    payload = snowflake_tools._tool_error(
        message="bad request",
        error_type="validation_error",
        retryable=False,
        details="details",
    )
    assert payload == {
        "error": "bad request",
        "error_type": "validation_error",
        "retryable": False,
        "details": "details",
    }


def test_qualify_accepts_database_dot_schema(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_SCHEMAS", None)
    assert (
        snowflake_tools._qualify("LEARNINGSOURCES_DB.INTERNAL_MARTS")
        == "LEARNINGSOURCES_DB.INTERNAL_MARTS"
    )


def test_qualify_schema_name_only_uses_database(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_SCHEMAS", None)
    assert snowflake_tools._qualify("INTERNAL_MARTS") == "LEARNINGSOURCES_DB.INTERNAL_MARTS"


def test_qualify_allowed_schemas_list(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_SCHEMAS",
        "LEARNINGSOURCES_DB.INTERNAL_MARTS,OTHER_DB.PUBLIC",
    )
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_DATABASES", None)
    assert snowflake_tools.settings.snowflake_allowed_schema_targets == [
        "LEARNINGSOURCES_DB.INTERNAL_MARTS",
        "OTHER_DB.PUBLIC",
    ]


def test_allowed_databases_with_shared_schema(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_SCHEMAS", None)
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_DATABASES",
        "LEARNINGSOURCES_DB,LEARNINGAGGREGATE_DB",
    )
    assert snowflake_tools.settings.snowflake_allowed_schema_targets == [
        "LEARNINGSOURCES_DB.INTERNAL_MARTS",
        "LEARNINGAGGREGATE_DB.INTERNAL_MARTS",
    ]
    assert snowflake_tools.settings.snowflake_allowed_databases == [
        "LEARNINGSOURCES_DB",
        "LEARNINGAGGREGATE_DB",
    ]


def test_qualify_ambiguous_schema_requires_database(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_SCHEMAS", None)
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_DATABASES",
        "DB_A,DB_B",
    )
    with pytest.raises(AppException) as exc:
        snowflake_tools._qualify("INTERNAL_MARTS")
    assert "multiple databases" in str(exc.value).lower()


def test_allowed_schema_singular_alias(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_SCHEMAS", None)
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_SCHEMA",
        "PUBLIC,INTERNAL_MARTS",
    )
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_ALLOWED_DATABASES", None)
    assert snowflake_tools.settings.snowflake_allowed_schema_targets == [
        "LEARNINGSOURCES_DB.PUBLIC",
        "LEARNINGSOURCES_DB.INTERNAL_MARTS",
    ]


def test_list_accessible_schemas_probes_snowflake(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_DATABASE", "LEARNINGSOURCES_DB")
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_SCHEMA", "INTERNAL_MARTS")
    monkeypatch.setattr(
        snowflake_tools.settings,
        "SNOWFLAKE_ALLOWED_SCHEMAS",
        "LEARNINGSOURCES_DB.INTERNAL_MARTS,LEARNINGAGGREGATE_DB.INTERNAL_MARTS",
    )

    def fake_fetch(target: str):
        if target == "LEARNINGSOURCES_DB.INTERNAL_MARTS":
            return ["T1"], None
        return [], "Object does not exist"

    monkeypatch.setattr(snowflake_tools, "_fetch_tables_in_schema", fake_fetch)
    result = snowflake_tools.list_accessible_schemas.invoke({})
    assert len(result["accessible"]) == 1
    assert result["accessible"][0]["target"] == "LEARNINGSOURCES_DB.INTERNAL_MARTS"
    assert result["accessible"][0]["table_count"] == 1
    assert len(result["inaccessible"]) == 1
    assert result["databases"] == ["LEARNINGSOURCES_DB"]
