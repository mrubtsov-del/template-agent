"""Tests for Snowflake tool validation and query execution helpers."""

from contextlib import contextmanager

import pytest

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


def test_run_select_query_success(monkeypatch):
    monkeypatch.setattr(snowflake_tools.settings, "SNOWFLAKE_MAX_ROWS", 10)
    monkeypatch.setattr(snowflake_tools, "_snowflake_cursor", _fake_cursor_ctx)

    result = snowflake_tools.run_select_query.invoke({"sql": "SELECT * FROM CUSTOMERS"})

    assert result["columns"] == ["ID", "NAME"]
    assert result["rows"] == [[1, "A"], [2, "B"]]
    assert result["row_count"] == 2
    assert result["truncated"] is False
