"""Snowflake tools for the agent.

This module exposes a small, read-only toolkit that the agent uses to explore
schemas and run SELECT queries against a Snowflake account. All tools are
LangChain ``@tool`` callables and can be passed directly to ``create_react_agent``.

"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

import snowflake.connector
from langchain_core.tools import tool
from snowflake.connector import DictCursor
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from template_agent.src.core.exceptions.exceptions import (
    AppException,
    AppExceptionCode,
)
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


_READ_ONLY_PREFIXES = {"SELECT", "WITH", "SHOW", "DESC", "DESCRIBE"}
_DISALLOWED_AST_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
)


def _tool_error(
    message: str, error_type: str, retryable: bool, details: str | None = None
) -> dict[str, Any]:
    """Return normalized tool error payload."""
    payload: dict[str, Any] = {
        "error": message,
        "error_type": error_type,
        "retryable": retryable,
    }
    if details:
        payload["details"] = details
    return payload


def _build_connect_kwargs() -> dict[str, Any]:
    """Build kwargs for ``snowflake.connector.connect`` from settings.

    Supports either password or RSA key-pair authentication. Key-pair takes
    precedence if ``SNOWFLAKE_PRIVATE_KEY`` is set.
    """
    if not settings.SNOWFLAKE_ACCOUNT:
        raise AppException(
            "SNOWFLAKE_ACCOUNT is not configured",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    effective_user = settings.snowflake_user_effective
    if not effective_user:
        raise AppException(
            "Snowflake user is not configured. Set SNOWFLAKE_USER_TEST or SNOWFLAKE_USER",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    kwargs: dict[str, Any] = {
        "account": settings.SNOWFLAKE_ACCOUNT,
        "user": effective_user,
        "client_session_keep_alive": True,
        "network_timeout": settings.SNOWFLAKE_QUERY_TIMEOUT,
        "login_timeout": 30,
    }

    if settings.SNOWFLAKE_WAREHOUSE:
        kwargs["warehouse"] = settings.SNOWFLAKE_WAREHOUSE
    if settings.SNOWFLAKE_DATABASE:
        kwargs["database"] = settings.SNOWFLAKE_DATABASE
    if settings.SNOWFLAKE_SCHEMA:
        kwargs["schema"] = settings.SNOWFLAKE_SCHEMA
    if settings.SNOWFLAKE_ROLE:
        kwargs["role"] = settings.SNOWFLAKE_ROLE

    if settings.SNOWFLAKE_PRIVATE_KEY:
        # Key-pair auth requires a DER-encoded private key. Convert from PEM
        # at runtime so the secret can be stored as a regular PEM string.
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        passphrase = (
            settings.SNOWFLAKE_PRIVATE_KEY_PASSPHRASE.encode()
            if settings.SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
            else None
        )
        pkey = serialization.load_pem_private_key(
            settings.SNOWFLAKE_PRIVATE_KEY.encode(),
            password=passphrase,
            backend=default_backend(),
        )
        kwargs["private_key"] = pkey.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    elif settings.SNOWFLAKE_PASSWORD:
        kwargs["password"] = settings.SNOWFLAKE_PASSWORD
    else:
        raise AppException(
            "Either SNOWFLAKE_PRIVATE_KEY or SNOWFLAKE_PASSWORD must be set",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    return kwargs


@contextmanager
def _snowflake_cursor() -> Iterator[DictCursor]:
    """Yield a Snowflake DictCursor with statement timeout applied.

    The cursor is closed and the connection released on exit even on errors.
    """
    conn = snowflake.connector.connect(**_build_connect_kwargs())
    try:
        cur = conn.cursor(DictCursor)
        try:
            cur.execute(
                "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = %s",
                (settings.SNOWFLAKE_QUERY_TIMEOUT,),
            )
            yield cur
        finally:
            cur.close()
    finally:
        conn.close()


def _qualify(schema_name: str | None) -> str:
    """Return ``DATABASE.SCHEMA`` for SHOW/DESC statements.

    Falls back to ``SNOWFLAKE_SCHEMA`` when ``schema_name`` is not provided.
    Raises if neither database nor schema can be determined.
    """
    db = settings.SNOWFLAKE_DATABASE
    sc = schema_name or settings.SNOWFLAKE_SCHEMA
    if not db or not sc:
        raise AppException(
            "SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA must be set",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    return f"{db}.{sc}"


@tool
def list_tables(schema_name: str | None = None) -> dict[str, Any]:
    """List tables available in a Snowflake schema.

    Args:
        schema_name: Snowflake schema name. Defaults to the configured
            ``SNOWFLAKE_SCHEMA`` if omitted.

    Returns:
        Dict with the schema queried and a list of table names.
    """
    target = _qualify(schema_name)
    started_at = perf_counter()
    logger.info("snowflake.list_tables.start target=%s", target)
    try:
        with _snowflake_cursor() as cur:
            cur.execute(f"SHOW TABLES IN SCHEMA {target}")
            rows = cur.fetchall()
        names = [r.get("name") for r in rows if r.get("name")]
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "snowflake.list_tables.done target=%s row_count=%s duration_ms=%s status=ok",
            target,
            len(names),
            duration_ms,
        )
        return {"schema": target, "tables": names, "count": len(names)}
    except snowflake.connector.Error as exc:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        details = exc.msg or str(exc)
        logger.error(
            "snowflake.list_tables.done target=%s duration_ms=%s status=error details=%s",
            target,
            duration_ms,
            details,
        )
        return _tool_error(
            message=f"Snowflake error: {details}",
            error_type="snowflake_error",
            retryable=False,
            details=details,
        )


@tool
def describe_table(table_name: str, schema_name: str | None = None) -> dict[str, Any]:
    """Return the column definitions for a Snowflake table.

    Args:
        table_name: Table to describe.
        schema_name: Snowflake schema name. Defaults to ``SNOWFLAKE_SCHEMA``.

    Returns:
        Dict with the fully qualified table name and a list of columns
        ``{name, type, nullable}``.
    """
    target = _qualify(schema_name)
    fqn = f"{target}.{table_name}"
    started_at = perf_counter()
    logger.info("snowflake.describe_table.start fqn=%s", fqn)
    try:
        with _snowflake_cursor() as cur:
            cur.execute(f"DESC TABLE {fqn}")
            rows = cur.fetchall()
        columns = [
            {
                "name": r.get("name"),
                "type": r.get("type"),
                "nullable": r.get("null?") == "Y",
                "default": r.get("default"),
            }
            for r in rows
        ]
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "snowflake.describe_table.done fqn=%s row_count=%s duration_ms=%s status=ok",
            fqn,
            len(columns),
            duration_ms,
        )
        return {"table": fqn, "columns": columns, "column_count": len(columns)}
    except snowflake.connector.Error as exc:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        details = exc.msg or str(exc)
        logger.error(
            "snowflake.describe_table.done fqn=%s duration_ms=%s status=error details=%s",
            fqn,
            duration_ms,
            details,
        )
        return _tool_error(
            message=f"Snowflake error: {details}",
            error_type="snowflake_error",
            retryable=False,
            details=details,
        )


def _parse_allowed_tables() -> set[str]:
    """Parse comma-separated allowed tables from settings."""
    if not settings.SNOWFLAKE_ALLOWED_TABLES:
        return set()
    return {
        table.strip().upper()
        for table in settings.SNOWFLAKE_ALLOWED_TABLES.split(",")
        if table.strip()
    }


def _is_read_only(sql: str) -> tuple[bool, str | None]:
    """Validate query as read-only using AST checks when possible."""
    cleaned = sql.strip().rstrip(";").lstrip("(")
    if not cleaned:
        return False, "SQL query is empty."
    first = cleaned.split(None, 1)[0].upper()
    if first not in _READ_ONLY_PREFIXES:
        return (
            False,
            (
                "Only read-only queries are allowed (SELECT, WITH, SHOW, DESC, "
                "DESCRIBE). The submitted statement was rejected."
            ),
        )

    # SHOW/DESC are Snowflake commands and are not always fully represented by sqlglot.
    if first in {"SHOW", "DESC", "DESCRIBE"}:
        if ";" in cleaned:
            return False, "Multiple statements are not allowed."
        return True, None

    try:
        parsed = parse(cleaned, read="snowflake")
    except ParseError as exc:
        return False, f"Invalid SQL syntax: {exc}"

    if len(parsed) != 1:
        return False, "Multiple statements are not allowed."

    statement = parsed[0]
    if not isinstance(statement, exp.Select):
        return False, "Only SELECT-like statements are allowed."

    for node_type in _DISALLOWED_AST_NODES:
        if any(statement.find_all(node_type)):
            return False, f"Disallowed SQL operation detected: {node_type.__name__}"

    allowed_tables = _parse_allowed_tables()
    if allowed_tables:
        referenced_tables = {
            table.name.upper() for table in statement.find_all(exp.Table) if table.name
        }
        disallowed = sorted(referenced_tables - allowed_tables)
        if disallowed:
            return (
                False,
                (
                    "Query references tables outside SNOWFLAKE_ALLOWED_TABLES: "
                    f"{', '.join(disallowed)}"
                ),
            )

    return True, None


@tool
def run_select_query(sql: str) -> dict[str, Any]:
    """Execute a read-only SQL query against Snowflake and return rows.

    Only ``SELECT``, ``WITH``, ``SHOW``, ``DESC`` and ``DESCRIBE`` statements
    are allowed. Results are capped at ``SNOWFLAKE_MAX_ROWS``.

    Args:
        sql: Snowflake SQL query.

    Returns:
        Dict with ``columns``, ``rows`` (list of lists), ``row_count`` and a
        ``truncated`` flag. On rejection or failure, returns ``{"error": ...}``.
    """
    is_valid, error_message = _is_read_only(sql)
    if not is_valid:
        return _tool_error(
            message=error_message or "SQL validation failed.",
            error_type="validation_error",
            retryable=False,
        )

    cleaned = sql.strip().rstrip(";")
    started_at = perf_counter()
    sql_preview = cleaned[:500]
    logger.info("snowflake.run_select_query.start sql_preview=%s", sql_preview)
    try:
        with _snowflake_cursor() as cur:
            cur.execute(cleaned)
            rows = cur.fetchmany(settings.SNOWFLAKE_MAX_ROWS)
            columns = [d.name for d in cur.description] if cur.description else []
        # DictCursor returns dicts; normalise to rows-of-lists for stable JSON.
        normalised = [[r.get(c) for c in columns] for r in rows]
        truncated = len(normalised) == settings.SNOWFLAKE_MAX_ROWS
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "snowflake.run_select_query.done row_count=%s truncated=%s duration_ms=%s status=ok",
            len(normalised),
            truncated,
            duration_ms,
        )
        return {
            "columns": columns,
            "rows": normalised,
            "row_count": len(normalised),
            "truncated": truncated,
        }
    except snowflake.connector.Error as exc:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        details = exc.msg or str(exc)
        logger.error(
            "snowflake.run_select_query.done duration_ms=%s status=error details=%s",
            duration_ms,
            details,
        )
        return _tool_error(
            message=f"Snowflake error: {details}",
            error_type="snowflake_error",
            retryable=False,
            details=details,
        )


SNOWFLAKE_TOOLS = [list_tables, describe_table, run_select_query]
