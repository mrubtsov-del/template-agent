"""Snowflake tools for the agent.

This module exposes a small, read-only toolkit that the agent uses to explore
schemas and run SELECT queries against a Snowflake account. All tools are
LangChain ``@tool`` callables and can be passed directly to ``create_react_agent``.

"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import snowflake.connector
from langchain_core.tools import tool
from snowflake.connector import DictCursor

from template_agent.src.core.exceptions.exceptions import (
    AppException,
    AppExceptionCode,
)
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


_READ_ONLY_PREFIXES = {"SELECT", "WITH", "SHOW", "DESC", "DESCRIBE"}


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
    logger.info("snowflake.list_tables target=%s", target)
    try:
        with _snowflake_cursor() as cur:
            cur.execute(f"SHOW TABLES IN SCHEMA {target}")
            rows = cur.fetchall()
        names = [r.get("name") for r in rows if r.get("name")]
        return {"schema": target, "tables": names, "count": len(names)}
    except snowflake.connector.Error as exc:
        logger.error("snowflake.list_tables failed: %s", exc)
        return {"error": f"Snowflake error: {exc.msg or str(exc)}"}


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
    logger.info("snowflake.describe_table fqn=%s", fqn)
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
        return {"table": fqn, "columns": columns, "column_count": len(columns)}
    except snowflake.connector.Error as exc:
        logger.error("snowflake.describe_table failed: %s", exc)
        return {"error": f"Snowflake error: {exc.msg or str(exc)}"}


def _is_read_only(sql: str) -> bool:
    """Coarse read-only check used in MVP.

    A stricter AST-based check using ``sqlglot`` is introduced on Day 4.
    """
    cleaned = sql.strip().rstrip(";").lstrip("(")
    if not cleaned:
        return False
    first = cleaned.split(None, 1)[0].upper()
    return first in _READ_ONLY_PREFIXES


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
    if not _is_read_only(sql):
        return {
            "error": (
                "Only read-only queries are allowed (SELECT, WITH, SHOW, "
                "DESC, DESCRIBE). The submitted statement was rejected."
            )
        }

    cleaned = sql.strip().rstrip(";")
    logger.info("snowflake.run_select_query sql=%s", cleaned[:500])
    try:
        with _snowflake_cursor() as cur:
            cur.execute(cleaned)
            rows = cur.fetchmany(settings.SNOWFLAKE_MAX_ROWS)
            columns = [d.name for d in cur.description] if cur.description else []
        # DictCursor returns dicts; normalise to rows-of-lists for stable JSON.
        normalised = [[r.get(c) for c in columns] for r in rows]
        truncated = len(normalised) == settings.SNOWFLAKE_MAX_ROWS
        return {
            "columns": columns,
            "rows": normalised,
            "row_count": len(normalised),
            "truncated": truncated,
        }
    except snowflake.connector.Error as exc:
        logger.error("snowflake.run_select_query failed: %s", exc)
        return {"error": f"Snowflake error: {exc.msg or str(exc)}"}


SNOWFLAKE_TOOLS = [list_tables, describe_table, run_select_query]
