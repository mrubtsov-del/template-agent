"""Snowflake tools for the agent.

This module exposes a small, read-only toolkit that the agent uses to explore
schemas and run SELECT queries against a Snowflake account. All tools are
LangChain ``@tool`` callables and can be passed directly to ``create_react_agent``.

"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Optional

import snowflake.connector
from langchain_core.tools import tool
from snowflake.connector import DictCursor
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from shadowbot_agent_api.models import CustomAuthHeaders

from template_agent.src.core.exceptions.exceptions import (
    AppException,
    AppExceptionCode,
)
from template_agent.src.routes.common import resolve_snowflake_request_token
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

# Per-request Snowflake credentials from Shadowbot ``X-Authorization-Snowflake``
# (wired by ``AgentManager`` around each agent run).
_snowflake_request_ctx: contextvars.ContextVar[
    tuple[Optional[CustomAuthHeaders], Optional[str]]
] = contextvars.ContextVar(
    "snowflake_request_auth",
    default=(None, None),
)


@contextmanager
def snowflake_request_auth_scope(
    custom_auth: Optional[CustomAuthHeaders],
    snowflake_login: Optional[str] = None,
) -> Iterator[None]:
    """Bind ``CustomAuthHeaders`` and optional Snowflake login for this request.

    When ``X-Authorization-Snowflake`` is present, tools connect with OAuth
    (``authenticator='oauth'``, ``token=...``). Otherwise env key/password
    auth is used as before.
    """
    reset_token = _snowflake_request_ctx.set((custom_auth, snowflake_login))
    try:
        yield
    finally:
        _snowflake_request_ctx.reset(reset_token)


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


def _env_credentials_configured() -> bool:
    return bool(settings.SNOWFLAKE_PRIVATE_KEY or settings.SNOWFLAKE_PASSWORD)


def _base_connect_kwargs() -> dict[str, Any]:
    if not settings.SNOWFLAKE_ACCOUNT:
        raise AppException(
            "SNOWFLAKE_ACCOUNT is not configured",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    kwargs: dict[str, Any] = {
        "account": settings.SNOWFLAKE_ACCOUNT,
        "client_session_keep_alive": True,
        "network_timeout": settings.SNOWFLAKE_QUERY_TIMEOUT,
        "login_timeout": 30,
    }
    if settings.SNOWFLAKE_WAREHOUSE:
        kwargs["warehouse"] = settings.SNOWFLAKE_WAREHOUSE
    default_target = settings.snowflake_default_schema_target
    if default_target and "." in default_target:
        db, _, schema = default_target.partition(".")
        kwargs["database"] = db
        kwargs["schema"] = schema
    else:
        if settings.SNOWFLAKE_DATABASE:
            kwargs["database"] = settings.SNOWFLAKE_DATABASE
        if settings.SNOWFLAKE_SCHEMA:
            kwargs["schema"] = settings.SNOWFLAKE_SCHEMA
    if settings.SNOWFLAKE_ROLE:
        kwargs["role"] = settings.SNOWFLAKE_ROLE
    return kwargs


def _resolve_effective_user(req_login: Optional[str]) -> str:
    effective_user = (req_login and req_login.strip()) or settings.snowflake_user_effective
    if not effective_user:
        raise AppException(
            "Snowflake user is not configured. Set SNOWFLAKE_USER_TEST or SNOWFLAKE_USER",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    return effective_user


def _build_oauth_connect_kwargs(
    oauth_token: str, *, req_login: Optional[str]
) -> dict[str, Any]:
    """OAuth via Shadowbot ``X-Authorization-Snowflake`` (per-user)."""
    kwargs = _base_connect_kwargs()
    kwargs["user"] = _resolve_effective_user(req_login)
    kwargs["authenticator"] = "oauth"
    kwargs["token"] = oauth_token
    logger.info(
        "snowflake.connect using OAuth token from request (user=%s)",
        kwargs["user"],
    )
    return kwargs


def _build_env_connect_kwargs(*, req_login: Optional[str] = None) -> dict[str, Any]:
    """Password or key-pair from env/secret (service account)."""
    kwargs = _base_connect_kwargs()
    kwargs["user"] = _resolve_effective_user(req_login)

    if settings.SNOWFLAKE_PRIVATE_KEY:
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

    logger.info(
        "snowflake.connect using env credentials (user=%s)",
        kwargs["user"],
    )
    return kwargs


def _should_use_request_oauth(oauth_token: Optional[str]) -> bool:
    if not oauth_token:
        return False
    if settings.SNOWFLAKE_PREFER_ENV_CREDENTIALS and _env_credentials_configured():
        logger.info(
            "snowflake.connect ignoring X-Authorization-Snowflake "
            "(SNOWFLAKE_PREFER_ENV_CREDENTIALS=true)"
        )
        return False
    return True


def _is_invalid_oauth_error(exc: snowflake.connector.Error) -> bool:
    text = f"{getattr(exc, 'msg', '')} {exc}".lower()
    return "oauth" in text and "token" in text


def _build_connect_kwargs() -> dict[str, Any]:
    """Build kwargs for ``snowflake.connector.connect``.

    Priority (shadowbot-agent-api skill):
    1. ``X-Authorization-Snowflake`` OAuth when present (unless preprod override).
    2. ``SNOWFLAKE_PRIVATE_KEY`` (preferred) or ``SNOWFLAKE_PASSWORD`` from env/secret.
    """
    req_custom_auth, req_login = _snowflake_request_ctx.get((None, None))
    oauth_token = resolve_snowflake_request_token(req_custom_auth)

    if _should_use_request_oauth(oauth_token):
        return _build_oauth_connect_kwargs(oauth_token, req_login=req_login)

    # Service account from secret: do not substitute Keycloak email as Snowflake user.
    env_login = (
        None
        if settings.SNOWFLAKE_PREFER_ENV_CREDENTIALS and _env_credentials_configured()
        else req_login
    )
    return _build_env_connect_kwargs(req_login=env_login)


def _connect_snowflake() -> snowflake.connector.SnowflakeConnection:
    """Connect, optionally falling back to env creds when platform OAuth fails."""
    kwargs = _build_connect_kwargs()
    used_oauth = kwargs.get("authenticator") == "oauth"
    try:
        return snowflake.connector.connect(**kwargs)
    except snowflake.connector.Error as exc:
        if (
            used_oauth
            and settings.SNOWFLAKE_OAUTH_FALLBACK_TO_ENV
            and _env_credentials_configured()
            and _is_invalid_oauth_error(exc)
        ):
            logger.warning(
                "Snowflake OAuth from X-Authorization-Snowflake failed; "
                "retrying with env credentials: %s",
                getattr(exc, "msg", exc),
            )
            req_custom_auth, req_login = _snowflake_request_ctx.get((None, None))
            # Service account: use SNOWFLAKE_USER_TEST, not Keycloak email.
            _ = req_custom_auth
            return snowflake.connector.connect(
                **_build_env_connect_kwargs(req_login=None)
            )
        raise


@contextmanager
def _snowflake_cursor() -> Iterator[DictCursor]:
    """Yield a Snowflake DictCursor with statement timeout applied.

    The cursor is closed and the connection released on exit even on errors.
    """
    conn = _connect_snowflake()
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


def _allowed_targets_label() -> str:
    allowed = settings.snowflake_allowed_schema_targets
    return ", ".join(allowed) if allowed else "(not configured)"


def _split_schema_target(target: str) -> tuple[str, str]:
    """Split ``DATABASE.SCHEMA``; raise if format is invalid."""
    parts = target.split(".")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise AppException(
            f"Invalid schema target '{target}'. Expected DATABASE.SCHEMA",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    return parts[0].strip(), parts[1].strip()


def _fetch_tables_in_schema(target: str) -> tuple[list[str], Optional[str]]:
    """Return table names in schema, or ``( [], error_message )`` on failure."""
    try:
        with _snowflake_cursor() as cur:
            cur.execute(f"SHOW TABLES IN SCHEMA {target}")
            rows = cur.fetchall()
        names = [r.get("name") for r in rows if r.get("name")]
        return names, None
    except snowflake.connector.Error as exc:
        return [], exc.msg or str(exc)


def _probe_schema_target(target: str) -> dict[str, Any]:
    """Check Snowflake access to a schema via ``SHOW TABLES IN SCHEMA``."""
    started_at = perf_counter()
    logger.info("snowflake.probe_schema.start target=%s", target)
    try:
        database, schema = _split_schema_target(target)
    except AppException as exc:
        return {
            "target": target,
            "accessible": False,
            "error": str(exc),
        }

    names, error = _fetch_tables_in_schema(target)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    if error:
        logger.info(
            "snowflake.probe_schema.done target=%s accessible=false duration_ms=%s",
            target,
            duration_ms,
        )
        return {
            "target": target,
            "accessible": False,
            "error": error,
        }

    logger.info(
        "snowflake.probe_schema.done target=%s accessible=true table_count=%s duration_ms=%s",
        target,
        len(names),
        duration_ms,
    )
    return {
        "target": target,
        "database": database,
        "schema": schema,
        "accessible": True,
        "table_count": len(names),
    }


def _qualify(
    schema_name: str | None = None,
    database_name: str | None = None,
) -> str:
    """Return ``DATABASE.SCHEMA`` for SHOW/DESC statements.

    Accepts:
    - ``LEARNINGSOURCES_DB.INTERNAL_MARTS`` in ``schema_name`` → used as-is.
    - ``database_name`` + ``schema_name`` → ``DB.SCHEMA``.
    - ``schema_name`` only → resolved against allowed targets / defaults.
    - neither → first allowed target or ``SNOWFLAKE_DATABASE`` + ``SNOWFLAKE_SCHEMA``.

    Raises if neither database nor schema can be determined.
    """
    allowed = settings.snowflake_allowed_schema_targets
    allowed_set = {a.upper() for a in allowed}

    if schema_name and "." in schema_name.strip():
        target = schema_name.strip()
        if allowed_set and target.upper() not in allowed_set:
            raise AppException(
                f"Schema '{target}' is not allowed. Configured: {_allowed_targets_label()}",
                AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
            )
        return target

    db = (database_name or "").strip() or None
    sc = (schema_name or "").strip() or settings.SNOWFLAKE_SCHEMA

    if db and sc:
        target = f"{db}.{sc}"
        if allowed_set and target.upper() not in allowed_set:
            raise AppException(
                f"Schema '{target}' is not allowed. Configured: {_allowed_targets_label()}",
                AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
            )
        return target

    if sc and not db:
        matches = [
            target
            for target in allowed
            if target.split(".", 1)[-1].upper() == sc.upper()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AppException(
                f"Schema name '{sc}' exists in multiple databases: {', '.join(matches)}. "
                "Pass database_name or use DATABASE.SCHEMA.",
                AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
            )

    if not sc:
        if allowed:
            return allowed[0]
        raise AppException(
            "SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA must be set",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    db = db or settings.SNOWFLAKE_DATABASE
    if not db:
        if allowed and len(allowed) == 1:
            return allowed[0]
        raise AppException(
            "database_name or SNOWFLAKE_DATABASE is required when multiple databases are configured",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    target = f"{db}.{sc}"
    if allowed_set and target.upper() not in allowed_set:
        raise AppException(
            f"Schema '{target}' is not allowed. Configured: {_allowed_targets_label()}",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    return target


@tool
def list_accessible_schemas() -> dict[str, Any]:
    """List databases/schemas the agent can actually access in Snowflake.

    Probes each entry from ``SNOWFLAKE_ALLOWED_SCHEMAS`` / env config with
    ``SHOW TABLES IN SCHEMA``. Use this before telling the user which schemas exist.

    Returns:
        ``configured`` (env), ``accessible`` (verified), ``inaccessible`` (failed probes),
        ``databases`` (from accessible only), and ``default``.
    """
    configured = settings.snowflake_allowed_schema_targets
    accessible: list[dict[str, Any]] = []
    inaccessible: list[dict[str, Any]] = []

    if not configured:
        return {
            "configured": [],
            "accessible": [],
            "inaccessible": [],
            "databases": [],
            "default": None,
            "configuration_error": (
                "No Snowflake schemas configured. Set SNOWFLAKE_DATABASE and "
                "SNOWFLAKE_SCHEMA in the Secret, or SNOWFLAKE_ALLOWED_SCHEMAS "
                "(not SNOWFLAKE_ALLOWED_SCHEMA) with DATABASE.SCHEMA entries."
            ),
            "hint": "Fix OpenShift Secret/ConfigMap and restart the Knative revision.",
        }

    for target in configured:
        if "." not in target:
            inaccessible.append(
                {
                    "target": target,
                    "error": (
                        "Invalid configuration format. Use DATABASE.SCHEMA "
                        "(comma-separated), e.g. LEARNINGSOURCES_DB.INTERNAL_MARTS"
                    ),
                }
            )
            continue
        probe = _probe_schema_target(target)
        if probe.get("accessible"):
            accessible.append(
                {
                    "target": probe["target"],
                    "database": probe["database"],
                    "schema": probe["schema"],
                    "table_count": probe.get("table_count", 0),
                }
            )
        else:
            inaccessible.append(
                {
                    "target": probe["target"],
                    "error": probe.get("error", "Access denied or schema does not exist"),
                }
            )

    databases = sorted({entry["database"] for entry in accessible})
    return {
        "configured": configured,
        "accessible": accessible,
        "inaccessible": inaccessible,
        "databases": databases,
        "default": accessible[0]["target"] if accessible else None,
        "hint": (
            "Report ONLY entries in 'accessible' to the user. "
            "'configured' is env intent, not proof of Snowflake privileges. "
            "Call list_tables(schema_name='DATABASE.SCHEMA') for one schema's tables."
        ),
    }


@tool
def list_tables(
    schema_name: str | None = None,
    database_name: str | None = None,
) -> dict[str, Any]:
    """List tables available in a Snowflake schema.

    Args:
        schema_name: Schema name, or fully qualified ``DATABASE.SCHEMA``.
        database_name: Database when schema_name is not qualified (optional if only one DB).

    Returns:
        Dict with the schema queried and a list of table names.
    """
    target = _qualify(schema_name, database_name)
    started_at = perf_counter()
    logger.info("snowflake.list_tables.start target=%s", target)
    names, error = _fetch_tables_in_schema(target)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    if error:
        logger.error(
            "snowflake.list_tables.done target=%s duration_ms=%s status=error details=%s",
            target,
            duration_ms,
            error,
        )
        return _tool_error(
            message=f"Cannot access schema '{target}': {error}",
            error_type="snowflake_error",
            retryable=False,
            details=error,
        )
    logger.info(
        "snowflake.list_tables.done target=%s row_count=%s duration_ms=%s status=ok",
        target,
        len(names),
        duration_ms,
    )
    return {"schema": target, "tables": names, "count": len(names)}


@tool
def describe_table(
    table_name: str,
    schema_name: str | None = None,
    database_name: str | None = None,
) -> dict[str, Any]:
    """Return the column definitions for a Snowflake table.

    Args:
        table_name: Table to describe.
        schema_name: Schema name, or fully qualified ``DATABASE.SCHEMA``.
        database_name: Database when schema_name is not qualified.

    Returns:
        Dict with the fully qualified table name and a list of columns
        ``{name, type, nullable}``.
    """
    target = _qualify(schema_name, database_name)
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


SNOWFLAKE_TOOLS = [
    list_accessible_schemas,
    list_tables,
    describe_table,
    run_select_query,
]
