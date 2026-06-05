"""Settings configuration for the template agent.

This module provides centralized configuration management using Pydantic
BaseSettings for environment variable loading, validation, and default
value handling for the template agent service.
"""

from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.utils.pylogger import get_python_logger

# Initialize logger
logger = get_python_logger()

# Load environment variables with error handling
try:
    load_dotenv()
except Exception as e:
    # Log error but don't fail - environment variables might be set directly
    logger.warning(f"Could not load .env file: {e}")


class Settings(BaseSettings):
    """Configuration settings for the template agent.

    Uses Pydantic BaseSettings to load and validate configuration from
    environment variables. Provides default values for optional settings
    and validation for required ones.

    The settings are organized into logical groups:
    - Server Configuration: Host, port, SSL settings
    - Database Configuration: PostgreSQL connection parameters
    - Langfuse Configuration: Tracing and analytics settings
    - Google Configuration: Service account credentials
    - MCP Configuration: MCP server connection settings
    """

    # Server Configuration
    AGENT_HOST: str = Field(default="0.0.0.0", json_schema_extra={"env": "AGENT_HOST"})
    AGENT_PORT: int = Field(default=8081, json_schema_extra={"env": "AGENT_PORT"})
    AGENT_SSL_KEYFILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "AGENT_SSL_KEYFILE"}
    )
    AGENT_SSL_CERTFILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "AGENT_SSL_CERTFILE"}
    )
    PYTHON_LOG_LEVEL: str = Field(
        default="INFO", json_schema_extra={"env": "PYTHON_LOG_LEVEL"}
    )
    USE_INMEMORY_SAVER: bool = Field(
        default=False, json_schema_extra={"env": "USE_INMEMORY_SAVER"}
    )

    # Database Configuration
    POSTGRES_USER: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_USER"}
    )
    POSTGRES_PASSWORD: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_PASSWORD"}
    )
    POSTGRES_DB: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_DB"}
    )
    POSTGRES_HOST: str = Field(
        default="pgvector", json_schema_extra={"env": "POSTGRES_HOST"}
    )
    POSTGRES_PORT: int = Field(default=5432, json_schema_extra={"env": "POSTGRES_PORT"})

    # Google Service Account Configuration
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "GOOGLE_SERVICE_ACCOUNT_FILE"}
    )

    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_PUBLIC_KEY"}
    )
    LANGFUSE_SECRET_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_SECRET_KEY"}
    )
    LANGFUSE_BASE_URL: Optional[str] = Field(
        default=None, json_schema_extra={"env": "LANGFUSE_BASE_URL"}
    )
    LANGFUSE_TRACING_ENVIRONMENT: str = Field(
        default="development", json_schema_extra={"env": "LANGFUSE_TRACING_ENVIRONMENT"}
    )

    # Google Application Credentials
    GOOGLE_APPLICATION_CREDENTIALS_CONTENT: Optional[str] = Field(
        default=None,
        json_schema_extra={"env": "GOOGLE_APPLICATION_CREDENTIALS_CONTENT"},
    )

    # MCP Server Configuration
    MCP_ENABLED: bool = Field(
        default=False,
        json_schema_extra={
            "env": "MCP_ENABLED",
            "description": "Enable integration with MCP server for tool execution",
        },
    )
    MCP_SERVER_NAME: str = Field(
        default="template-mcp-server",
        json_schema_extra={"env": "MCP_SERVER_NAME"},
    )
    MCP_SERVER_URL: str = Field(
        default="http://localhost:5001/mcp/",
        json_schema_extra={"env": "MCP_SERVER_URL"},
    )
    MCP_TRANSPORT_PROTOCOL: str = Field(
        default="streamable_http",
        json_schema_extra={"env": "MCP_TRANSPORT_PROTOCOL"},
    )
    MCP_CONNECTION_TIMEOUT: int = Field(
        default=30,
        json_schema_extra={"env": "MCP_CONNECTION_TIMEOUT"},
    )
    MCP_SSL_VERIFY: bool = Field(
        default=False,
        json_schema_extra={
            "env": "MCP_SSL_VERIFY",
            "description": "Enable SSL certificate verification for MCP connections",
        },
    )
    MCP_API_KEY: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "MCP_API_KEY",
            "description": "Static API key for MCP server auth (e.g. Atlan). Used when sso_token is not available.",
        },
    )

    # Atlan MCP Server (data catalog — second MCP server)
    ATLAN_MCP_URL: Optional[str] = Field(
        default=None,
        json_schema_extra={"env": "ATLAN_MCP_URL"},
    )
    ATLAN_MCP_API_KEY: Optional[str] = Field(
        default=None,
        json_schema_extra={"env": "ATLAN_MCP_API_KEY"},
    )

    # Snowflake Configuration
    SNOWFLAKE_ACCOUNT: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_ACCOUNT"}
    )
    SNOWFLAKE_USER: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_USER"}
    )
    SNOWFLAKE_USER_TEST: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_USER_TEST"}
    )
    SNOWFLAKE_PASSWORD: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_PASSWORD"}
    )
    SNOWFLAKE_ROLE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_ROLE"}
    )
    SNOWFLAKE_PRIVATE_KEY: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_PRIVATE_KEY"}
    )
    SNOWFLAKE_PRIVATE_KEY_PASSPHRASE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"}
    )
    SNOWFLAKE_WAREHOUSE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_WAREHOUSE"}
    )
    SNOWFLAKE_DATABASE: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_DATABASE"}
    )
    SNOWFLAKE_SCHEMA: Optional[str] = Field(
        default=None, json_schema_extra={"env": "SNOWFLAKE_SCHEMA"}
    )
    SNOWFLAKE_ALLOWED_TABLES: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "SNOWFLAKE_ALLOWED_TABLES",
            "description": "Comma-separated list of allowed Snowflake tables for querying",
        },
    )
    SNOWFLAKE_ALLOWED_DATABASES: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "SNOWFLAKE_ALLOWED_DATABASES",
            "description": (
                "Comma-separated Snowflake databases. Combined with "
                "SNOWFLAKE_SCHEMA or schema-only SNOWFLAKE_ALLOWED_SCHEMAS entries."
            ),
        },
    )
    SNOWFLAKE_ALLOWED_SCHEMAS: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "SNOWFLAKE_ALLOWED_SCHEMAS",
            "description": (
                "Comma-separated schemas: SCHEMA names (paired with "
                "SNOWFLAKE_ALLOWED_DATABASES or SNOWFLAKE_DATABASE) or "
                "fully qualified DATABASE.SCHEMA"
            ),
        },
    )
    # Deprecated alias seen in some OpenShift secrets (singular key name).
    SNOWFLAKE_ALLOWED_SCHEMA: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "SNOWFLAKE_ALLOWED_SCHEMA",
            "description": "Deprecated: use SNOWFLAKE_ALLOWED_SCHEMAS instead.",
        },
    )
    SNOWFLAKE_QUERY_TIMEOUT: int = Field(
        default=60,
        json_schema_extra={
            "env": "SNOWFLAKE_QUERY_TIMEOUT",
            "description": "Query timeout in seconds for Snowflake connections",
        },
    )
    SNOWFLAKE_MAX_ROWS: int = Field(
        default=1000,
        json_schema_extra={
            "env": "SNOWFLAKE_MAX_ROWS",
            "description": "Maximum number of rows to fetch from Snowflake queries",
        },
    )
    SNOWFLAKE_PREFER_ENV_CREDENTIALS: bool = Field(
        default=False,
        json_schema_extra={
            "env": "SNOWFLAKE_PREFER_ENV_CREDENTIALS",
            "description": (
                "When true, ignore X-Authorization-Snowflake and use SNOWFLAKE_PRIVATE_KEY "
                "or SNOWFLAKE_PASSWORD from env/secret (preprod service account)."
            ),
        },
    )
    SNOWFLAKE_OAUTH_FALLBACK_TO_ENV: bool = Field(
        default=True,
        json_schema_extra={
            "env": "SNOWFLAKE_OAUTH_FALLBACK_TO_ENV",
            "description": (
                "If X-Authorization-Snowflake OAuth fails, retry with env credentials "
                "when SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY is configured."
            ),
        },
    )

    # Google Workspace tools (Sheets + Docs reader)
    GOOGLE_TOOLS_ENABLED: bool = Field(
        default=True,
        json_schema_extra={
            "env": "GOOGLE_TOOLS_ENABLED",
            "description": (
                "Enable Google Sheets/Docs reader tools. Requires "
                "GOOGLE_APPLICATION_CREDENTIALS_CONTENT to be configured."
            ),
        },
    )

    # Plotting (matplotlib + seaborn — data-viz-plots skill)
    PLOT_ENABLED: bool = Field(
        default=True,
        json_schema_extra={"env": "PLOT_ENABLED"},
    )
    PLOT_ARTIFACT_DIR: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "PLOT_ARTIFACT_DIR",
            "description": "Directory for PNG chart artifacts served at /api/v1/plots/",
        },
    )
    PLOT_MAX_ROWS: int = Field(
        default=5000,
        json_schema_extra={
            "env": "PLOT_MAX_ROWS",
            "description": "Maximum rows from query results used for a single chart",
        },
    )
    AGENT_PUBLIC_BASE_URL: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "AGENT_PUBLIC_BASE_URL",
            "description": (
                "Public base URL for plot links in Shadowbot responses "
                "(e.g. https://snowflake-bot-....openshiftapps.com)"
            ),
        },
    )

    @property
    def snowflake_user_effective(self) -> Optional[str]:
        """Return Snowflake username with backward-compatible fallback order."""
        return self.SNOWFLAKE_USER_TEST or self.SNOWFLAKE_USER

    @staticmethod
    def _csv_env_list(value: Optional[str]) -> list[str]:
        if not value:
            return []
        return [part.strip() for part in value.split(",") if part.strip()]

    @property
    def snowflake_allowed_schema_targets(self) -> list[str]:
        """Normalized ``DATABASE.SCHEMA`` targets from env lists.

        Resolution order:
        1. ``DATABASE.SCHEMA`` entries in ``SNOWFLAKE_ALLOWED_SCHEMAS``.
        2. Schema-only entries × each DB in ``SNOWFLAKE_ALLOWED_DATABASES``
           (or ``SNOWFLAKE_DATABASE``).
        3. Each DB in ``SNOWFLAKE_ALLOWED_DATABASES`` × ``SNOWFLAKE_SCHEMA``.
        4. Fallback ``SNOWFLAKE_DATABASE`` + ``SNOWFLAKE_SCHEMA``.
        """
        schemas_csv = self.SNOWFLAKE_ALLOWED_SCHEMAS or self.SNOWFLAKE_ALLOWED_SCHEMA
        schemas_raw = self._csv_env_list(schemas_csv)
        databases_raw = self._csv_env_list(self.SNOWFLAKE_ALLOWED_DATABASES)

        explicit = [entry for entry in schemas_raw if "." in entry]
        schema_only = [entry for entry in schemas_raw if "." not in entry]

        targets: list[str] = list(explicit)

        if schema_only:
            databases = databases_raw or (
                [self.SNOWFLAKE_DATABASE] if self.SNOWFLAKE_DATABASE else []
            )
            for database in databases:
                for schema in schema_only:
                    targets.append(f"{database}.{schema}")
        elif databases_raw and self.SNOWFLAKE_SCHEMA:
            for database in databases_raw:
                targets.append(f"{database}.{self.SNOWFLAKE_SCHEMA}")

        if not targets and self.SNOWFLAKE_DATABASE and self.SNOWFLAKE_SCHEMA:
            targets.append(f"{self.SNOWFLAKE_DATABASE}.{self.SNOWFLAKE_SCHEMA}")

        seen: set[str] = set()
        ordered: list[str] = []
        for target in targets:
            key = target.upper()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(target)
        return ordered

    @property
    def snowflake_allowed_databases(self) -> list[str]:
        """Unique database names from allowed schema targets and env lists."""
        from_databases = self._csv_env_list(self.SNOWFLAKE_ALLOWED_DATABASES)
        from_targets = [t.split(".", 1)[0] for t in self.snowflake_allowed_schema_targets if "." in t]
        seen: set[str] = set()
        ordered: list[str] = []
        for name in from_databases + from_targets + ([self.SNOWFLAKE_DATABASE] if self.SNOWFLAKE_DATABASE else []):
            if not name:
                continue
            key = name.upper()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(name)
        return ordered

    @property
    def snowflake_default_schema_target(self) -> Optional[str]:
        """First configured ``DATABASE.SCHEMA`` (session default)."""
        targets = self.snowflake_allowed_schema_targets
        return targets[0] if targets else None

    # Shadowbot JWT — team doc OAUTH_ISSUER / JWKS_URL map to AUTH_ISSUER / AUTH_JWKS_URL
    AUTH_ENABLED: bool = Field(
        default=False,
        json_schema_extra={
            "env": "AUTH_ENABLED",
            "description": "Enable JWT validation for @require_auth Shadowbot endpoints",
        },
    )
    AUTH_ISSUER: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "AUTH_ISSUER",
            "description": "JWT issuer (team doc: OAUTH_ISSUER). Prod: auth.redhat.com/.../EmployeeIDP",
        },
    )
    AUTH_AUDIENCE: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "AUTH_AUDIENCE",
            "description": "JWT audience (team doc: AUTH_AUDIENCE=account)",
        },
    )
    AUTH_JWKS_URL: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "env": "AUTH_JWKS_URL",
            "description": "JWKS certs URL (team doc: JWKS_URL). Must match AUTH_ISSUER env",
        },
    )
    AUTH_ALGORITHMS: str = Field(
        default="RS256",
        json_schema_extra={"env": "AUTH_ALGORITHMS"},
    )
    AUTH_VERIFY_EXP: bool = Field(
        default=True,
        json_schema_extra={"env": "AUTH_VERIFY_EXP"},
    )
    AUTH_VERIFY_AUD: bool = Field(
        default=True,
        json_schema_extra={"env": "AUTH_VERIFY_AUD"},
    )

    # Request Logging Configuration
    REQUEST_LOGGING_ENABLED: bool = Field(
        default=True,
        json_schema_extra={
            "env": "REQUEST_LOGGING_ENABLED",
            "description": "Enable request/response logging",
        },
    )
    REQUEST_LOG_HEADERS: bool = Field(
        default=True,
        json_schema_extra={
            "env": "REQUEST_LOG_HEADERS",
            "description": "Include headers in request/response logs",
        },
    )
    REQUEST_LOG_BODY: bool = Field(
        default=False,
        json_schema_extra={
            "env": "REQUEST_LOG_BODY",
            "description": "Include body content in request/response logs",
        },
    )
    REQUEST_LOG_BODY_MAX_SIZE: int = Field(
        default=10240,
        json_schema_extra={
            "env": "REQUEST_LOG_BODY_MAX_SIZE",
            "description": "Maximum body size in bytes to log (0 for unlimited)",
        },
    )

    @property
    def database_uri(self) -> str:
        """Generate database URI from individual components.

        Constructs a PostgreSQL connection URI using the configured
        database settings including user, password, host, port, and
        database name.

        Returns:
            The complete PostgreSQL database URI string.
        """
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


def validate_config(settings: Settings) -> None:
    """Validate configuration settings.

    Performs comprehensive validation to ensure required settings are
    present and values are within acceptable ranges. This function
    validates port ranges, log levels, and transport protocols.

    Args:
        settings: Settings instance to validate.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    # Validate port range
    if not (1024 <= settings.AGENT_PORT <= 65535):
        logger.error(
            f"AGENT_PORT must be between 1024 and 65535, got {settings.AGENT_PORT}"
        )
        raise AppException(
            f"AGENT_PORT must be between 1024 and 65535, got {settings.AGENT_PORT}",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.PYTHON_LOG_LEVEL.upper() not in valid_log_levels:
        logger.error(
            f"PYTHON_LOG_LEVEL must be one of {valid_log_levels}, got {settings.PYTHON_LOG_LEVEL}"
        )
        raise AppException(
            f"PYTHON_LOG_LEVEL must be one of {valid_log_levels}, got {settings.PYTHON_LOG_LEVEL}",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    from template_agent.src.core.shadowbot_auth import validate_shadowbot_auth_settings

    validate_shadowbot_auth_settings(settings)


# Create settings instance without validation (validation happens in main.py)
settings = Settings()
