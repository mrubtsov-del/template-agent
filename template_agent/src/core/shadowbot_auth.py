"""Shadowbot JWT auth bootstrap from application settings.

Team / platform docs use OAUTH_ISSUER and JWKS_URL; this agent maps them to
AUTH_ISSUER and AUTH_JWKS_URL (see shadowbot-agent-api-integration skill).
"""

from __future__ import annotations

from shadowbot_agent_api import AuthConfig, configure_auth

from template_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from template_agent.src.settings import Settings, settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def normalize_issuer(url: str) -> str:
    """Strip trailing slash so Keycloak ``iss`` matches configured AUTH_ISSUER."""
    return url.strip().rstrip("/")


def validate_shadowbot_auth_settings(cfg: Settings) -> None:
    """Fail fast when AUTH_ENABLED but EmployeeIDP settings are incomplete."""
    if not cfg.AUTH_ENABLED:
        return
    missing = [
        name
        for name, value in (
            ("AUTH_ISSUER (OAUTH_ISSUER)", cfg.AUTH_ISSUER),
            ("AUTH_AUDIENCE", cfg.AUTH_AUDIENCE),
            ("AUTH_JWKS_URL (JWKS_URL)", cfg.AUTH_JWKS_URL),
        )
        if not value
    ]
    if missing:
        raise AppException(
            "AUTH_ENABLED=true but required auth settings are missing: "
            + ", ".join(missing)
            + ". Issuer and JWKS must both be prod or both be stage.",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )


def configure_shadowbot_auth_from_settings(cfg: Settings | None = None) -> bool:
    """Configure shadowbot-agent-api JWT validation from ``settings``.

    Returns:
        True if auth was configured and enabled, False otherwise.
    """
    cfg = cfg or settings

    if not cfg.AUTH_ENABLED:
        logger.warning(
            "Shadowbot auth DISABLED (AUTH_ENABLED=false). "
            "@require_auth V1/V2 endpoints will return 401."
        )
        return False

    validate_shadowbot_auth_settings(cfg)

    algorithms = [
        part.strip() for part in cfg.AUTH_ALGORITHMS.split(",") if part.strip()
    ]
    issuer = normalize_issuer(cfg.AUTH_ISSUER or "")
    auth_config = AuthConfig(
        enabled=True,
        issuer=issuer,
        audience=cfg.AUTH_AUDIENCE or "account",
        jwks_url=(cfg.AUTH_JWKS_URL or "").strip(),
        algorithms=algorithms or ["RS256"],
        verify_exp=cfg.AUTH_VERIFY_EXP,
        verify_aud=cfg.AUTH_VERIFY_AUD,
    )
    configure_auth(auth_config)
    logger.info(
        "Shadowbot auth ENABLED",
        issuer=auth_config.issuer,
        audience=auth_config.audience,
        jwks_url=auth_config.jwks_url,
    )
    return True
