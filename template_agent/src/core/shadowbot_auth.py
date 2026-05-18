"""Shadowbot JWT auth bootstrap from application settings."""

from shadowbot_agent_api import AuthConfig, configure_auth

from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


def configure_shadowbot_auth_from_settings() -> bool:
    """Configure shadowbot-agent-api JWT validation from ``settings``.

    Returns:
        True if auth was configured and enabled, False otherwise.
    """
    if not settings.AUTH_ENABLED:
        logger.warning(
            "Shadowbot auth DISABLED (AUTH_ENABLED=false). "
            "@require_auth V2/V1 endpoints will return 401."
        )
        return False

    if not settings.AUTH_ISSUER or not settings.AUTH_AUDIENCE or not settings.AUTH_JWKS_URL:
        logger.warning(
            "Shadowbot auth enabled but incomplete: set AUTH_ISSUER, AUTH_AUDIENCE, "
            "and AUTH_JWKS_URL (issuer and JWKS must match the same environment, prod or stage)."
        )
        return False

    algorithms = [
        part.strip() for part in settings.AUTH_ALGORITHMS.split(",") if part.strip()
    ]
    auth_config = AuthConfig(
        enabled=True,
        issuer=settings.AUTH_ISSUER,
        audience=settings.AUTH_AUDIENCE,
        jwks_url=settings.AUTH_JWKS_URL,
        algorithms=algorithms or ["RS256"],
        verify_exp=settings.AUTH_VERIFY_EXP,
        verify_aud=settings.AUTH_VERIFY_AUD,
    )
    configure_auth(auth_config)
    logger.info(
        "Shadowbot auth ENABLED",
        issuer=auth_config.issuer,
        audience=auth_config.audience,
        jwks_url=auth_config.jwks_url,
    )
    return True
