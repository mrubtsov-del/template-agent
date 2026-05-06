import asyncio
import json
from typing import Optional, Dict, Any, Callable

import httpx
import jwt
from functools import wraps
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta, timezone

from .constants import Constants
from .models import AuthConfig, UserContext, CustomAuthHeaders
from shadowbot_agent_api.logger import get_python_logger

logger = get_python_logger(Constants.PYTHON_LOG_LEVEL)

# Global auth configuration
_auth_config: Optional[AuthConfig] = None
_jwks_cache: Dict[str, Dict] = {}
_jwks_cache_expiry: Dict[str, datetime] = {}

security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Custom exception for authentication errors."""
    pass


def configure_auth(config: AuthConfig) -> None:
    """Configure global authentication settings."""
    global _auth_config
    _auth_config = config
    logger.info(f"Authentication configured for issuer: {config.issuer}")


def get_auth_config() -> Optional[AuthConfig]:
    """Get the current auth configuration."""
    return _auth_config


async def get_jwks(jwks_url: str, max_retries: int = 3) -> Dict[str, Any]:
    """Fetch JWKS (JSON Web Key Set) from the provided URL with caching.

    Args:
        jwks_url: The JWKS endpoint URL
        max_retries: Maximum number of retry attempts on failure (default: 3)

    Returns:
        The JWKS response as a dictionary

    Raises:
        AuthenticationError: If fetching fails after all retries
    """
    global _jwks_cache, _jwks_cache_expiry

    # Check cache first
    now = datetime.now(timezone.utc)
    if jwks_url in _jwks_cache and jwks_url in _jwks_cache_expiry:
        if now < _jwks_cache_expiry[jwks_url]:
            return _jwks_cache[jwks_url]

    last_error = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_url, timeout=10.0)
                response.raise_for_status()
                jwks = response.json()

                # Cache for 24 hours
                _jwks_cache[jwks_url] = jwks
                _jwks_cache_expiry[jwks_url] = now + timedelta(hours=24)

                if attempt > 0:
                    logger.info(f"Successfully fetched JWKS on attempt {attempt + 1}")

                return jwks
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Failed to fetch JWKS (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to fetch JWKS after {max_retries} attempts: {e}"
                )

    raise AuthenticationError(f"Failed to fetch JWKS after {max_retries} attempts: {last_error}")


def get_key_from_jwks(token_header: Dict[str, Any], jwks: Dict[str, Any]) -> Optional[str]:
    """Extract the public key from JWKS based on the token's kid."""
    kid = token_header.get("kid")
    if not kid:
        return None
        
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
    
    return None


async def verify_token(token: str, config: AuthConfig) -> UserContext:
    """Verify JWT token and return user context."""
    try:
        # Decode header without verification to get kid
        unverified_header = jwt.get_unverified_header(token)
        
        # Get JWKS
        if not config.jwks_url:
            raise AuthenticationError("JWKS URL not configured")
        jwks = await get_jwks(config.jwks_url)
        
        # Get the key for verification
        key = get_key_from_jwks(unverified_header, jwks)
        if not key:
            raise AuthenticationError("Unable to find appropriate key in JWKS")
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            key,
            algorithms=config.algorithms,
            issuer=config.issuer if config.verify_aud else None,
            audience=config.audience if config.verify_aud else None,
            options={
                "verify_exp": config.verify_exp,
                "verify_aud": config.verify_aud,
                "verify_iss": True
            }
        )
        
        # Extract user information
        user_context = UserContext(
            sub=payload.get("sub", ""),
            email=payload.get("email"),
            name=payload.get("name"),
            preferred_username=payload.get("preferred_username"),
            groups=payload.get("groups", []),
            raw_token=payload
        )
        
        return user_context
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise AuthenticationError(f"Token verification failed: {e}")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserContext]:
    """FastAPI dependency to get the current user from JWT token."""
    config = get_auth_config()
    
    # If auth is not configured or disabled, return None
    if not config or not config.enabled:
        return None
    
    # If no credentials provided and auth is required, raise error
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user_context = await verify_token(credentials.credentials, config)
        return user_context
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserContext]:
    """FastAPI dependency to optionally get the current user from JWT token."""
    config = get_auth_config()

    # If auth is not configured or disabled, return None
    if not config or not config.enabled:
        return None

    # If no credentials provided, return None (optional auth)
    if not credentials:
        return None

    try:
        user_context = await verify_token(credentials.credentials, config)
        return user_context
    except AuthenticationError:
        # For optional auth, we don't raise errors, just return None
        return None


async def get_custom_auth_headers(request: Request) -> CustomAuthHeaders:
    """
    FastAPI dependency to extract authentication headers from request.

    Extracts:
    1. Authorization bearer token from "Authorization: Bearer <token>" header
    2. All X-Authorization-* headers for third-party services

    Examples:
        Authorization: Bearer eyJhbGci...
        X-Authorization-Snowflake: sk_123
        X-Authorization-PeopleAI: pk_456

        Returns:
            CustomAuthHeaders(
                bearer_token="eyJhbGci...",
                auth_tokens={"Snowflake": "sk_123", "PeopleAI": "pk_456"}
            )

    Usage in handler:
        @chat_handler_v2()
        async def my_handler(
            request: ConversationRequestV2,
            custom_auth: CustomAuthHeaders = Depends(get_custom_auth_headers)
        ):
            # Access Authorization bearer token
            jwt_token = custom_auth.bearer_token

            # Access service-specific tokens
            if custom_auth.has("PeopleAI"):
                token = custom_auth.get("PeopleAI")
                data = await query_peopleai(token)
    """
    auth_tokens = {}
    bearer_token = None

    # Iterate through all headers
    for header_name, header_value in request.headers.items():
        # Extract Authorization bearer token
        if header_name.lower() == "authorization" and header_value.startswith("Bearer "):
            bearer_token = header_value[7:]  # Remove "Bearer " prefix
            logger.debug("Extracted Authorization bearer token")

        # Check if header matches X-Authorization-* pattern (case-insensitive)
        elif header_name.lower().startswith("x-authorization-"):
            # Extract service name (everything after "X-Authorization-")
            service_name = header_name[16:]  # len("X-Authorization-") = 16
            auth_tokens[service_name] = header_value

            logger.debug(f"Extracted custom auth header for service: {service_name}")

    return CustomAuthHeaders(bearer_token=bearer_token, auth_tokens=auth_tokens)


def auth_required(func: Callable) -> Callable:
    """Decorator to require authentication for a handler function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # The user context should be injected as the first argument
        # This decorator is meant to be used with auth-enabled handlers
        return await func(*args, **kwargs)
    return wrapper


def setup_auth_from_env() -> Optional[AuthConfig]:
    """Setup auth configuration from environment variables."""
    if not Constants.AUTH_ENABLED:
        return None
    
    if not Constants.AUTH_ISSUER or not Constants.AUTH_AUDIENCE or not Constants.AUTH_JWKS_URL:
        logger.warning("Auth is enabled but issuer, audience, or JWKS URL not configured properly")
        logger.warning("Required environment variables: AUTH_ISSUER, AUTH_AUDIENCE, AUTH_JWKS_URL")
        return None
    
    config = AuthConfig(
        enabled=True,
        issuer=Constants.AUTH_ISSUER,
        audience=Constants.AUTH_AUDIENCE,
        jwks_url=Constants.AUTH_JWKS_URL,
        algorithms=Constants.AUTH_ALGORITHMS,
        verify_exp=Constants.AUTH_VERIFY_EXP,
        verify_aud=Constants.AUTH_VERIFY_AUD
    )
    
    return config 