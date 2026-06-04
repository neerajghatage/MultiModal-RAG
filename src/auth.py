"""
JWT Bearer Token authentication middleware for the RAG API.

Architecture aligned with RAG API:
- Validates JWT tokens issued by Azure AD (Entra ID)
- Verifies audience, issuer, signature, and expiry
- Fetches OIDC discovery document and caches signing keys
- Supports configurable tenant, client ID, and allowed domains/groups

Configuration (via environment variables / .env):
    AZURE_AD_TENANT_ID          — Entra ID tenant
    AZURE_AD_CLIENT_ID          — App Registration client ID (expected audience)
    AZURE_AD_ALLOWED_DOMAIN     — Restrict to email domain (e.g., <your-domain.com>)
    AZURE_AD_ALLOWED_GROUP_ID   — Restrict to AD group members (optional)
    AUTH_ENABLED                 — Set to "false" to disable auth (local dev only)
"""

import logging
import os
import time
from typing import Optional

import httpx
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

AZURE_AD_TENANT_ID = os.getenv("AZURE_AD_TENANT_ID", "")
AZURE_AD_CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID", "")
AZURE_AD_ALLOWED_DOMAIN = os.getenv("AZURE_AD_ALLOWED_DOMAIN", "")
AZURE_AD_ALLOWED_GROUP_ID = os.getenv("AZURE_AD_ALLOWED_GROUP_ID", "")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() != "false"

_OIDC_DISCOVERY_URL = (
    f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0/.well-known/openid-configuration"
    if AZURE_AD_TENANT_ID
    else ""
)

# ── OIDC Key Cache ────────────────────────────────────────────

_jwks_cache: dict = {}
_jwks_cache_expiry: float = 0
_JWKS_CACHE_TTL = 3600  # 1 hour


async def _get_signing_keys() -> dict:
    """Fetch and cache JWKS from Azure AD OIDC discovery endpoint."""
    global _jwks_cache, _jwks_cache_expiry

    if _jwks_cache and time.time() < _jwks_cache_expiry:
        return _jwks_cache

    if not _OIDC_DISCOVERY_URL:
        raise HTTPException(status_code=500, detail="OIDC discovery URL not configured")

    async with httpx.AsyncClient() as client:
        # Fetch discovery document
        discovery_resp = await client.get(_OIDC_DISCOVERY_URL)
        discovery_resp.raise_for_status()
        discovery = discovery_resp.json()

        # Fetch JWKS
        jwks_uri = discovery["jwks_uri"]
        jwks_resp = await client.get(jwks_uri)
        jwks_resp.raise_for_status()

        _jwks_cache = jwks_resp.json()
        _jwks_cache_expiry = time.time() + _JWKS_CACHE_TTL

    logger.info("Refreshed OIDC signing keys from %s", jwks_uri)
    return _jwks_cache


def _get_issuer() -> str:
    """Return the expected token issuer for the configured tenant."""
    return f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0"


def _get_issuers() -> list:
    """Return all accepted issuer values (v1 and v2 endpoints)."""
    return [
        f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0",
        f"https://sts.windows.net/{AZURE_AD_TENANT_ID}/",
    ]


# ── Security Scheme ───────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


async def validate_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> Optional[dict]:
    """
    FastAPI dependency that validates the JWT Bearer token.

    Returns the decoded token claims if valid, or None if auth is disabled.
    Raises HTTP 401/403 on invalid/missing tokens when auth is enabled.
    """
    if not AUTH_ENABLED or not AZURE_AD_TENANT_ID or not AZURE_AD_CLIENT_ID:
        # Auth not configured — allow all (local development)
        return None

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Get signing keys
        jwks = await _get_signing_keys()

        # Decode and validate the token
        # Accept both GUID and api:// URI forms of audience (v1 vs v2 tokens)
        expected_audiences = {AZURE_AD_CLIENT_ID, f"api://{AZURE_AD_CLIENT_ID}"}
        valid_issuers = _get_issuers()
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={
                "verify_aud": False,
                "verify_iss": False,
                "verify_exp": True,
                "verify_nbf": True,
            },
        )

        # Manual audience validation
        token_aud = claims.get("aud", "")
        if token_aud not in expected_audiences:
            raise JWTError(f"Invalid audience: {token_aud}")

        # Manual issuer validation (accept both v1 and v2 issuer formats)
        token_issuer = claims.get("iss", "")
        if token_issuer not in valid_issuers:
            raise JWTError(f"Invalid issuer: {token_issuer}")

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        logger.warning("JWT validation failed: %s", str(e))
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Additional claims validation ──────────────────────────

    # Validate email domain
    email = claims.get("preferred_username", "") or claims.get("email", "")
    if AZURE_AD_ALLOWED_DOMAIN and email:
        if not email.lower().endswith(f"@{AZURE_AD_ALLOWED_DOMAIN.lower()}"):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Only @{AZURE_AD_ALLOWED_DOMAIN} accounts are allowed.",
            )

    # Validate group membership
    if AZURE_AD_ALLOWED_GROUP_ID:
        user_groups = claims.get("groups", [])
        if AZURE_AD_ALLOWED_GROUP_ID not in user_groups:
            raise HTTPException(
                status_code=403,
                detail="Access denied. User is not a member of the required group.",
            )

    return claims
