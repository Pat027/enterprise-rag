"""API key auth (Bearer token) for FastAPI endpoints.

Behavior
--------
* When ``API_KEYS`` is empty or unset, auth is **disabled** — every request is
  accepted and identified by client IP address. A startup warning is logged.
* When ``API_KEYS`` is set (comma-separated), requests must include
  ``Authorization: Bearer <key>`` matching one of the configured keys, or a 401
  is returned.

The dependency yields a ``caller_id`` string used downstream (e.g. by the rate
limiter). For authenticated calls we use the key's prefix
(everything before the last ``-``); for unauthenticated calls we use the
client's IP address.

NOTE: This is process-local. For multi-process / multi-replica deploys a shared
key store (and the rate-limit's shared bucket store) would be required.
"""

from __future__ import annotations

import structlog
from fastapi import Header, HTTPException, Request, status

from ..config import get_settings

log = structlog.get_logger()

_warned_disabled = False


def _key_prefix(key: str) -> str:
    """Return a short, log-safe identifier for a key."""
    if "-" in key:
        # e.g. "sk-alice-abc123" → "sk-alice"
        parts = key.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
    return key[:8]


def warn_if_auth_disabled() -> None:
    """Call once at startup to log a warning when auth is disabled."""
    global _warned_disabled
    if not get_settings().api_keys() and not _warned_disabled:
        log.warning(
            "auth_disabled",
            message=(
                "API_KEYS env var is empty — authentication is DISABLED. "
                "All callers will be rate-limited by IP."
            ),
        )
        _warned_disabled = True


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    """FastAPI dependency: validate Bearer token, return ``caller_id``.

    If auth is disabled, returns the client IP as caller_id.
    """
    valid_keys = get_settings().api_keys()

    if not valid_keys:
        # Auth disabled — identify by IP for rate limiting.
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid Authorization header (expected 'Bearer <token>')",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()
    if token not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _key_prefix(token)
