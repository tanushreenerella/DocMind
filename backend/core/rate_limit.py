"""Shared slowapi limiter: keyed per user when logged in, else per client IP."""

import math
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from core.auth import ACCESS_TOKEN_COOKIE
from core.config import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_STORAGE_URI,
)

_FALLBACK_RETRY_AFTER_SECONDS = 60


def rate_limit_key(request: Request) -> str:
    """User id from a valid auth cookie, else first X-Forwarded-For IP, else peer IP.

    The cookie's signature is verified (no DB lookup) so a forged token can't
    pick someone else's bucket; an invalid one falls back to the IP key.
    """
    token = request.cookies.get(ACCESS_TOKEN_COOKIE)
    if token:
        try:
            user_id = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]).get("sub")
            if user_id:
                return f"user:{user_id}"
        except JWTError:
            pass

    forwarded_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded_ip:
        return f"ip:{forwarded_ip}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


limiter = Limiter(
    key_func=rate_limit_key,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=RATE_LIMIT_STORAGE_URI,
    in_memory_fallback_enabled=True,
)


def _retry_after_seconds(request: Request) -> int:
    try:
        limit_item, key_args = request.state.view_rate_limit
        reset_at, _remaining = limiter.limiter.get_window_stats(limit_item, *key_args)
        return max(1, math.ceil(reset_at - time.time()))
    except Exception:
        return _FALLBACK_RETRY_AFTER_SECONDS


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests, try again shortly"},
        headers={"Retry-After": str(_retry_after_seconds(request))},
    )
