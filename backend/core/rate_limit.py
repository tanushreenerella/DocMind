"""Shared slowapi limiter: keyed per user when logged in, else per client IP.

The client IP comes only from x-client-ip, which the Next.js proxy sets and the
backend trusts because of the proxy-secret check. X-Forwarded-For and the TCP
peer are never used: the former is client-controlled, the latter is the proxy.
"""

import math
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from limits import parse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from core.auth import ACCESS_TOKEN_COOKIE
from core.config import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    RATE_LIMIT_LOGIN_EMAIL,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_STORAGE_URI,
)

_FALLBACK_RETRY_AFTER_SECONDS = 60
CLIENT_IP_HEADER = "x-client-ip"
_LOGIN_EMAIL_LIMIT = parse(RATE_LIMIT_LOGIN_EMAIL)


def rate_limit_key(request: Request) -> str:
    """User id from a valid auth cookie, else the proxy-supplied x-client-ip.

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

    client_ip = request.headers.get(CLIENT_IP_HEADER, "").strip()
    return f"ip:{client_ip or 'unknown'}"


limiter = Limiter(
    key_func=rate_limit_key,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=RATE_LIMIT_STORAGE_URI,
    in_memory_fallback_enabled=True,
)


def _seconds_until_reset(limit_item, *key_args) -> int:
    try:
        reset_at, _remaining = limiter.limiter.get_window_stats(limit_item, *key_args)
        return max(1, math.ceil(reset_at - time.time()))
    except Exception:
        return _FALLBACK_RETRY_AFTER_SECONDS


def _retry_after_seconds(request: Request) -> int:
    try:
        limit_item, key_args = request.state.view_rate_limit
    except Exception:
        return _FALLBACK_RETRY_AFTER_SECONDS
    return _seconds_until_reset(limit_item, *key_args)


def too_many_requests(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests, try again shortly"},
        headers={"Retry-After": str(retry_after)},
    )


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return too_many_requests(_retry_after_seconds(request))


def hit_login_email_limit(email: str) -> int | None:
    """Count a login attempt for this email; return Retry-After seconds if over the limit.

    Separate from the per-user/IP limit so one account can't be brute-forced
    from many IPs.
    """
    key = ("login-email", email.strip().lower())
    try:
        allowed = limiter.limiter.hit(_LOGIN_EMAIL_LIMIT, *key)
    except Exception as exc:
        # Storage down: fail open like slowapi's own limits.
        print(f"[rate-limit] login email limit unavailable: {exc}")
        return None
    return None if allowed else _seconds_until_reset(_LOGIN_EMAIL_LIMIT, *key)
