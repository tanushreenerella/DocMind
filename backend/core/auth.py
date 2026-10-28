import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from fastapi import Cookie, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import (
    ALLOWED_ORIGINS,
    COOKIE_SECURE,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET_KEY,
    PROXY_SECRET,
)
from core.database import get_pool

ACCESS_TOKEN_COOKIE = "access_token"
# Personal access tokens: "dm_" + 32 random bytes, sent as "Authorization: Bearer dm_...".
PAT_PREFIX = "dm_"
# What listings show of a token so users can tell them apart: "dm_" + 8 of its 43 random characters.
PAT_DISPLAY_LENGTH = len(PAT_PREFIX) + 8
# The MCP server (routers/mcp.py): token-only, so exempt from the cookie CSRF check.
MCP_PATH = "/api/mcp"
PROXY_SECRET_HEADER = "x-proxy-secret"
_STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=token,
        max_age=JWT_EXPIRE_MINUTES * 60,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


def clear_auth_cookie(response: Response) -> None:
    # Attributes must match set_auth_cookie or browsers keep the original cookie.
    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


def generate_personal_access_token() -> str:
    return PAT_PREFIX + secrets.token_urlsafe(32)


def hash_personal_access_token(token: str) -> str:
    """SHA-256 hex digest: the only form a token is stored in.

    Unsalted is enough here: the token carries 256 random bits, so there is
    nothing to brute-force or look up in a table.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def _bearer_token(request: Request) -> Optional[str]:
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    return credentials.strip() if scheme.lower() == "bearer" else None


def _credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )


async def get_current_user(
    request: Request,
    access_token: Optional[str] = Cookie(default=None),
) -> dict:
    """The caller, from an "Authorization: Bearer dm_..." token if one is sent, else the session cookie.

    A Bearer header is never ignored: if it isn't a valid, unrevoked personal
    access token the request is 401 even when a valid cookie is present.
    """
    bearer = _bearer_token(request)
    if bearer is not None:
        return await user_from_personal_access_token(bearer)
    return await _user_from_session(access_token)


async def get_session_user(access_token: Optional[str] = Cookie(default=None)) -> dict:
    """The caller from the session cookie only: for managing tokens, which a token itself may not do."""
    return await _user_from_session(access_token)


async def user_from_personal_access_token(token: str) -> dict:
    """The owner of an unrevoked personal access token, checked against the DB every time; else 401."""
    if not token.startswith(PAT_PREFIX):
        raise _credentials_error()
    pool = get_pool()
    # users.id is compared as text so this works whatever its column type.
    row = await pool.fetchrow(
        """
        SELECT t.id AS token_id, u.id, u.email, u.full_name
        FROM personal_access_tokens t
        JOIN users u ON u.id::text = t.user_id
        WHERE t.token_hash = $1 AND t.revoked_at IS NULL
        """,
        hash_personal_access_token(token),
    )
    if row is None:
        raise _credentials_error()
    await pool.execute(
        "UPDATE personal_access_tokens SET last_used_at = now() WHERE id = $1", row["token_id"]
    )
    return {"id": str(row["id"]), "email": row["email"], "full_name": row["full_name"]}


# ── token -> user for rate limiting ─────────────────────────────────────────

# How long a token's owner (or its being invalid) is remembered for rate-limit
# keying. Authentication never uses this cache, so a revoked token stops
# working at once; it just keeps its user's rate-limit bucket for up to this long.
TOKEN_USER_CACHE_SECONDS = 60.0
_TOKEN_USER_CACHE_MAX = 10_000
# Token hash -> (owner's user id, or None for an unknown/revoked token; expiry on time.monotonic()).
_token_user_cache: dict[str, tuple[Optional[str], float]] = {}


async def cached_token_user_id(token: str) -> Optional[str]:
    """The user id behind a valid personal access token, cached for TOKEN_USER_CACHE_SECONDS."""
    if not token.startswith(PAT_PREFIX):
        return None
    token_hash = hash_personal_access_token(token)
    now = time.monotonic()
    cached = _token_user_cache.get(token_hash)
    if cached is not None and cached[1] > now:
        return cached[0]

    user_id = await get_pool().fetchval(
        """
        SELECT t.user_id
        FROM personal_access_tokens t
        JOIN users u ON u.id::text = t.user_id
        WHERE t.token_hash = $1 AND t.revoked_at IS NULL
        """,
        token_hash,
    )
    if len(_token_user_cache) >= _TOKEN_USER_CACHE_MAX:
        for key in [k for k, (_, expires) in _token_user_cache.items() if expires <= now]:
            del _token_user_cache[key]
        if len(_token_user_cache) >= _TOKEN_USER_CACHE_MAX:
            _token_user_cache.clear()
    _token_user_cache[token_hash] = (str(user_id) if user_id else None, now + TOKEN_USER_CACHE_SECONDS)
    return _token_user_cache[token_hash][0]


async def identify_token_user(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Put a valid token's user id on ``request.state.token_user_id`` before rate limiting.

    The rate limiter then counts token requests per user rather than per IP
    (core.rate_limit.rate_limit_key). An invalid token, or a DB error, leaves it
    unset: the request is limited by IP and rejected later by get_current_user.
    """
    bearer = _bearer_token(request)
    if bearer is not None and bearer.startswith(PAT_PREFIX):
        try:
            user_id = await cached_token_user_id(bearer)
        except Exception as exc:
            print(f"[auth] token lookup for rate limiting failed: {type(exc).__name__}: {exc}")
            user_id = None
        if user_id:
            request.state.token_user_id = user_id
    return await call_next(request)


async def _user_from_session(access_token: Optional[str]) -> dict:
    credentials_exc = _credentials_error()
    if not access_token:
        raise credentials_exc
    try:
        payload = jwt.decode(access_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        email: Optional[str] = payload.get("email")
        if user_id is None or email is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    pool = get_pool()
    row = await pool.fetchrow("SELECT id, email, full_name FROM users WHERE id = $1", user_id)
    if row is None:
        raise credentials_exc

    return {"id": str(row["id"]), "email": row["email"], "full_name": row["full_name"]}


async def require_allowed_origin(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """CSRF guard: state-changing requests must come from an allowed Origin.

    Requests without an Origin header are rejected too, since the auth cookie
    is sent automatically and the header is the only proof of where they came from.

    Requests carrying a personal access token are exempt: they authenticate
    with that token, never the cookie (see get_current_user), and a cross-site
    page can't attach an Authorization header without passing CORS preflight.
    That is what lets scripts without an Origin use POST/DELETE endpoints.
    """
    bearer = _bearer_token(request)
    if bearer is not None and bearer.startswith(PAT_PREFIX):
        return await call_next(request)
    # The MCP endpoint accepts only tokens, never the cookie, so there's nothing to forge;
    # a client without a token gets its 401 rather than "Origin not allowed".
    if request.url.path == MCP_PATH:
        return await call_next(request)
    if (
        request.method in _STATE_CHANGING_METHODS
        and request.headers.get("origin") not in ALLOWED_ORIGINS
    ):
        return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})
    return await call_next(request)


async def require_proxy_secret(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Only the Next.js proxy may call /api: it alone knows PROXY_SECRET.

    This is also what makes its x-client-ip header trustworthy for rate limiting.
    """
    path = request.url.path
    if path == "/api" or path.startswith("/api/"):
        supplied = request.headers.get(PROXY_SECRET_HEADER, "")
        if not hmac.compare_digest(supplied.encode(), PROXY_SECRET.encode()):
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)
