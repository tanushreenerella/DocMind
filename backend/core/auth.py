import hmac
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


async def get_current_user(
    access_token: Optional[str] = Cookie(default=None),
) -> dict:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )
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
    """
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
