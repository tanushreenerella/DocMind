from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from core.auth import (
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    hash_password,
    set_auth_cookie,
    verify_password,
)
from core.config import RATE_LIMIT_AUTH
from core.database import get_pool
from core.rate_limit import limiter

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str = ""

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str
    full_name: str


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(RATE_LIMIT_AUTH)
async def signup(request: Request, body: SignupRequest, response: Response):
    pool = get_pool()

    existing = await pool.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(body.password)
    row = await pool.fetchrow(
        """
        INSERT INTO users (email, hashed_password, full_name)
        VALUES ($1, $2, $3)
        RETURNING id, email, full_name
        """,
        body.email,
        hashed,
        body.full_name,
    )

    set_auth_cookie(response, create_access_token(str(row["id"]), row["email"]))
    return AuthResponse(
        user_id=str(row["id"]),
        email=row["email"],
        full_name=row["full_name"] or "",
    )
@router.post("/login", response_model=AuthResponse)
@limiter.limit(RATE_LIMIT_AUTH)
async def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    pool = get_pool()

    row = await pool.fetchrow(
        "SELECT id, email, hashed_password, full_name FROM users WHERE email = $1",
        form_data.username,
    )

    if not row or not verify_password(
        form_data.password,
        row["hashed_password"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    set_auth_cookie(response, create_access_token(str(row["id"]), row["email"]))

    return AuthResponse(
        user_id=str(row["id"]),
        email=row["email"],
        full_name=row["full_name"] or "",
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response):
    clear_auth_cookie(response)


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user
