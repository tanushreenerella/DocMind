from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from core.auth import hash_password, verify_password, create_access_token, get_current_user
from core.database import get_pool

router = APIRouter(prefix="/auth", tags=["auth"])


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
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest):
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

    token = create_access_token(str(row["id"]), row["email"])
    return AuthResponse(
        access_token=token,
        user_id=str(row["id"]),
        email=row["email"],
        full_name=row["full_name"] or "",
    )
@router.post("/login", response_model=AuthResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
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

    token = create_access_token(str(row["id"]), row["email"])

    return AuthResponse(
        access_token=token,
        user_id=str(row["id"]),
        email=row["email"],
        full_name=row["full_name"] or "",
    )

@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user
