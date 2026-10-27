"""Personal access tokens: create (shown once), list (never the secret), revoke.

Tokens authenticate API calls as "Authorization: Bearer dm_..." (see
core.auth.get_current_user). Managing them needs the browser session: a token
can't mint or revoke tokens, so a leaked one can't make itself permanent.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, field_validator

from core.auth import (
    PAT_DISPLAY_LENGTH,
    generate_personal_access_token,
    get_session_user,
    hash_personal_access_token,
)
from core.database import get_pool

router = APIRouter()

MAX_NAME_LENGTH = 100
_COLUMNS = "id, name, prefix, created_at, last_used_at, revoked_at"


class CreateTokenRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        if len(v) > MAX_NAME_LENGTH:
            raise ValueError(f"Name too long (max {MAX_NAME_LENGTH} chars)")
        return v


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _token_info(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "prefix": row["prefix"],
        "created_at": _iso(row["created_at"]),
        "last_used_at": _iso(row["last_used_at"]),
        "revoked_at": _iso(row["revoked_at"]),
    }


@router.post("/tokens", status_code=status.HTTP_201_CREATED)
async def create_token(
    body: CreateTokenRequest,
    response: Response,
    current_user: dict = Depends(get_session_user),
):
    """The new token, in full, this once; only its hash is stored."""
    token = generate_personal_access_token()
    row = await get_pool().fetchrow(
        f"""
        INSERT INTO personal_access_tokens (id, user_id, name, token_hash, prefix)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING {_COLUMNS}
        """,
        str(uuid.uuid4()),
        current_user["id"],
        body.name,
        hash_personal_access_token(token),
        token[:PAT_DISPLAY_LENGTH],
    )
    response.headers["Cache-Control"] = "no-store"
    return {**_token_info(row), "token": token}


@router.get("/tokens")
async def list_tokens(current_user: dict = Depends(get_session_user)):
    rows = await get_pool().fetch(
        f"SELECT {_COLUMNS} FROM personal_access_tokens WHERE user_id = $1 ORDER BY created_at DESC",
        current_user["id"],
    )
    return {"tokens": [_token_info(r) for r in rows]}


@router.delete("/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(token_id: str, current_user: dict = Depends(get_session_user)):
    """Revoke (not delete) the caller's token; revoking twice is a no-op. Others' tokens are 404."""
    pool = get_pool()
    owned = await pool.fetchval(
        "SELECT 1 FROM personal_access_tokens WHERE id = $1 AND user_id = $2",
        token_id,
        current_user["id"],
    )
    if not owned:
        raise HTTPException(404, "Token not found")
    await pool.execute(
        "UPDATE personal_access_tokens SET revoked_at = now() "
        "WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL",
        token_id,
        current_user["id"],
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
