import asyncpg
from core.config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def connect():
    global _pool
    _pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        # Drop idle connections after 4 min — Neon suspends compute at 5 min
        max_inactive_connection_lifetime=240,
        # If a connection is broken, retry the query once automatically
        command_timeout=30,
    )


async def disconnect():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool


# Additive, idempotent schema changes, applied at startup (there is no
# migration tool). Each must be safe to run any number of times.
SCHEMA_UPDATES = (
    # Pages whose OCR failed for a "partial" document.
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS failed_pages INTEGER[] NOT NULL DEFAULT '{}'",
    # Hidden-text injection: pages it was stripped from, characters removed, and the flag.
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS hidden_text_pages INTEGER[] NOT NULL DEFAULT '{}'",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS hidden_text_removed INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS suspicious BOOLEAN NOT NULL DEFAULT FALSE",
    # Daily Groq token counts when Redis isn't configured or is down (core/token_budget.py).
    "CREATE TABLE IF NOT EXISTS token_usage ("
    "user_id TEXT NOT NULL, day DATE NOT NULL, tokens BIGINT NOT NULL DEFAULT 0, "
    "PRIMARY KEY (user_id, day))",
    # Personal access tokens (routers/tokens.py). Only the SHA-256 of a token is
    # kept; prefix is its first few characters, for telling tokens apart.
    "CREATE TABLE IF NOT EXISTS personal_access_tokens ("
    "id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, "
    "token_hash TEXT NOT NULL UNIQUE, prefix TEXT NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), last_used_at TIMESTAMPTZ, revoked_at TIMESTAMPTZ)",
    "CREATE INDEX IF NOT EXISTS personal_access_tokens_user_id ON personal_access_tokens (user_id)",
)


async def ensure_schema() -> None:
    pool = get_pool()
    for statement in SCHEMA_UPDATES:
        await pool.execute(statement)
