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
)


async def ensure_schema() -> None:
    pool = get_pool()
    for statement in SCHEMA_UPDATES:
        await pool.execute(statement)
