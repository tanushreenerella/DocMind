"""Per-user daily Groq token budget.

Every Groq chat call made while a user is bound (``charged_to``) adds the
response's ``usage.total_tokens`` to that user's count for the current UTC day.
The chat endpoints call ``ensure_within_budget`` first and answer 429 once the
day's count has reached DAILY_TOKEN_BUDGET (so does the evidence timeline;
uploads skip classification instead of failing). A request that starts under the
budget runs to the end, so a day can overshoot by up to one answer.

Counts live in Redis when TOKEN_BUDGET_REDIS_URL is set, else in Postgres,
else in process memory. Each read and write falls through to the next store if
one fails (and skips a failed one for _RETRY_FAILED_STORE_S), so an outage
never blocks chat. During one the count is only what
that fallback has seen, so the budget is enforced loosely until it recovers.
"""

import math
import time as _time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, time, timedelta, timezone

from fastapi import HTTPException

from core.config import DAILY_TOKEN_BUDGET, TOKEN_BUDGET_REDIS_URL
from core.database import get_pool

BUDGET_EXCEEDED_MESSAGE = "Daily AI usage limit reached, resets at midnight UTC"
_KEY_PREFIX = "token-usage"
# A day's Redis key outlives that day by a margin, then expires on its own.
_KEY_TTL_SECONDS = 2 * 24 * 3600

_charged_user: ContextVar[str | None] = ContextVar("token_budget_user", default=None)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def seconds_until_reset(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    midnight = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return max(1, math.ceil((midnight - now).total_seconds()))


# ── stores ──────────────────────────────────────────────────────────────────

class RedisStore:
    name = "redis"

    def __init__(self, url: str) -> None:
        self.url = url
        self._client = None

    def _redis(self):
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.Redis.from_url(self.url, socket_timeout=1, socket_connect_timeout=1)
        return self._client

    @staticmethod
    def _key(user_id: str, day: date) -> str:
        return f"{_KEY_PREFIX}:{user_id}:{day.isoformat()}"

    async def add(self, user_id: str, day: date, tokens: int) -> None:
        key = self._key(user_id, day)
        async with self._redis().pipeline(transaction=True) as pipe:
            await pipe.incrby(key, tokens).expire(key, _KEY_TTL_SECONDS).execute()

    async def used(self, user_id: str, day: date) -> int:
        return int(await self._redis().get(self._key(user_id, day)) or 0)


class PostgresStore:
    """The token_usage table (core.database.SCHEMA_UPDATES)."""

    name = "postgres"

    async def add(self, user_id: str, day: date, tokens: int) -> None:
        await get_pool().execute(
            """
            INSERT INTO token_usage (user_id, day, tokens) VALUES ($1, $2, $3)
            ON CONFLICT (user_id, day) DO UPDATE SET tokens = token_usage.tokens + EXCLUDED.tokens
            """,
            user_id,
            day,
            tokens,
        )

    async def used(self, user_id: str, day: date) -> int:
        tokens = await get_pool().fetchval(
            "SELECT tokens FROM token_usage WHERE user_id = $1 AND day = $2", user_id, day
        )
        return int(tokens or 0)


class MemoryStore:
    """Per process: with several workers each keeps its own count."""

    name = "memory"

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, date], int] = {}

    async def add(self, user_id: str, day: date, tokens: int) -> None:
        # Earlier days can never be read again.
        for key in [k for k in self._tokens if k[1] < day]:
            del self._tokens[key]
        self._tokens[(user_id, day)] = self._tokens.get((user_id, day), 0) + tokens

    async def used(self, user_id: str, day: date) -> int:
        return self._tokens.get((user_id, day), 0)


def _build_stores() -> list:
    stores: list = [PostgresStore(), MemoryStore()]
    if TOKEN_BUDGET_REDIS_URL:
        stores.insert(0, RedisStore(TOKEN_BUDGET_REDIS_URL))
    return stores


_stores = _build_stores()
# A store that failed is skipped this long, so an outage doesn't add a connect
# timeout to every chat request, and is logged once rather than per call.
_RETRY_FAILED_STORE_S = 30.0
# Store name -> time.monotonic() at which to try it again.
_failing: dict[str, float] = {}


async def _first_working(operation: str, *args):
    for store in _stores:
        if _failing.get(store.name, 0.0) > _time.monotonic():
            continue
        try:
            result = await getattr(store, operation)(*args)
        except Exception as exc:
            if store.name not in _failing:
                print(f"[token-budget] {store.name} unavailable, falling back: {type(exc).__name__}: {exc}")
            _failing[store.name] = _time.monotonic() + _RETRY_FAILED_STORE_S
            continue
        if _failing.pop(store.name, None) is not None:
            print(f"[token-budget] {store.name} back")
        return result
    raise AssertionError("unreachable: MemoryStore never fails")


# ── public API ──────────────────────────────────────────────────────────────

@contextmanager
def charged_to(user_id: str):
    """Charge Groq calls made inside this block to ``user_id``."""
    token = _charged_user.set(str(user_id))
    try:
        yield
    finally:
        _charged_user.reset(token)


async def tokens_used(user_id: str) -> int:
    return await _first_working("used", str(user_id), _today())


async def record_usage(response) -> None:
    """Add a Groq response's total tokens to the bound user's count for today.

    Does nothing when no user is bound (eval scripts) or the response carries
    no usage. Never raises: a lost count must not lose the answer.
    """
    user_id = _charged_user.get()
    total = getattr(getattr(response, "usage", None), "total_tokens", None)
    if user_id is None or not isinstance(total, int) or total <= 0:
        return
    try:
        await _first_working("add", user_id, _today(), total)
    except Exception as exc:
        print(f"[token-budget] could not record {total} tokens for {user_id}: {exc}")


async def over_budget(user_id: str) -> bool:
    """Whether ``user_id`` has used up today's DAILY_TOKEN_BUDGET."""
    return await tokens_used(user_id) >= DAILY_TOKEN_BUDGET


async def ensure_within_budget(user_id: str) -> None:
    """Raise 429 if ``user_id`` has used up today's DAILY_TOKEN_BUDGET."""
    if await over_budget(user_id):
        raise HTTPException(
            status_code=429,
            detail=BUDGET_EXCEEDED_MESSAGE,
            headers={"Retry-After": str(seconds_until_reset())},
        )
