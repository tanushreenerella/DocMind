"""Daily token budget: recording Groq usage, store fallback, and the 429 on chat."""

import asyncio
import contextlib
import io
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from core import config, token_budget
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from core.token_budget import (
    BUDGET_EXCEEDED_MESSAGE,
    MemoryStore,
    PostgresStore,
    RedisStore,
    charged_to,
    ensure_within_budget,
    record_usage,
    tokens_used,
)
from main import app
from routers import upload
from services import classifier, rag
from services.classifier import BUDGET_SKIPPED_SUMMARY, budget_skipped_classification
from tests import PROXY_HEADERS
from tests.test_document_storage import USER as UPLOAD_USER, _ProcessingTestCase

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TODAY = date(2026, 10, 27)


def _groq_response(total_tokens) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=total_tokens), choices=[])


class FailingStore:
    name = "failing"

    async def add(self, *args) -> None:
        raise ConnectionError("down")

    async def used(self, *args) -> int:
        raise ConnectionError("down")


class FakeTokenPool:
    """The two token_usage statements PostgresStore runs, over a dict."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, date], int] = {}

    async def execute(self, query: str, user_id: str, day: date, tokens: int) -> None:
        assert "ON CONFLICT (user_id, day) DO UPDATE" in query
        self.rows[(user_id, day)] = self.rows.get((user_id, day), 0) + tokens

    async def fetchval(self, query: str, user_id: str, day: date):
        assert query.startswith("SELECT tokens FROM token_usage")
        return self.rows.get((user_id, day))


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis
        self.queued: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def incrby(self, key: str, amount: int):
        self.queued.append(("incrby", key, amount))
        return self

    def expire(self, key: str, seconds: int):
        self.queued.append(("expire", key, seconds))
        return self

    async def execute(self) -> list:
        for op, key, value in self.queued:
            if op == "incrby":
                self.redis.values[key] = self.redis.values.get(key, 0) + value
            else:
                self.redis.ttls[key] = value
        return []


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.transactions: list[bool] = []

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        self.transactions.append(transaction)
        return FakePipeline(self)

    async def get(self, key: str):
        value = self.values.get(key)
        return None if value is None else str(value).encode()


class _BudgetTestCase(unittest.IsolatedAsyncioTestCase):
    """Fresh stores and a fixed day for every test."""

    stores: list

    def setUp(self) -> None:
        self.memory = MemoryStore()
        self.patches = [
            patch.object(token_budget, "_stores", getattr(self, "stores", None) or [self.memory]),
            patch.object(token_budget, "_failing", {}),
            patch.object(token_budget, "_today", return_value=TODAY),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()


class RecordUsageTests(_BudgetTestCase):
    async def test_usage_is_added_for_the_bound_user_only(self) -> None:
        with charged_to(USER):
            await record_usage(_groq_response(120))
            await record_usage(_groq_response(30))

        self.assertEqual(await tokens_used(USER), 150)
        self.assertEqual(await tokens_used(OTHER), 0)

    async def test_nothing_is_recorded_without_a_bound_user(self) -> None:
        await record_usage(_groq_response(500))
        self.assertEqual(self.memory._tokens, {})

    async def test_responses_without_usable_usage_are_ignored(self) -> None:
        with charged_to(USER):
            for response in (SimpleNamespace(), SimpleNamespace(usage=None), _groq_response(None),
                             _groq_response(0), MagicMock()):
                await record_usage(response)
        self.assertEqual(await tokens_used(USER), 0)

    async def test_binding_ends_with_the_block(self) -> None:
        with charged_to(USER):
            pass
        await record_usage(_groq_response(10))
        self.assertEqual(await tokens_used(USER), 0)

    async def test_counts_start_again_each_utc_day(self) -> None:
        with charged_to(USER):
            await record_usage(_groq_response(100))
            with patch.object(token_budget, "_today", return_value=TODAY + timedelta(days=1)):
                self.assertEqual(await tokens_used(USER), 0)
                await record_usage(_groq_response(7))
                self.assertEqual(await tokens_used(USER), 7)
        # The memory store drops days that can no longer be read.
        self.assertEqual(list(self.memory._tokens), [(USER, TODAY + timedelta(days=1))])


class EnsureWithinBudgetTests(_BudgetTestCase):
    async def test_under_the_budget_passes(self) -> None:
        with patch.object(token_budget, "DAILY_TOKEN_BUDGET", 100):
            await self.memory.add(USER, TODAY, 99)
            await ensure_within_budget(USER)

    async def test_at_or_over_the_budget_is_429_until_midnight_utc(self) -> None:
        with patch.object(token_budget, "DAILY_TOKEN_BUDGET", 100):
            await self.memory.add(USER, TODAY, 100)
            with self.assertRaises(HTTPException) as caught:
                await ensure_within_budget(USER)

        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.detail, "Daily AI usage limit reached, resets at midnight UTC")
        retry_after = int(caught.exception.headers["Retry-After"])
        self.assertTrue(1 <= retry_after <= 86400)

    def test_seconds_until_reset(self) -> None:
        def at(*hms) -> datetime:
            return datetime(2026, 10, 27, *hms, tzinfo=timezone.utc)

        self.assertEqual(token_budget.seconds_until_reset(at(23, 59, 30)), 30)
        self.assertEqual(token_budget.seconds_until_reset(at(0, 0, 0)), 86400)
        self.assertEqual(token_budget.seconds_until_reset(at(23, 59, 59, 900000)), 1)


class StoreFallbackTests(_BudgetTestCase):
    def setUp(self) -> None:
        self.pool = FakeTokenPool()
        self.stores = [FailingStore(), PostgresStore(), MemoryStore()]
        super().setUp()
        self.patches.append(patch.object(token_budget, "get_pool", return_value=self.pool))
        self.patches[-1].start()

    async def test_redis_down_falls_back_to_postgres(self) -> None:
        with charged_to(USER), contextlib.redirect_stdout(io.StringIO()) as log:
            await record_usage(_groq_response(40))
            await record_usage(_groq_response(2))
            used = await tokens_used(USER)

        self.assertEqual(used, 42)
        self.assertEqual(self.pool.rows, {(USER, TODAY): 42})
        # The outage is logged once, not on every call.
        self.assertEqual(log.getvalue().count("failing unavailable"), 1)

    async def test_postgres_down_too_falls_back_to_memory(self) -> None:
        with (
            patch.object(token_budget, "get_pool", side_effect=RuntimeError("Database pool not initialised")),
            charged_to(USER),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await record_usage(_groq_response(5))
            self.assertEqual(await tokens_used(USER), 5)
        self.assertEqual(token_budget._stores[2]._tokens, {(USER, TODAY): 5})

    async def test_a_failed_store_is_skipped_until_its_retry_time(self) -> None:
        failing = token_budget._stores[0]
        failing.used = AsyncMock(side_effect=ConnectionError("down"))
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                await tokens_used(USER)
            self.assertEqual(failing.used.await_count, 1)

            with patch.object(token_budget._time, "monotonic", return_value=token_budget._time.monotonic() + 31):
                await tokens_used(USER)
            self.assertEqual(failing.used.await_count, 2)

    async def test_recovery_is_logged(self) -> None:
        healthy = MemoryStore()
        healthy.name = "failing"
        with patch.object(token_budget, "_RETRY_FAILED_STORE_S", 0.0), contextlib.redirect_stdout(io.StringIO()) as log:
            await tokens_used(USER)
            token_budget._stores[0] = healthy
            await tokens_used(USER)
        self.assertIn("failing back", log.getvalue())


class RedisStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_per_user_and_day_with_an_expiring_key(self) -> None:
        store = RedisStore("redis://example:6379/0")
        store._client = fake = FakeRedis()

        await store.add(USER, TODAY, 10)
        await store.add(USER, TODAY, 5)
        await store.add(USER, TODAY + timedelta(days=1), 1)

        self.assertEqual(await store.used(USER, TODAY), 15)
        self.assertEqual(await store.used(OTHER, TODAY), 0)
        key = f"token-usage:{USER}:2026-10-27"
        self.assertEqual(fake.values[key], 15)
        self.assertEqual(fake.ttls[key], 2 * 24 * 3600)
        self.assertEqual(fake.transactions, [True, True, True])

    async def test_unreachable_redis_falls_back(self) -> None:
        memory = MemoryStore()
        with (
            patch.object(token_budget, "_stores", [RedisStore("redis://127.0.0.1:1/0"), memory]),
            patch.object(token_budget, "_failing", {}),
            charged_to(USER),
            contextlib.redirect_stdout(io.StringIO()) as log,
        ):
            await record_usage(_groq_response(9))
            self.assertEqual(await tokens_used(USER), 9)
        self.assertIn("redis unavailable", log.getvalue())

    def test_redis_url_defaults_to_the_rate_limiters_redis(self) -> None:
        self.assertTrue(
            config.TOKEN_BUDGET_REDIS_URL == ""
            or config.TOKEN_BUDGET_REDIS_URL.startswith(("redis://", "rediss://"))
        )
        with patch.object(token_budget, "TOKEN_BUDGET_REDIS_URL", "redis://h:6379/0"):
            names = [s.name for s in token_budget._build_stores()]
        self.assertEqual(names, ["redis", "postgres", "memory"])
        with patch.object(token_budget, "TOKEN_BUDGET_REDIS_URL", ""):
            names = [s.name for s in token_budget._build_stores()]
        self.assertEqual(names, ["postgres", "memory"])


class ChatCompletionUsageTests(_BudgetTestCase):
    def _client(self, create) -> SimpleNamespace:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def test_each_successful_call_is_recorded(self) -> None:
        client = self._client(MagicMock(return_value=_groq_response(321)))
        with charged_to(USER):
            response = await rag.chat_completion(client, model="m", messages=[])
        self.assertEqual(response.usage.total_tokens, 321)
        self.assertEqual(await tokens_used(USER), 321)

    async def test_a_retried_call_is_recorded_once(self) -> None:
        create = MagicMock(side_effect=[RuntimeError("429"), _groq_response(50)])
        client = self._client(create)
        with (
            patch.object(rag, "_per_minute_wait", return_value=0.0),
            charged_to(USER),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await rag.chat_completion(client, model="m", messages=[])
        self.assertEqual(create.call_count, 2)
        self.assertEqual(await tokens_used(USER), 50)

    async def test_a_failed_call_records_nothing(self) -> None:
        client = self._client(MagicMock(side_effect=RuntimeError("500")))
        with charged_to(USER), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(rag.AIUnavailable):
            await rag.chat_completion(client, model="m", messages=[])
        self.assertEqual(await tokens_used(USER), 0)


class FakeUserPool:
    async def fetchrow(self, query: str, *args: object):
        if "FROM users WHERE id" in query:
            return {"id": args[0], "email": f"{args[0]}@example.com", "full_name": ""}
        if "FROM documents WHERE doc_id" in query:
            return {"doc_id": args[0]}  # every document is the caller's
        return None

    execute = AsyncMock()
    fetchval = AsyncMock(return_value=1)


class ChatEndpointBudgetTests(unittest.TestCase):
    TIMELINE = "/api/documents/11111111-2222-3333-4444-555555555555/evidence-timeline"
    ENDPOINTS = ("/api/chat", "/api/chat/agentic", TIMELINE)

    def setUp(self) -> None:
        limiter.reset()
        self.memory = MemoryStore()

        async def answer(*args, **kwargs) -> dict:
            # Stands in for a pipeline (or timeline) whose Groq calls used 400 tokens.
            await record_usage(_groq_response(400))
            return {"answer": "ok", "citations": [], "has_answer": True}

        self.answer = AsyncMock(side_effect=answer)
        pool = FakeUserPool()
        self.patches = [
            patch("core.auth.get_pool", return_value=pool),
            patch("routers.chat.get_pool", return_value=pool),
            patch("routers.documents.get_pool", return_value=pool),
            patch("routers.chat.answer_query", self.answer),
            patch("routers.chat.answer_agentic_query", self.answer),
            patch("routers.documents.build_evidence_timeline", self.answer),
            patch.object(token_budget, "_stores", [self.memory]),
            patch.object(token_budget, "DAILY_TOKEN_BUDGET", 1000),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app, headers={**PROXY_HEADERS, "Origin": config.ALLOWED_ORIGINS[0]})

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        limiter.reset()

    def _ask(self, endpoint: str, user: str = USER):
        token = create_access_token(user, f"{user}@example.com")
        body = None if endpoint == self.TIMELINE else {"question": "What is the total?"}
        return self.client.post(endpoint, json=body, headers={"Cookie": f"{ACCESS_TOKEN_COOKIE}={token}"})

    def test_usage_is_charged_to_the_caller_and_chat_stops_at_the_budget(self) -> None:
        for endpoint in self.ENDPOINTS:
            with self.subTest(endpoint):
                self.memory._tokens.clear()
                self.answer.reset_mock()

                for _ in range(3):  # 0 -> 400 -> 800 -> 1200 tokens
                    self.assertEqual(self._ask(endpoint).status_code, 200)
                self.assertEqual(asyncio.run(self.memory.used(USER, token_budget._today())), 1200)

                refused = self._ask(endpoint)
                self.assertEqual(refused.status_code, 429)
                self.assertEqual(refused.json(), {"detail": BUDGET_EXCEEDED_MESSAGE})
                self.assertTrue(1 <= int(refused.headers["Retry-After"]) <= 86400)
                self.assertEqual(self.answer.await_count, 3)  # no Groq work once refused

                # Other users keep their own budget.
                self.assertEqual(self._ask(endpoint, user=OTHER).status_code, 200)

    def test_all_endpoints_share_one_budget(self) -> None:
        with patch.object(token_budget, "DAILY_TOKEN_BUDGET", 400):
            self.assertEqual(self._ask("/api/chat").status_code, 200)
            self.assertEqual(self._ask("/api/chat/agentic").status_code, 429)
            self.assertEqual(self._ask(self.TIMELINE).status_code, 429)
        with patch.object(token_budget, "DAILY_TOKEN_BUDGET", 800):
            self.assertEqual(self._ask(self.TIMELINE).status_code, 200)
            self.assertEqual(self._ask("/api/chat").status_code, 429)


class UploadClassificationUsageTests(_ProcessingTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.memory = MemoryStore()

        async def classify(text: str, filename: str) -> dict:
            await record_usage(_groq_response(77))
            return {"document_type": "invoice", "sensitivity_level": "confidential", "summary": "An invoice."}

        async def parse(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
            return [{"page_number": 1, "text": "hello", "image_path": "", "image_filename": "",
                     "has_tables": False, "tables": []}]

        self.classify = AsyncMock(side_effect=classify)
        self.upload_patches = [
            patch.object(token_budget, "_stores", [self.memory]),
            patch.object(token_budget, "DAILY_TOKEN_BUDGET", 100),
            patch.object(upload, "classify_document", self.classify),
            patch.object(upload, "parse_document", parse),
        ]
        for p in self.upload_patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.upload_patches:
            p.stop()
        super().tearDown()

    def _classified_as(self) -> dict:
        return next(s for s in self.statuses if s.get("stage") == "indexing")

    def test_classification_tokens_are_charged_to_the_uploader(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self._process()

        self.assertEqual(asyncio.run(self.memory.used(UPLOAD_USER, token_budget._today())), 77)
        self.assertEqual(self._classified_as()["document_type"], "invoice")

    def test_over_budget_upload_is_indexed_unclassified_without_a_groq_call(self) -> None:
        asyncio.run(self.memory.add(UPLOAD_USER, token_budget._today(), 100))

        with contextlib.redirect_stdout(io.StringIO()) as log:
            self._process()

        self.classify.assert_not_awaited()
        classified = self._classified_as()
        self.assertEqual(classified["document_type"], "other")
        self.assertEqual(classified["sensitivity_level"], "internal")
        self.assertEqual(classified["summary"], BUDGET_SKIPPED_SUMMARY)
        upload.index_document.assert_awaited_once()
        self.assertEqual(upload.index_document.await_args.args[4]["document_type"], "other")
        self.assertEqual(self.statuses[-1]["status"], "complete")
        self.assertEqual(asyncio.run(self.memory.used(UPLOAD_USER, token_budget._today())), 100)
        self.assertIn("skipping classification", log.getvalue())

    def test_the_skipped_classification_has_the_full_fallback_shape(self) -> None:
        skipped = budget_skipped_classification()
        self.assertEqual(set(skipped), set(classifier._FALLBACK))
        self.assertEqual(skipped["document_type"], "other")


if __name__ == "__main__":
    unittest.main()
