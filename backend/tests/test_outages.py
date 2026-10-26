"""Outages (Groq 429/5xx/timeouts, Chroma down, slow answers) are reported, never mistaken
for "no documents" or "insufficient evidence"."""

import asyncio
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import groq
import httpx
from fastapi.testclient import TestClient

from core import config
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from main import app
from services import agentic_rag, embedder, rag, timeline
from services.errors import AIUnavailable, SearchUnavailable
from tests import PROXY_HEADERS

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls, status: int, message: str):
    return cls(message, response=httpx.Response(status, request=_REQUEST), body=None)


GROQ_FAILURES = {
    "429": lambda: _status_error(groq.RateLimitError, 429, "Rate limit reached: tokens per minute (TPM)"),
    "503": lambda: _status_error(groq.InternalServerError, 503, "Service unavailable"),
    "timeout": lambda: groq.APITimeoutError(request=_REQUEST),
}


def _chunk(query: str) -> dict:
    return {
        "chunk_id": f"{query}-0", "chunk_text": f"evidence for {query}", "doc_name": "a.pdf",
        "page_number": 1, "image_filename": "", "relevance_score": 0.9, "doc_id": "doc-a",
    }


def _reply(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeGroq:
    """Answers each agentic prompt; ``fail`` maps a prompt name to the exception it raises."""

    PROMPTS = {
        agentic_rag._PLANNER_PROMPT: "planner",
        agentic_rag._CRITIC_PROMPT: "critic",
        agentic_rag._SYNTHESIZER_PROMPT: "synthesizer",
    }

    def __init__(self, fail: dict | None = None, verdict: str = "sufficient", sub_questions=("q1", "q2")) -> None:
        self.fail = fail or {}
        self.verdict = verdict
        self.sub_questions = list(sub_questions)
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, **_kwargs):
        name = self.PROMPTS.get(messages[0]["content"], "basic")
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]
        if name == "planner":
            return _reply(json.dumps({"is_simple": False, "question_type": "factual",
                                      "sub_questions": self.sub_questions}))
        if name == "critic":
            return _reply(json.dumps({"verdict": self.verdict, "rewrite": "broader query"}))
        return _reply("Covered [SOURCE 1].")


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_chroma_failure_is_search_unavailable_not_no_hits(self) -> None:
        with (
            patch.object(embedder, "search_or_raise", AsyncMock(side_effect=ConnectionError("chroma down"))),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SearchUnavailable):
                await embedder.search("q", user_id=USER)

    async def test_an_empty_index_is_still_no_hits(self) -> None:
        with patch.object(embedder.collection, "count", return_value=0):
            self.assertEqual(await embedder.search("q", user_id=USER), [])


class BasicRagTests(unittest.IsolatedAsyncioTestCase):
    async def answer(self, client, search) -> dict:
        with (
            patch.object(rag, "_client", client),
            patch.object(rag, "search", search),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return await rag.answer_query("What is covered?", [], user_id=USER)

    def failing_client(self, exc: Exception) -> MagicMock:
        client = MagicMock()
        client.chat.completions.create.side_effect = exc
        return client

    async def test_groq_failures_say_the_ai_service_is_busy(self) -> None:
        for name, failure in GROQ_FAILURES.items():
            with self.subTest(name):
                client = self.failing_client(failure())
                result = await self.answer(client, AsyncMock(return_value=[_chunk("q")]))

                self.assertEqual(result["error"], "ai_unavailable")
                self.assertFalse(result["has_answer"])
                self.assertEqual(result["answer"], AIUnavailable.message)
                self.assertEqual(client.chat.completions.create.call_count, 1)

    async def test_empty_model_reply_is_an_outage(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _reply("  ")
        result = await self.answer(client, AsyncMock(return_value=[_chunk("q")]))
        self.assertEqual(result["error"], "ai_unavailable")

    async def test_search_outage_is_not_reported_as_no_documents(self) -> None:
        client = MagicMock()
        result = await self.answer(client, AsyncMock(side_effect=SearchUnavailable()))

        self.assertEqual(result["error"], "search_unavailable")
        self.assertEqual(result["answer"], SearchUnavailable.message)
        self.assertNotIn("don't have any documents", result["answer"])
        client.chat.completions.create.assert_not_called()

    async def test_no_documents_and_answers_have_no_error(self) -> None:
        result = await self.answer(MagicMock(), AsyncMock(return_value=[]))
        self.assertIsNone(result["error"])
        self.assertIn("don't have any documents", result["answer"])

        client = MagicMock()
        client.chat.completions.create.return_value = _reply("Covered [SOURCE 1].")
        result = await self.answer(client, AsyncMock(return_value=[_chunk("q")]))
        self.assertEqual((result["error"], result["has_answer"]), (None, True))

    def test_groq_client_times_out_at_20s_and_never_retries(self) -> None:
        with patch.object(rag, "_client", None):
            client = rag.get_groq_client()
        self.assertEqual(client.timeout, 20.0)
        self.assertEqual(client.max_retries, 0)


class AgenticOutageTests(unittest.IsolatedAsyncioTestCase):
    async def answer(self, client: FakeGroq, search) -> dict:
        with (
            patch.object(agentic_rag, "get_groq_client", return_value=client),
            patch.object(agentic_rag, "search", search),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return await agentic_rag.answer_agentic_query("q1 and q2?", [], user_id=USER)

    async def test_critic_groq_failure_stops_without_retrying_retrieval(self) -> None:
        for name, failure in GROQ_FAILURES.items():
            with self.subTest(name):
                client = FakeGroq(fail={"critic": failure()}, sub_questions=["q1", "q2"])
                search = AsyncMock(side_effect=lambda q, **_: [_chunk(q)])

                result = await self.answer(client, search)

                self.assertEqual(result["error"], "ai_unavailable")
                self.assertEqual(result["answer"], AIUnavailable.message)
                self.assertNotIn("reliable answer", result["answer"])
                # One search per sub-question; no rewritten-query retries.
                self.assertEqual(sorted(c.args[0] for c in search.await_args_list), ["q1", "q2"])
                self.assertNotIn("synthesizer", client.calls)

    async def test_planner_groq_failure_stops_before_searching(self) -> None:
        client = FakeGroq(fail={"planner": GROQ_FAILURES["429"]()})
        search = AsyncMock(return_value=[_chunk("q")])

        result = await self.answer(client, search)

        self.assertEqual(result["error"], "ai_unavailable")
        search.assert_not_awaited()
        self.assertEqual(client.calls, ["planner"])

    async def test_synthesizer_groq_failure_is_flagged(self) -> None:
        result = await self.answer(
            FakeGroq(fail={"synthesizer": GROQ_FAILURES["503"]()}),
            AsyncMock(side_effect=lambda q, **_: [_chunk(q)]),
        )
        self.assertEqual((result["error"], result["has_answer"]), ("ai_unavailable", False))

    async def test_search_outage_stops_and_cancels_the_other_sub_questions(self) -> None:
        sibling_cancelled = asyncio.Event()

        async def search(query: str, **_kwargs):
            if query == "q1":
                raise SearchUnavailable()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        client = FakeGroq()
        result = await asyncio.wait_for(self.answer(client, search), timeout=5)

        self.assertEqual(result["error"], "search_unavailable")
        self.assertEqual(result["answer"], SearchUnavailable.message)
        self.assertTrue(sibling_cancelled.is_set())
        self.assertNotIn("critic", client.calls)

    async def test_a_real_insufficient_verdict_still_means_no_reliable_answer(self) -> None:
        client = FakeGroq(verdict="insufficient", sub_questions=["q1", "q2"])
        search = AsyncMock(side_effect=lambda q, **_: [_chunk(q)])

        result = await self.answer(client, search)

        self.assertIsNone(result["error"])
        self.assertFalse(result["has_answer"])
        self.assertIn("couldn't find a reliable answer", result["answer"])
        # Each sub-question: the first search plus _MAX_RETRIEVAL_RETRIES rewrites.
        self.assertEqual(search.await_count, 2 * (1 + agentic_rag._MAX_RETRIEVAL_RETRIES))

    async def test_answers_have_no_error(self) -> None:
        result = await self.answer(FakeGroq(), AsyncMock(side_effect=lambda q, **_: [_chunk(q)]))
        self.assertEqual((result["error"], result["has_answer"]), (None, True))

    async def test_overall_deadline(self) -> None:
        async def slow_search(query: str, **_kwargs):
            await asyncio.sleep(30)

        with patch.object(agentic_rag, "AGENTIC_DEADLINE_S", 0.05):
            result = await asyncio.wait_for(self.answer(FakeGroq(), slow_search), timeout=5)

        self.assertEqual((result["error"], result["has_answer"]), ("timeout", False))
        self.assertIn("try again shortly", result["answer"])

    async def test_graph_runs_with_an_explicit_recursion_limit(self) -> None:
        state = {"final_answer": "a", "citations": [], "has_answer": True}
        with patch.object(agentic_rag, "agentic_graph") as graph:
            graph.ainvoke = AsyncMock(return_value=state)
            await agentic_rag.answer_agentic_query("q", [], user_id=USER)

        self.assertEqual(graph.ainvoke.await_args.kwargs["config"], {"recursion_limit": 6})

    async def test_recursion_limit_is_enforced_and_leaves_room_for_the_graph(self) -> None:
        from langgraph.errors import GraphRecursionError

        def run():
            return self.answer(FakeGroq(), AsyncMock(side_effect=lambda q, **_: [_chunk(q)]))

        # The 3-node graph needs a limit of 4; RUN_CONFIG's 6 leaves headroom, nothing more.
        with patch.object(agentic_rag, "RUN_CONFIG", {"recursion_limit": 3}):
            with self.assertRaises(GraphRecursionError):
                await run()
        self.assertTrue((await run())["has_answer"])


class TimelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_outage_is_not_reported_as_missing_evidence(self) -> None:
        with patch.object(timeline, "search", AsyncMock(side_effect=SearchUnavailable())):
            result = await timeline.build_evidence_timeline("doc-a", user_id=USER)

        self.assertEqual(result["error"], "search_unavailable")
        self.assertEqual(result["briefing"], SearchUnavailable.message)
        self.assertFalse(result["has_timeline"])


class FakePool:
    async def fetchrow(self, query: str, *args: object):
        return {"id": args[0], "email": f"{args[0]}@example.com", "full_name": ""}

    def __init__(self) -> None:
        self.execute = AsyncMock()
        self.fetchval = AsyncMock(return_value=1)


class ChatRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        self.pool = FakePool()
        self.patches = [
            patch("core.auth.get_pool", return_value=self.pool),
            patch("routers.chat.get_pool", return_value=self.pool),
        ]
        for p in self.patches:
            p.start()
        token = create_access_token(USER, f"{USER}@example.com")
        self.client = TestClient(app, headers={
            **PROXY_HEADERS, "Origin": config.ALLOWED_ORIGINS[0],
            "Cookie": f"{ACCESS_TOKEN_COOKIE}={token}",
        })

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        limiter.reset()

    def test_outage_notices_are_returned_but_not_saved_to_history(self) -> None:
        notice = {"answer": AIUnavailable.message, "citations": [], "has_answer": False, "error": "ai_unavailable"}
        for path, target in (("/api/chat", "routers.chat.answer_query"),
                             ("/api/chat/agentic", "routers.chat.answer_agentic_query")):
            with self.subTest(path), patch(target, AsyncMock(return_value=notice)):
                response = self.client.post(path, json={"question": "What is covered?"})

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), notice)
                self.pool.execute.assert_not_awaited()

    def test_answers_are_saved_to_history(self) -> None:
        answer = {"answer": "ok", "citations": [], "has_answer": True, "error": None}
        with patch("routers.chat.answer_query", AsyncMock(return_value=answer)):
            response = self.client.post("/api/chat", json={"question": "What is covered?"})

        self.assertEqual(response.json(), answer)
        self.assertEqual(self.pool.execute.await_count, 2)


if __name__ == "__main__":
    unittest.main()
