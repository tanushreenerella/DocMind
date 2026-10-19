"""Agentic graph: concurrent per-sub-question research, retries, node timing logs."""

import asyncio
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services import agentic_rag


def _chunk(query: str) -> dict:
    return {
        "chunk_id": f"{query}-0",
        "chunk_text": f"evidence for {query}",
        "doc_name": "a.pdf",
        "page_number": 1,
        "image_filename": "",
        "relevance_score": 0.9,
        "doc_id": "doc-a",
    }


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_sub_questions_are_researched_concurrently(self) -> None:
        # Each search waits until both have started; run sequentially, this
        # would never get past the first search and the wait_for times out.
        both_started = asyncio.Barrier(2)

        async def search(query: str, **_kwargs) -> list[dict]:
            await both_started.wait()
            return [_chunk(query)]

        grade = AsyncMock(side_effect=lambda q, chunks, qt: (q, "sufficient", ""))
        with (
            patch.object(agentic_rag, "search", side_effect=search),
            patch.object(agentic_rag, "_grade_sub_question", grade),
        ):
            result = await asyncio.wait_for(
                agentic_rag._research({
                    "original_question": "both?",
                    "user_id": "user-a",
                    "sub_questions": ["q1", "q2"],
                }),
                timeout=2,
            )

        self.assertEqual(result["critic_verdicts"], {"q1": "sufficient", "q2": "sufficient"})
        self.assertEqual(grade.await_count, 2)

    async def test_insufficient_retries_with_rewrite_then_gives_up(self) -> None:
        search = AsyncMock(side_effect=lambda query, **_: [_chunk(query)])
        rewrites = iter(["broader 1", "broader 2", "broader 3"])
        grade = AsyncMock(side_effect=lambda q, chunks, qt: (q, "insufficient", next(rewrites)))
        with (
            patch.object(agentic_rag, "search", search),
            patch.object(agentic_rag, "_grade_sub_question", grade),
        ):
            chunks, verdict, retries, query = await agentic_rag._research_sub_question(
                "q1", "factual", "user-a", "doc-a"
            )

        self.assertEqual(
            [c.args[0] for c in search.await_args_list],
            ["q1", "broader 1", "broader 2"],
        )
        self.assertEqual(search.await_args_list[0].kwargs, {"n_results": 6, "doc_id": "doc-a", "user_id": "user-a"})
        self.assertEqual(verdict, agentic_rag._NO_RELIABLE_VERDICT)
        self.assertEqual(retries, agentic_rag._MAX_RETRIEVAL_RETRIES)
        self.assertEqual(query, "broader 2")
        self.assertEqual(chunks, [_chunk("broader 2")])

    async def test_stops_retrying_once_sufficient(self) -> None:
        search = AsyncMock(side_effect=lambda query, **_: [_chunk(query)])
        verdicts = iter([("insufficient", "broader"), ("sufficient", "")])
        grade = AsyncMock(side_effect=lambda q, chunks, qt: (q, *next(verdicts)))
        with (
            patch.object(agentic_rag, "search", search),
            patch.object(agentic_rag, "_grade_sub_question", grade),
        ):
            chunks, verdict, retries, query = await agentic_rag._research_sub_question(
                "q1", "factual", "user-a", None
            )

        self.assertEqual((verdict, retries, query), ("sufficient", 1, "broader"))
        self.assertEqual(search.await_count, 2)


class FakeGroq:
    """Answers planner, critic and synthesizer calls by their system prompt."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, **_kwargs):
        system = messages[0]["content"]
        if system == agentic_rag._PLANNER_PROMPT:
            content = json.dumps({
                "is_simple": False,
                "question_type": "factual",
                "sub_questions": ["q1", "q2"],
            })
        elif system == agentic_rag._CRITIC_PROMPT:
            content = json.dumps({"verdict": "sufficient", "rewrite": ""})
        else:
            content = "Both are covered [SOURCE 1]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_answers_logs_node_durations_and_no_raw_output(self) -> None:
        output = io.StringIO()
        with (
            patch.object(agentic_rag, "get_groq_client", return_value=FakeGroq()),
            patch.object(agentic_rag, "search", AsyncMock(side_effect=lambda q, **_: [_chunk(q)])),
            contextlib.redirect_stdout(output),
        ):
            result = await agentic_rag.answer_agentic_query("q1 and q2?", [], user_id="user-a")

        self.assertTrue(result["has_answer"])
        self.assertEqual(result["answer"], "Both are covered [a.pdf, p.1].")
        logs = output.getvalue()
        for node in ("planner", "research", "synthesizer"):
            self.assertRegex(logs, rf"\[agentic timing\] {node}: \d+ ms")
        self.assertNotIn("raw]", logs)


if __name__ == "__main__":
    unittest.main()
