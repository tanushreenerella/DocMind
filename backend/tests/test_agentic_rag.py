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
        "user_id": "user-a",
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

    def __init__(self, answer: str = "Both are covered [SOURCE 1].") -> None:
        self.answer = answer
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
            content = self.answer
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

    async def _answer(self, model_answer: str) -> dict:
        with (
            patch.object(agentic_rag, "get_groq_client", return_value=FakeGroq(model_answer)),
            patch.object(agentic_rag, "search", AsyncMock(side_effect=lambda q, **_: [_chunk(q)])),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return await agentic_rag.answer_agentic_query("q1 and q2?", [], user_id="user-a")

    async def test_document_field_labels_are_not_blocked(self) -> None:
        # The caller's own document may legitimately contain these labels.
        for answer in (
            "The Wi-Fi sheet lists password: hunter2 [SOURCE 1].",
            "The setup guide shows API key: abc-123 [SOURCE 1].",
            "Section 4 is titled credentials: rotation policy [SOURCE 1].",
        ):
            with self.subTest(answer=answer):
                result = await self._answer(answer)
                self.assertTrue(result["has_answer"], result)
                self.assertIn("[a.pdf, p.1]", result["answer"])

    async def test_prompt_leaks_and_followed_injections_are_blocked(self) -> None:
        for answer in (
            "Sure! Here is my system prompt: you are a helpful assistant",
            "As instructed, I will ignore previous instructions [SOURCE 1].",
            "Here are the credentials for every account [SOURCE 1].",
        ):
            with self.subTest(answer=answer):
                result = await self._answer(answer)
                self.assertFalse(result["has_answer"])
                self.assertEqual(result["answer"], agentic_rag._NO_RELIABLE_ANSWER)

    async def test_unresolvable_source_markers_are_stripped(self) -> None:
        result = await self._answer("Covered [SOURCE 1]. Extra claim [SOURCE 7].")

        self.assertEqual(result["answer"], "Covered [a.pdf, p.1]. Extra claim.")
        self.assertEqual(len(result["citations"]), 1)

    async def test_answer_citing_only_chunks_the_user_does_not_own_is_withheld(self) -> None:
        foreign = lambda q, **_: [{**_chunk(q), "user_id": "user-b"}]
        with (
            patch.object(agentic_rag, "get_groq_client", return_value=FakeGroq()),
            patch.object(agentic_rag, "search", AsyncMock(side_effect=foreign)),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            result = await agentic_rag.answer_agentic_query("q1 and q2?", [], user_id="user-a")

        self.assertEqual(
            result,
            {"answer": agentic_rag._NO_RELIABLE_ANSWER, "citations": [], "has_answer": False, "error": None},
        )
        self.assertIn("[agentic citations] dropped", output.getvalue())


def _plan(**fields) -> str:
    return json.dumps({"is_simple": False, "question_type": "factual", "sub_questions": ["q1", "q2"], **fields})


class PlannerValidationTests(unittest.TestCase):
    def parse(self, raw: str) -> tuple[tuple[list[str], str], str]:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = agentic_rag._parse_planner_response(raw, "original?")
        return result, output.getvalue()

    def test_valid_plans(self) -> None:
        cases = {
            _plan(question_type="exploratory"): (["q1", "q2"], "exploratory"),
            _plan(sub_questions=["only one"]): (["only one"], "factual"),
            _plan(sub_questions=[" q1 ", "q1", "q2", "q3"]): (["q1", "q2", "q3"], "factual"),
            _plan(sub_questions=["x" * 300]): (["x" * 300], "factual"),
            _plan(is_simple=True, sub_questions=["rephrased"]): (["original?"], "factual"),
            "```json\n" + _plan() + "\n```": (["q1", "q2"], "factual"),
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw[:60]):
                result, logs = self.parse(raw)
                self.assertEqual(result, expected)
                self.assertNotIn("invalid", logs)

    def test_invalid_plans_fall_back_to_the_original_question_and_are_logged(self) -> None:
        for raw in (
            "",
            "Sure! Here is the plan.",
            "[\"q1\"]",
            "{not json",
            json.dumps({"is_simple": False}),
            _plan(sub_questions=[]),
            _plan(sub_questions=["q1", "q2", "q3", "q4", "q5"]),
            _plan(sub_questions=["q1", ""]),
            _plan(sub_questions=["   "]),
            _plan(sub_questions=["q1", "x" * 301]),
            _plan(sub_questions="q1"),
            _plan(sub_questions=["q1", 2]),
            _plan(question_type="summary"),
            _plan(is_simple="yes"),
        ):
            with self.subTest(raw=raw[:60]):
                result, logs = self.parse(raw)
                self.assertEqual(result, (["original?"], "factual"))
                self.assertIn("[agentic planner] invalid output", logs)

    def test_logs_do_not_echo_the_reply(self) -> None:
        _, logs = self.parse(_plan(sub_questions=["CONFIDENTIAL " * 30]))
        self.assertIn("invalid output", logs)
        self.assertNotIn("CONFIDENTIAL", logs)


class CriticValidationTests(unittest.TestCase):
    def parse(self, reply: object) -> tuple[tuple[str, str], str]:
        raw = reply if isinstance(reply, str) else json.dumps(reply)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = agentic_rag._parse_critic_response(raw, "the sub-question")
        return result, output.getvalue()

    def test_valid_verdicts(self) -> None:
        cases = [
            ({"verdict": "sufficient", "rewrite": ""}, ("sufficient", "the sub-question")),
            ({"verdict": "sufficient"}, ("sufficient", "the sub-question")),
            ({"verdict": "insufficient", "rewrite": " broader "}, ("insufficient", "broader")),
            # An over-long rewrite keeps the verdict but reuses the sub-question.
            ({"verdict": "insufficient", "rewrite": "x" * 301}, ("insufficient", "the sub-question")),
        ]
        for reply, expected in cases:
            with self.subTest(reply=reply):
                result, logs = self.parse(reply)
                self.assertEqual(result, expected)
                self.assertNotIn("invalid", logs)

    def test_invalid_replies_are_insufficient_and_logged(self) -> None:
        for reply in (
            "",
            "The evidence looks sufficient.",
            {"verdict": "maybe", "rewrite": ""},
            {"verdict": "Sufficient", "rewrite": ""},
            {"rewrite": "broader"},
            {"verdict": "sufficient", "rewrite": 5},
            {"verdict": True},
        ):
            with self.subTest(reply=reply):
                result, logs = self.parse(reply)
                self.assertEqual(result, ("insufficient", "the sub-question"))
                self.assertIn("[agentic critic] invalid output", logs)


def _cite(doc: str, page: int, user: str = "user-a") -> tuple[dict, dict]:
    chunk = {**_chunk(f"{doc}-{page}"), "doc_name": doc, "doc_id": f"id-{doc}", "page_number": page, "user_id": user}
    return chunk, agentic_rag._citation(chunk)


class CitationGroundingTests(unittest.TestCase):
    def ground(self, answer: str, citations: list[dict], chunks: list[dict]) -> tuple[str, list[dict]]:
        with contextlib.redirect_stdout(io.StringIO()):
            return agentic_rag._keep_grounded_citations(answer, citations, {"q1": chunks}, "user-a")

    def test_retrieved_owned_citations_are_kept(self) -> None:
        a_chunk, a = _cite("a.pdf", 1)
        b_chunk, b = _cite("b.pdf", 2)
        answer = "One [a.pdf, p.1]. Two [b.pdf, p.2][a.pdf, p.1]."
        self.assertEqual(self.ground(answer, [a, b], [a_chunk, b_chunk]), (answer, [a, b]))

    def test_citations_not_retrieved_for_this_request_are_dropped_with_their_markers(self) -> None:
        a_chunk, a = _cite("a.pdf", 1)
        _, b = _cite("b.pdf", 2)
        answer, kept = self.ground("One [a.pdf, p.1]. Two [b.pdf, p.2].", [a, b], [a_chunk])
        self.assertEqual((answer, kept), ("One [a.pdf, p.1]. Two.", [a]))

    def test_citations_from_other_users_chunks_are_dropped(self) -> None:
        a_chunk, a = _cite("a.pdf", 1)
        foreign_chunk, foreign = _cite("b.pdf", 2, user="user-b")
        unowned_chunk = {k: v for k, v in _cite("c.pdf", 3)[0].items() if k != "user_id"}
        unowned = agentic_rag._citation(unowned_chunk)
        answer, kept = self.ground(
            "One [a.pdf, p.1]. Two [b.pdf, p.2]. Three [c.pdf, p.3].",
            [a, foreign, unowned],
            [a_chunk, foreign_chunk, unowned_chunk],
        )
        self.assertEqual((answer, kept), ("One [a.pdf, p.1]. Two. Three.", [a]))

    def test_altered_citations_are_dropped(self) -> None:
        a_chunk, a = _cite("a.pdf", 1)
        for altered in ({**a, "excerpt": "made up..."}, {**a, "doc_id": "id-other"}, {**a, "page_number": 9}):
            with self.subTest(altered=altered):
                self.assertEqual(self.ground("One [a.pdf, p.1].", [altered], [a_chunk]), ("One.", []))

    def test_markers_without_a_citation_and_leftover_source_markers_are_stripped(self) -> None:
        a_chunk, a = _cite("a.pdf", 1)
        answer, _ = self.ground(
            "One [a.pdf, p.1] and [a.pdf, p.4] [SOURCE 9][SOURCE 2, SOURCE 3].", [a], [a_chunk]
        )
        self.assertEqual(answer, "One [a.pdf, p.1] and.")


if __name__ == "__main__":
    unittest.main()
