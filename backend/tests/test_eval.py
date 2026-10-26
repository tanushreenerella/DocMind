"""eval/: metrics, question drafting helpers, and the runner's retrieval and instrumentation."""

import argparse
import asyncio
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import groq
import httpx

from eval import generate_questions as gen
from eval import run_eval
from eval.corpus import IndexedDoc
from eval.metrics import (
    QuestionError,
    decline_reason,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    validate_questions,
)
from services import agentic_rag, embedder, rag

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def hit(page: int, doc: str = "a.pdf", chunk: str | None = None, score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk or f"{doc}_p{page}_c0", "chunk_text": f"text of {doc} p{page}",
        "doc_name": doc, "page_number": page, "image_filename": "", "relevance_score": score,
        "doc_id": f"id-{doc}",
    }


def question(**overrides) -> dict:
    return {
        "question": "What is the total?", "doc_name": "a.pdf", "expected_pages": [2],
        "reference_answer": "$5", "type": "factual", **overrides,
    }


def answer_run(latency: float = 1000.0, faithfulness=None, **overrides) -> dict:
    """One pipeline run as _run_pipeline records it."""
    return {
        "answer": "The total is $5.", "has_answer": True, "declined": False, "decline_reason": None,
        "latency_ms": latency, "latency_excl_rate_limit_ms": latency, "retrieval_ms": 200.0,
        "rate_limit_retries": 0, "rate_limit_wait_ms": 0.0, "llm_calls": 1, "prompt_tokens": 100,
        "completion_tokens": 10, "search_calls": 1, "errors": [], "contexts": [["a.pdf", 2]],
        "context_texts": ["text of a.pdf p2"], "faithfulness": faithfulness, "answer_relevancy": faithfulness,
        **overrides,
    }


def answered_record(number: int, recall: float = 1.0, **run) -> dict:
    return {
        "number": number, **question(question=f"Q{number}"),
        "retrieval": {"vector": {"recall@5": recall, "mrr@10": recall, "ndcg@10": recall,
                                 "latency_ms": 10.0 * number, "retrieved": [], "error": None}},
        "answers": {"basic": answer_run(**run)},
    }


def saved_run(path: Path, records: list[dict], offset: int = 0, stopped=None, judge="judge-x", **config) -> Path:
    """A results file as main() writes it."""
    cfg = {
        "questions_file": "q.json", "questions_sha256": "abc", "questions": len(records),
        "questions_by_type": {"factual": len(records)}, "offset": offset, "limit": None,
        "retrieval_only": False, "scope": "all", "pacing": {}, "user_id": USER, "chroma_database": "db",
        "retrieval": {"modes": ["vector"], "recall_k": 5, "rank_k": 10},
        "answers": {"pipelines": ["basic"], "retrieval_mode": "hybrid", "answer_models": ["m"],
                    "judge_model": judge, "ragas_metrics": ["faithfulness", "answer_relevancy"],
                    "answer_relevancy_strictness": 1},
        "versions": {"ragas": "0.4.3"},
        **config,
    }
    meta = {"timestamp": f"2026-10-2{offset % 10}T00:00:00+00:00", "duration_s": 10.0,
            "git": {"commit": "abc", "dirty": False}, "completed_questions": len(records), "stopped": stopped}
    path.write_text(json.dumps({"meta": meta, "config": cfg, "retrieval": {}, "answers": {},
                                "per_question": records}), encoding="utf-8")
    return path


class RankingMetricTests(unittest.TestCase):
    def test_recall_counts_expected_pages_not_chunks(self) -> None:
        hits = [hit(2, chunk="c0"), hit(2, chunk="c1"), hit(9), hit(9, "b.pdf"), hit(4), hit(3)]
        self.assertEqual(recall_at_k(hits, "a.pdf", [2, 3], 5), 0.5)  # p3 is 6th
        self.assertEqual(recall_at_k(hits, "a.pdf", [2, 3], 6), 1.0)

    def test_same_page_of_another_document_is_not_relevant(self) -> None:
        hits = [hit(2, "b.pdf"), hit(2, "a.pdf")]
        self.assertEqual(mrr_at_k(hits, "a.pdf", [2], 10), 0.5)

    def test_mrr_is_zero_without_a_hit_in_k(self) -> None:
        self.assertEqual(mrr_at_k([hit(1)] * 3 + [hit(2)], "a.pdf", [2], 3), 0.0)

    def test_ndcg_perfect_and_partial(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([hit(2), hit(3)], "a.pdf", [2, 3], 10), 1.0)
        # Second chunk from an already-found page gains nothing.
        partial = ndcg_at_k([hit(2, chunk="c0"), hit(2, chunk="c1"), hit(3)], "a.pdf", [2, 3], 10)
        self.assertAlmostEqual(partial, (1 + 1 / 2) / (1 + 1 / 1.5849625007211562))
        self.assertEqual(ndcg_at_k([], "a.pdf", [2], 10), 0.0)

    def test_percentile_interpolates(self) -> None:
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([5], 95), 5)
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile(list(range(1, 101)), 95), 95.05)


class DeclineTests(unittest.TestCase):
    def test_declines(self) -> None:
        for answer in (
            "The provided documents don't contain information about the CEO's salary.",
            "The salary is not mentioned in the sources.",
            "There is no information about that in the documents.",
            "I couldn't find information relevant enough to answer this question confidently.",
        ):
            with self.subTest(answer=answer):
                self.assertIsNotNone(decline_reason(answer, True))
        self.assertEqual(decline_reason("anything", False), "has_answer=false")

    def test_answers(self) -> None:
        for answer in (
            "The invoice total is $5,000 [a.pdf, p.2].",
            "Multi-tenancy means customers share the same physical resources [cloud.pdf, p.1].",
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(decline_reason(answer, True))


class QuestionValidationTests(unittest.TestCase):
    def test_valid_questions_pass(self) -> None:
        items = [question(), question(type="unanswerable", expected_pages=[]), question(type="multi_part", expected_pages=[1, 3])]
        self.assertEqual(validate_questions(items), items)

    def test_every_problem_is_reported(self) -> None:
        with self.assertRaises(QuestionError) as ctx:
            validate_questions([
                question(type="trivia"),
                question(expected_pages=[]),
                question(type="unanswerable"),
                question(expected_pages=[0, "2"]),
                {"question": "q"},
            ])
        message = str(ctx.exception)
        for fragment in ("question 1: type", "question 2: factual questions need", "question 3: unanswerable",
                         "question 4: expected_pages", "question 5: missing"):
            self.assertIn(fragment, message)


class DraftingTests(unittest.TestCase):
    def doc(self, pages: dict[int, list[str]]) -> IndexedDoc:
        doc = IndexedDoc("a.pdf", "id-a")
        for page, chunks in pages.items():
            doc.chunks[page] = list(enumerate(chunks))
        return doc

    def test_page_text_drops_the_chunk_overlap(self) -> None:
        overlap = embedder.CHUNK_OVERLAP
        first = "A" * 450 + "B" * overlap
        second = "B" * overlap + "C" * 10
        self.assertEqual(self.doc({1: [first, second]}).page_text(1), "A" * 450 + "B" * overlap + "C" * 10)

    def test_type_counts(self) -> None:
        self.assertEqual(gen._type_counts(10, 5), {"factual": 6, "multi_part": 2, "unanswerable": 2})
        # One page: nothing to combine across pages.
        self.assertEqual(gen._type_counts(10, 1), {"factual": 8, "multi_part": 0, "unanswerable": 2})

    def test_long_documents_are_sampled_across_pages_within_budget(self) -> None:
        doc = self.doc({p: ["x" * 1000] for p in range(1, 21)})
        text, pages = gen._document_text(doc, 5000)
        self.assertLessEqual(len(text), 5000)
        self.assertGreater(len(pages), 1)
        self.assertGreater(pages[-1], 10)  # not just the first pages

    def test_clean_fixes_pages_and_drops_bad_items(self) -> None:
        doc = self.doc({1: ["x"], 2: ["y"]})
        raw = [
            {"question": "Q1", "expected_pages": [1, 7], "reference_answer": "A", "type": "factual"},
            {"question": "Q2", "expected_pages": [9], "reference_answer": "A", "type": "factual"},
            {"question": "Q3", "expected_pages": [2], "reference_answer": "Not in it", "type": "unanswerable"},
            {"question": "Q4", "expected_pages": [1], "reference_answer": "A", "type": "trivia"},
            "junk",
        ]
        kept, warnings = gen._clean(raw, doc, [1, 2])
        self.assertEqual(kept, [
            {"question": "Q1", "doc_name": "a.pdf", "expected_pages": [1], "reference_answer": "A", "type": "factual"},
            {"question": "Q3", "doc_name": "a.pdf", "expected_pages": [], "reference_answer": "Not in it", "type": "unanswerable"},
        ])
        validate_questions(kept)
        # Q1 pages trimmed; Q2 trimmed then dropped; Q4 and "junk" dropped.
        self.assertEqual(len(warnings), 5)


class RetrievalModeTests(unittest.TestCase):
    def run_modes(self, vector=None, bm25=None, rerank=None):
        vector_mock = MagicMock(return_value=vector if vector is not None else [hit(9), hit(2), hit(3)])
        if isinstance(vector, Exception):
            vector_mock = MagicMock(side_effect=vector)
        rerank_mock = AsyncMock(side_effect=rerank or (lambda q, hits, n: list(reversed(hits))[:n]))
        with (
            patch.object(embedder, "get_embedding", AsyncMock(return_value=[0.1])),
            patch.object(embedder, "_vector_search", vector_mock),
            patch.object(embedder, "_search_bm25", MagicMock(return_value=bm25 if bm25 is not None else [hit(3), hit(5)])),
            patch.object(embedder, "_rerank_hits", rerank_mock),
        ):
            return asyncio.run(run_eval._retrieve_all_modes(
                question(expected_pages=[2, 3]), run_eval.RETRIEVAL_MODES, USER, None, total_chunks=100,
            ))

    def test_each_mode_is_scored_on_its_own_ranking(self) -> None:
        results = self.run_modes()

        self.assertEqual(results["vector"]["retrieved"][:3], [["a.pdf", 9], ["a.pdf", 2], ["a.pdf", 3]])
        self.assertEqual(results["vector"]["mrr@10"], 0.5)
        self.assertEqual(results["bm25"]["mrr@10"], 1.0)
        self.assertEqual(results["bm25"]["recall@5"], 0.5)
        # RRF: page 3 is in both rankings, so it is fused to the top.
        self.assertEqual(results["hybrid"]["retrieved"][0], ["a.pdf", 3])
        self.assertEqual(results["hybrid"]["recall@5"], 1.0)
        for mode in run_eval.RETRIEVAL_MODES:
            self.assertIsNone(results[mode]["error"])
            self.assertIsNotNone(results[mode]["latency_ms"])

    def test_vector_failure_fails_vector_based_modes_only(self) -> None:
        results = self.run_modes(vector=ConnectionError("chroma down"))

        for mode in ("vector", "hybrid", "hybrid_rerank"):
            self.assertIn("chroma down", results[mode]["error"])
            self.assertEqual(results[mode]["recall@5"], 0.0)
            self.assertIsNone(results[mode]["latency_ms"])
        self.assertIsNone(results["bm25"]["error"])
        self.assertEqual(results["bm25"]["mrr@10"], 1.0)


TPD_MESSAGE = (
    "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` "
    "on tokens per day (TPD): Limit 500000, Used 499812, Requested 1310. Please try again in 7m12.5s."
)
TPM_MESSAGE = (
    "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` "
    "on tokens per minute (TPM): Limit 6000, Used 5800, Requested 900. Please try again in 7s."
)


class FakeGroq:
    def __init__(self, content: str = "The total is $5 [SOURCE 1].", fail: bool = False) -> None:
        def create(**kwargs):
            if fail:
                raise RuntimeError("groq 503")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.max_retries = 2
        self._sleep_for_retry = lambda **kwargs: None


class InstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        # _install_instrumentation rebinds these; restore them afterwards.
        self.patches = [
            patch.object(rag, "search", rag.search),
            patch.object(agentic_rag, "search", agentic_rag.search),
            patch.object(rag, "RERANK_ENABLED", False),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def run_basic(self, client: FakeGroq, search: AsyncMock) -> dict:
        with (
            patch.object(rag, "_client", client),
            patch.object(embedder, "search_or_raise", search),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_eval._install_instrumentation()
            return asyncio.run(run_eval._run_pipeline("basic", question(), USER, None))

    def test_basic_run_records_calls_tokens_retrieval_and_contexts(self) -> None:
        result = self.run_basic(FakeGroq(), AsyncMock(return_value=[hit(2), hit(4, score=0.05)]))

        self.assertEqual(result["answer"], "The total is $5 [a.pdf, p.2].")
        self.assertFalse(result["declined"])
        self.assertEqual((result["llm_calls"], result["prompt_tokens"], result["completion_tokens"]), (1, 120, 30))
        self.assertEqual(result["search_calls"], 1)
        self.assertEqual(result["errors"], [])
        # The low-relevance hit was cut before the model saw it, so it isn't a context.
        self.assertEqual(result["contexts"], [["a.pdf", 2]])
        self.assertIn("qwen/qwen3.8-27b", run_eval._answer_models)

    def test_outages_are_failed_runs_not_declines(self) -> None:
        result = self.run_basic(FakeGroq(fail=True), AsyncMock(return_value=[hit(2)]))
        self.assertEqual(result["llm_calls"], 1)
        self.assertIn("llm: RuntimeError: groq 503", result["errors"][0])
        self.assertIn("outage notice instead of an answer: ai_unavailable", result["errors"][-1])
        self.assertEqual((result["answer"], result["declined"]), (None, False))

        result = self.run_basic(FakeGroq(), AsyncMock(side_effect=ConnectionError("chroma down")))
        self.assertIn("search: ConnectionError: chroma down", result["errors"][0])
        self.assertIn("search_unavailable", result["errors"][-1])
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual((result["answer"], result["declined"]), (None, False))

    def groq_with_responses(self, *responses: tuple) -> groq.Groq:
        """A real Groq client whose HTTP calls get ``responses`` (status, headers[, message]) in order."""
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            status, headers, *message = queue.pop(0)
            if status != 200:
                return httpx.Response(status, headers=headers, json={"error": {
                    "message": message[0] if message else "slow down",
                    "type": "tokens", "code": "rate_limit_exceeded",
                }})
            return httpx.Response(200, json={
                "id": "c1", "object": "chat.completion", "created": 0, "model": "qwen/qwen3.8-27b",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "The total is $5 [SOURCE 1]."}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
            })

        return groq.Groq(
            api_key="test", max_retries=2,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def test_rate_limit_retries_and_wait_are_recorded(self) -> None:
        # The SDK honours retry-after-ms, so the waits are short but real.
        client = self.groq_with_responses((429, {"retry-after-ms": "60"}), (429, {"retry-after-ms": "60"}), (200, {}))

        result = self.run_basic(client, AsyncMock(return_value=[hit(2)]))

        self.assertEqual(result["answer"], "The total is $5 [a.pdf, p.2].")
        self.assertEqual(result["rate_limit_retries"], 2)
        self.assertEqual(result["llm_calls"], 1)  # retries happen inside one create() call
        self.assertGreaterEqual(result["rate_limit_wait_ms"], 100)
        self.assertAlmostEqual(
            result["latency_excl_rate_limit_ms"], result["latency_ms"] - result["rate_limit_wait_ms"], delta=0.2
        )
        self.assertEqual(result["errors"], [])

    def test_other_retries_are_not_rate_limits(self) -> None:
        client = self.groq_with_responses((503, {"retry-after-ms": "10"}), (200, {}))

        result = self.run_basic(client, AsyncMock(return_value=[hit(2)]))

        self.assertEqual((result["rate_limit_retries"], result["rate_limit_wait_ms"]), (0, 0.0))
        self.assertEqual(result["latency_excl_rate_limit_ms"], result["latency_ms"])

    def test_rate_limit_that_never_clears_is_an_error(self) -> None:
        client = self.groq_with_responses(*[(429, {"retry-after-ms": "10"})] * 3)

        result = self.run_basic(client, AsyncMock(return_value=[hit(2)]))

        self.assertEqual(result["rate_limit_retries"], 2)
        self.assertIn("llm: RateLimitError", result["errors"][0])

    def test_per_minute_limit_is_an_ordinary_error(self) -> None:
        client = self.groq_with_responses(*[(429, {"retry-after-ms": "10"}, TPM_MESSAGE)] * 3)

        result = self.run_basic(client, AsyncMock(return_value=[hit(2)]))

        self.assertIn("tokens per minute", result["errors"][0])
        self.assertIsNone(result["answer"])  # a failed run, not a decline

    def test_daily_limit_stops_the_basic_pipeline_instead_of_failing_the_answer(self) -> None:
        client = self.groq_with_responses(*[(429, {"retry-after-ms": "10"}, TPD_MESSAGE)] * 3)

        with self.assertRaises(run_eval.DailyLimitReached) as ctx:
            self.run_basic(client, AsyncMock(return_value=[hit(2)]))

        self.assertEqual(ctx.exception.retry_in, "7m12.5s")
        self.assertIn("tokens per day (TPD)", ctx.exception.detail)
        self.assertIsNone(run_eval._current_run.get())  # the run's context was still reset

    def test_daily_limit_gets_out_of_the_agentic_graph(self) -> None:
        # LangGraph nodes call Groq from worker threads, and the critic/planner catch
        # Exception; the daily limit must still reach the run loop.
        limit = groq.RateLimitError(
            TPD_MESSAGE,
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None,
        )
        client = FakeGroq()
        client.chat.completions.create = MagicMock(side_effect=limit)

        with (
            patch.object(rag, "_client", client),
            patch.object(embedder, "search_or_raise", AsyncMock(return_value=[hit(2)])),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_eval._install_instrumentation()
            with self.assertRaises(run_eval.DailyLimitReached):
                asyncio.run(run_eval._run_pipeline("agentic", question(), USER, None))

    def test_calls_outside_a_run_are_not_counted(self) -> None:
        with patch.object(rag, "_client", FakeGroq()):
            run_eval._install_instrumentation()
            rag.get_groq_client().chat.completions.create(model="m", messages=[])  # e.g. a judge call
        self.assertIsNone(run_eval._current_run.get())


def rate_limit_error(message: str) -> groq.RateLimitError:
    return groq.RateLimitError(
        message, response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")), body=None
    )


class StopAndResumeTests(unittest.TestCase):
    def collect(self, offset: int, outcomes: list) -> tuple[list[dict], dict | None]:
        args = argparse.Namespace(
            user_id=USER, scope="all", modes=(), pipelines=("basic",), delay=0, sleep=0, offset=offset,
        )
        doc = IndexedDoc("a.pdf", "id-a")
        doc.chunks[2] = [(0, "text")]
        questions = [question(question=f"Q{n}") for n in range(len(outcomes))]
        with (
            patch.object(run_eval, "load_user_docs", return_value={"a.pdf": doc}),
            patch.object(run_eval, "_run_pipeline", AsyncMock(side_effect=outcomes)),
            patch.object(embedder.collection, "count", return_value=10),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return asyncio.run(run_eval._collect(args, questions))

    def test_daily_limit_keeps_finished_questions_and_drops_the_current_one(self) -> None:
        limit = run_eval.DailyLimitReached(rate_limit_error(TPD_MESSAGE))
        records, stopped = self.collect(offset=10, outcomes=[{"answer": "a"}, {"answer": "b"}, limit, {"answer": "d"}])

        self.assertEqual([r["number"] for r in records], [11, 12])
        self.assertEqual(stopped["stage"], "answers")
        self.assertEqual((stopped["at_question"], stopped["completed"], stopped["resume_offset"]), (13, 2, 12))
        self.assertEqual(stopped["retry_in"], "7m12.5s")
        self.assertIn("resume with --offset 12", run_eval._stop_summary(stopped))

    def test_complete_run_has_no_stop(self) -> None:
        records, stopped = self.collect(offset=0, outcomes=[{"answer": "a"}])
        self.assertEqual(([r["number"] for r in records], stopped), ([1], None))


class JudgeTests(unittest.TestCase):
    def test_daily_limit_on_the_judge_stops_judging(self) -> None:
        judge = SimpleNamespace(
            client=SimpleNamespace(create=MagicMock(side_effect=rate_limit_error(TPD_MESSAGE))),
            async_client=SimpleNamespace(create=AsyncMock(side_effect=rate_limit_error(TPD_MESSAGE))),
        )
        run_eval._stop_judge_on_daily_limit(judge)

        with self.assertRaises(run_eval.DailyLimitReached):
            judge.client.create(model="m", messages=[])
        with self.assertRaises(run_eval.DailyLimitReached):
            asyncio.run(judge.async_client.create(model="m", messages=[]))

    def test_other_judge_rate_limits_are_left_to_ragas(self) -> None:
        judge = SimpleNamespace(
            client=SimpleNamespace(create=MagicMock(side_effect=rate_limit_error(TPM_MESSAGE))),
            async_client=SimpleNamespace(create=AsyncMock()),
        )
        run_eval._stop_judge_on_daily_limit(judge)

        with self.assertRaises(groq.RateLimitError):
            judge.client.create(model="m", messages=[])

    @unittest.skipUnless(importlib.util.find_spec("ragas"), "requirements-eval.txt not installed")
    def test_judge_model_and_single_generation_answer_relevancy(self) -> None:
        evaluation = MagicMock()
        evaluation.to_pandas.return_value.to_dict.return_value = [
            {"faithfulness": 0.8, "answer_relevancy": float("nan")}
        ]
        with (
            patch("langchain_groq.ChatGroq") as chat_groq,
            patch("ragas.evaluate", return_value=evaluation) as evaluate,
        ):
            scores = run_eval._ragas_score([{"user_input": "q", "response": "a"}], "judge-x", 1, 60)

        self.assertEqual(chat_groq.call_args.kwargs["model"], "judge-x")
        relevancy = evaluate.call_args.kwargs["metrics"][1]
        self.assertEqual((relevancy.name, relevancy.strictness), ("answer_relevancy", 1))
        self.assertEqual(scores, [{"faithfulness": 0.8, "answer_relevancy": None}])


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.questions = Path(self.dir) / "questions.json"
        self.questions.write_text(json.dumps([question(question=f"Q{n}") for n in range(1, 5)]))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def main(self, argv: list[str], collected: tuple[list[dict], dict | None]):
        seen = {}

        async def collect(args, questions):
            seen.update(args=args, questions=questions)
            return collected

        score = MagicMock()
        with (
            patch.object(sys, "argv", ["run_eval", "--user-id", USER, "--questions", str(self.questions),
                                       "--out-dir", self.dir, *argv]),
            patch.object(run_eval, "_collect", collect),
            patch.object(run_eval, "_install_instrumentation"),
            patch.object(run_eval, "_git_info", return_value={"commit": "abc", "dirty": False}),
            patch.object(run_eval, "_score_answers", score),
            patch.object(rag, "_client", FakeGroq()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                run_eval.main()
                exit_code = 0
            except SystemExit as exc:
                exit_code = exc.code
        return seen, score, exit_code

    def test_retrieval_only_with_offset_writes_its_own_files(self) -> None:
        records = [
            {**question(question=f"Q{n}"), "number": n, "answers": {},
             "retrieval": {"vector": {"recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0, "latency_ms": 5.0, "error": None}}}
            for n in (3, 4)
        ]
        seen, score, exit_code = self.main(["--retrieval-only", "--offset", "2", "--modes", "vector"], (records, None))

        self.assertEqual(exit_code, 0)
        self.assertEqual([q["question"] for q in seen["questions"]], ["Q3", "Q4"])
        self.assertEqual((seen["args"].pipelines, seen["args"].skip_ragas), ((), True))
        score.assert_not_called()
        results = json.loads((Path(self.dir) / "results.offset2.json").read_text())
        self.assertEqual((results["config"]["offset"], results["config"]["retrieval_only"]), (2, True))
        self.assertEqual(results["answers"], {})
        self.assertIn("questions 3–4 of the file", (Path(self.dir) / "results.offset2.md").read_text(encoding="utf-8"))
        self.assertFalse((Path(self.dir) / "results.json").exists())

    STOPPED = {"stage": "answers", "reason": "groq_daily_limit", "at_question": 3, "completed": 1,
               "resume_offset": 2, "retry_in": "7m12.5s", "detail": TPD_MESSAGE}

    def test_stopped_run_still_judges_finished_answers_and_exits_3(self) -> None:
        records = [answered_record(2)]
        seen, score, exit_code = self.main(["--offset", "1", "--judge-model", "judge-x"], (records, self.STOPPED))

        self.assertEqual(exit_code, 3)
        score.assert_called_once()
        self.assertEqual(score.call_args.args[0], records)
        self.assertEqual(score.call_args.args[2], "judge-x")
        results = json.loads((Path(self.dir) / "results.offset1.json").read_text(encoding="utf-8"))
        self.assertEqual(results["meta"]["stopped"], self.STOPPED)
        self.assertIsNone(results["meta"]["judging_stopped"])
        self.assertEqual(results["meta"]["completed_questions"], 1)
        self.assertEqual(results["config"]["answers"]["judge_model"], "judge-x")
        self.assertEqual(results["config"]["answers"]["answer_relevancy_strictness"], 1)
        self.assertEqual(results["per_question"][0]["answers"]["basic"]["context_texts"], ["text of a.pdf p2"])
        markdown = (Path(self.dir) / "results.offset1.md").read_text(encoding="utf-8")
        self.assertIn("Partial run", markdown)
        self.assertIn("resume with --offset 2", markdown)

    def test_judge_that_is_the_exhausted_answer_model_is_not_tried(self) -> None:
        with patch.object(run_eval, "_answer_models", {"judge-x"}):
            _, score, exit_code = self.main(["--judge-model", "judge-x"], ([answered_record(1)], self.STOPPED))

        self.assertEqual(exit_code, 3)
        score.assert_not_called()
        results = json.loads((Path(self.dir) / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["meta"]["judging_stopped"]["stage"], "judging")
        self.assertIn("Scores incomplete", (Path(self.dir) / "results.md").read_text(encoding="utf-8"))

    def test_judge_daily_limit_is_recorded_apart_from_the_answers_stop(self) -> None:
        score = MagicMock(side_effect=run_eval.DailyLimitReached(rate_limit_error(TPD_MESSAGE)))
        with patch.object(run_eval, "_score_answers", score):
            args = argparse.Namespace(judge_model="j", judge_workers=1, judge_timeout=1)
            with contextlib.redirect_stdout(io.StringIO()):
                stopped = run_eval._judge([answered_record(1)], ("basic",), args)
        self.assertEqual((stopped["stage"], stopped["retry_in"]), ("judging", "7m12.5s"))
        self.assertIn("--score-only", run_eval._stop_summary(stopped))

    def test_offset_past_the_end_is_refused(self) -> None:
        _, _, exit_code = self.main(["--offset", "4"], ([], None))
        self.assertIn("skips every question", str(exit_code))


class SavedRunTests(unittest.TestCase):
    """--score-only and --merge: no retrieval or answering, only saved records."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_main(self, argv: list[str], score=None):
        def fill_scores(per_question, pipelines, judge_model, workers, timeout, only_unscored=False):
            for q in per_question:
                run = q["answers"]["basic"]
                if not (only_unscored and run["faithfulness"] is not None):
                    run.update(faithfulness=0.5, answer_relevancy=0.5)

        score = score or MagicMock(side_effect=fill_scores)
        with (
            patch.object(sys, "argv", ["run_eval", "--out-dir", str(self.dir), *argv]),
            patch.object(run_eval, "_score_answers", score),
            patch.object(run_eval, "_git_info", return_value={"commit": "def", "dirty": False}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                run_eval.main()
                exit_code = 0
            except SystemExit as exc:
                exit_code = exc.code
        return score, exit_code

    def test_score_only_judges_unscored_answers_and_rewrites_both_files(self) -> None:
        stopped = {"stage": "answers", "reason": "groq_daily_limit", "at_question": 3, "completed": 2,
                   "resume_offset": 2, "retry_in": "1m", "detail": "x"}
        path = saved_run(self.dir / "results.json", [answered_record(1, faithfulness=0.9), answered_record(2)],
                         stopped=stopped, judge=None)

        score, exit_code = self.run_main(["--score-only", str(path), "--judge-model", "judge-x"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(score.call_args.args[5])  # only_unscored
        results = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([q["answers"]["basic"]["faithfulness"] for q in results["per_question"]], [0.9, 0.5])
        self.assertAlmostEqual(results["answers"]["basic"]["faithfulness"], 0.7)
        self.assertEqual(results["config"]["answers"]["judge_model"], "judge-x")
        self.assertEqual(results["meta"]["stopped"], stopped)  # answering is still partial
        self.assertEqual(results["meta"]["scoring_runs"][0]["judge_model"], "judge-x")
        self.assertIn("0.700 (n=2)", (self.dir / "results.md").read_text(encoding="utf-8"))

    def test_score_only_refuses_to_mix_judges_without_rescore(self) -> None:
        path = saved_run(self.dir / "results.json", [answered_record(1, faithfulness=0.9), answered_record(2)])
        score, exit_code = self.run_main(["--score-only", str(path), "--judge-model", "other"])
        self.assertIn("would mix judges", str(exit_code))
        score.assert_not_called()

        score, exit_code = self.run_main(["--score-only", str(path), "--judge-model", "other", "--rescore"])
        self.assertEqual(exit_code, 0)
        self.assertFalse(score.call_args.args[5])
        results = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(results["answers"]["basic"]["faithfulness"], 0.5)

    def test_score_only_needs_context_texts_or_approx_contexts(self) -> None:
        record = answered_record(1)
        del record["answers"]["basic"]["context_texts"]
        record["answers"]["basic"]["contexts"] = [["a.pdf", 2], ["a.pdf", 2], ["b.pdf", 1]]
        path = saved_run(self.dir / "results.json", [record])

        score, exit_code = self.run_main(["--score-only", str(path)])
        self.assertIn("--approx-contexts", str(exit_code))
        score.assert_not_called()

        doc = IndexedDoc("a.pdf", "id-a")
        doc.chunks[2] = [(0, "page two text")]
        with patch.object(run_eval, "load_user_docs", return_value={"a.pdf": doc}) as load:
            _, exit_code = self.run_main(["--score-only", str(path), "--approx-contexts"])
        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(USER)  # the run's own user when --user-id is absent
        run = json.loads(path.read_text(encoding="utf-8"))["per_question"][0]["answers"]["basic"]
        self.assertEqual(run["context_texts"], ["page two text"])  # each page once; unknown docs dropped
        self.assertTrue(run["context_texts_approximate"])
        self.assertIn("rebuilt from the index", (self.dir / "results.md").read_text(encoding="utf-8"))

    def test_score_only_judge_daily_limit_saves_progress_and_exits_3(self) -> None:
        path = saved_run(self.dir / "results.json", [answered_record(1)])
        limit = MagicMock(side_effect=run_eval.DailyLimitReached(rate_limit_error(TPD_MESSAGE)))
        _, exit_code = self.run_main(["--score-only", str(path)], score=limit)
        self.assertEqual(exit_code, 3)
        results = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(results["meta"]["judging_stopped"]["retry_in"], "7m12.5s")

    def test_merge_recomputes_summaries_from_every_question(self) -> None:
        stopped = {"stage": "answers", "reason": "groq_daily_limit", "at_question": 3, "completed": 2,
                   "resume_offset": 2, "retry_in": "1m", "detail": "x"}
        first = saved_run(self.dir / "results.json",
                          [answered_record(1, latency=1000.0, faithfulness=1.0),
                           answered_record(2, latency=2000.0, faithfulness=0.5)], stopped=stopped)
        second = saved_run(self.dir / "results.offset2.json",
                           [answered_record(3, recall=0.0, latency=6000.0, faithfulness=0.0)], offset=2)

        # Order on the command line doesn't matter.
        _, exit_code = self.run_main(["--merge", str(second), str(first)])

        self.assertEqual(exit_code, 0)
        results = json.loads((self.dir / "results.merged.json").read_text(encoding="utf-8"))
        self.assertEqual([q["number"] for q in results["per_question"]], [1, 2, 3])
        self.assertEqual((results["config"]["questions"], results["config"]["offset"]), (3, 0))
        self.assertIsNone(results["meta"]["stopped"])  # the resume finished the file
        self.assertEqual(results["meta"]["completed_questions"], 3)
        self.assertEqual([m["file"] for m in results["meta"]["merged_from"]], ["results.json", "results.offset2.json"])
        self.assertAlmostEqual(results["retrieval"]["vector"]["recall@5"], 2 / 3)
        self.assertEqual(results["retrieval"]["vector"]["latency"], {"p50_ms": 20.0, "p95_ms": 29.0})
        basic = results["answers"]["basic"]
        self.assertAlmostEqual(basic["faithfulness"], 0.5)
        self.assertEqual(basic["latency_end_to_end"], {"p50_ms": 2000.0, "p95_ms": 5600.0})
        markdown = (self.dir / "results.merged.md").read_text(encoding="utf-8")
        self.assertIn("Merged from 2 runs", markdown)
        self.assertNotIn("Partial run", markdown)

    def test_merge_refuses_incompatible_runs(self) -> None:
        a = saved_run(self.dir / "a.json", [answered_record(1)])
        b = saved_run(self.dir / "b.json", [answered_record(1)], scope="doc")
        _, exit_code = self.run_main(["--merge", str(a), str(b)])
        self.assertIn("scope differs", str(exit_code))
        self.assertIn("question 1 is in both", str(exit_code))

        c = saved_run(self.dir / "c.json", [answered_record(2, faithfulness=0.4)], judge="other")
        d = saved_run(self.dir / "d.json", [answered_record(3, faithfulness=0.4)])
        _, exit_code = self.run_main(["--merge", str(c), str(d)])
        self.assertIn("different judges", str(exit_code))
        self.assertFalse((self.dir / "results.merged.json").exists())


class RateLimitWaitTests(unittest.TestCase):
    def test_overlapping_waits_are_counted_once(self) -> None:
        stats = run_eval.RunStats()
        for started, ended in ((10.0, 12.0), (11.0, 13.0), (20.0, 21.0), (20.5, 20.8)):
            stats.rate_limited(started, ended)

        self.assertEqual(stats.rate_limit_retries, 4)
        self.assertAlmostEqual(stats.rate_limit_wait_s, 4.0)  # 10–13 and 20–21

    def test_no_waits(self) -> None:
        self.assertEqual(run_eval.RunStats().rate_limit_wait_s, 0.0)


class SummaryTests(unittest.TestCase):
    def answers(self, declined: bool, faithfulness=None, latency=1000.0, errors=(), wait=0.0, retries=0):
        return {
            "answer": "x", "declined": declined, "latency_ms": latency, "retrieval_ms": 200.0,
            "latency_excl_rate_limit_ms": latency - wait, "rate_limit_wait_ms": wait,
            "rate_limit_retries": retries,
            "llm_calls": 3, "prompt_tokens": 1000, "completion_tokens": 100, "errors": list(errors),
            "faithfulness": faithfulness, "answer_relevancy": faithfulness,
        }

    def test_summaries_and_markdown(self) -> None:
        per_question = [
            {**question(), "retrieval": {"vector": {"recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0, "latency_ms": 100.0, "error": None}},
             "answers": {"basic": self.answers(False, 0.9)}},
            {**question(), "retrieval": {"vector": {"recall@5": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "latency_ms": None, "error": "x"}},
             "answers": {"basic": self.answers(True, None, latency=3000.0, errors=["search: down"], wait=2000.0, retries=2)}},
            {**question(type="unanswerable", expected_pages=[]), "retrieval": {},
             "answers": {"basic": self.answers(True)}},
        ]
        retrieval = run_eval._summarize_retrieval(per_question, ("vector",))
        answers = run_eval._summarize_answers(per_question, ("basic",))

        self.assertEqual(retrieval["vector"]["n"], 2)
        self.assertEqual(retrieval["vector"]["recall@5"], 0.5)
        self.assertEqual(retrieval["vector"]["errors"], 1)
        self.assertEqual(retrieval["vector"]["latency"]["p50_ms"], 100.0)
        basic = answers["basic"]
        self.assertEqual((basic["faithfulness"], basic["faithfulness_scored"]), (0.9, 1))
        self.assertEqual((basic["unanswerable"]["declined"], basic["unanswerable"]["n"]), (1, 1))
        self.assertEqual((basic["answerable_declined"]["declined"], basic["answerable_declined"]["n"]), (1, 2))
        self.assertEqual(basic["llm_calls"], {"total": 9, "per_question": 3.0})
        self.assertEqual(basic["errors"]["questions_with_errors"], 1)
        # Latencies 1000/3000/1000 ms; without the 2 s wait the slow one is 1000 too.
        self.assertEqual(basic["latency_end_to_end"], {"p50_ms": 1000.0, "p95_ms": 2800.0})
        self.assertEqual(basic["latency_end_to_end_excl_rate_limit"], {"p50_ms": 1000.0, "p95_ms": 1000.0})
        self.assertEqual(basic["rate_limit"], {"retries": 2, "questions_with_retries": 1, "wait_total_ms": 2000.0})

        markdown = run_eval._markdown({
            "meta": {"timestamp": "2026-10-22T10:00:00+00:00", "git": {"commit": "414f7f42adcd3dd4", "dirty": True}},
            "config": {"questions": 3, "questions_by_type": {"factual": 2, "unanswerable": 1}, "scope": "all",
                       "answers": {"retrieval_mode": "hybrid", "judge_model": "qwen/qwen3.8-27b"}},
            "retrieval": retrieval,
            "answers": answers,
        })
        self.assertIn("Commit `414f7f42ad (uncommitted changes)`", markdown)
        self.assertIn("| vector | 0.500 | 0.500 | 0.500 | 100 | 100 | 1 |", markdown)
        self.assertIn("| basic | 0.900 (n=1) | 0.900 (n=1) | 1/1 | 1/2 | 1 q (1) |", markdown)
        self.assertIn("| basic | 1,000 / 2,800 | 1,000 / 1,000 | 200 / 200 | 2 (1) | 2.0 | 1000 / 100 | 3.0 |", markdown)


if __name__ == "__main__":
    unittest.main()
