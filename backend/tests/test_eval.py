"""eval/: metrics, question drafting helpers, and the runner's retrieval and instrumentation."""

import asyncio
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_errors_the_app_swallows_are_counted(self) -> None:
        result = self.run_basic(FakeGroq(fail=True), AsyncMock(return_value=[hit(2)]))
        self.assertEqual(result["llm_calls"], 1)
        self.assertIn("llm: RuntimeError: groq 503", result["errors"][0])
        self.assertTrue(result["declined"])  # the app's fallback has has_answer=false

        result = self.run_basic(FakeGroq(), AsyncMock(side_effect=ConnectionError("chroma down")))
        self.assertIn("search: ConnectionError: chroma down", result["errors"][0])
        self.assertEqual(result["llm_calls"], 0)
        self.assertTrue(result["declined"])

    def test_calls_outside_a_run_are_not_counted(self) -> None:
        with patch.object(rag, "_client", FakeGroq()):
            run_eval._install_instrumentation()
            rag.get_groq_client().chat.completions.create(model="m", messages=[])  # e.g. a judge call
        self.assertIsNone(run_eval._current_run.get())


class SummaryTests(unittest.TestCase):
    def answers(self, declined: bool, faithfulness=None, latency=1000.0, errors=()):
        return {
            "answer": "x", "declined": declined, "latency_ms": latency, "retrieval_ms": 200.0,
            "llm_calls": 3, "prompt_tokens": 1000, "completion_tokens": 100, "errors": list(errors),
            "faithfulness": faithfulness, "answer_relevancy": faithfulness,
        }

    def test_summaries_and_markdown(self) -> None:
        per_question = [
            {**question(), "retrieval": {"vector": {"recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0, "latency_ms": 100.0, "error": None}},
             "answers": {"basic": self.answers(False, 0.9)}},
            {**question(), "retrieval": {"vector": {"recall@5": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "latency_ms": None, "error": "x"}},
             "answers": {"basic": self.answers(True, None, latency=3000.0, errors=["search: down"])}},
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

        markdown = run_eval._markdown({
            "meta": {"timestamp": "2026-10-22T10:00:00+00:00", "git": {"commit": "414f7f42adcd3dd4", "dirty": True}},
            "config": {"questions": 3, "questions_by_type": {"factual": 2, "unanswerable": 1}, "scope": "all",
                       "answers": {"retrieval_mode": "hybrid", "judge_model": "qwen/qwen3.8-27b"}},
            "retrieval": retrieval,
            "answers": answers,
        })
        self.assertIn("Commit `414f7f42ad (uncommitted changes)`", markdown)
        self.assertIn("| vector | 0.500 | 0.500 | 0.500 | 100 | 100 | 1 |", markdown)
        self.assertIn("| basic | 0.900 (n=1) | 0.900 (n=1) | 1/1 | 1/2 |", markdown)


if __name__ == "__main__":
    unittest.main()
