"""Lazy FlashRank loading, off-loop scoring, and candidate count."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from services import embedder


def _hits(count: int) -> list[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "chunk_text": f"text {i}",
            "doc_name": "a.pdf",
            "page_number": i,
            "image_filename": "",
            "relevance_score": 0.5,
            "doc_id": "doc-a",
        }
        for i in range(count)
    ]


class LazyLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reset = patch.object(embedder, "_reranker", None)
        self.reset.start()

    def tearDown(self) -> None:
        self.reset.stop()

    def test_model_is_not_loaded_at_import(self) -> None:
        self.assertIsNone(embedder._reranker)

    def test_concurrent_first_use_loads_the_model_once(self) -> None:
        with patch("flashrank.Ranker") as ranker_cls:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rerankers = list(pool.map(lambda _: embedder._get_reranker(), range(16)))

        ranker_cls.assert_called_once()
        self.assertEqual(ranker_cls.call_args.kwargs["model_name"], embedder.RERANK_MODEL)
        self.assertTrue(all(r is rerankers[0] for r in rerankers))

    def test_default_model_is_a_flashrank_model(self) -> None:
        embedder._validate_rerank_model(embedder.RERANK_MODEL)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ms-marco-MiniLM-L-6-v2"):
            embedder._validate_rerank_model("ms-marco-MiniLM-L-6-v2")


class RerankHitsTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_off_the_event_loop_and_caps_candidates(self) -> None:
        loop_thread = threading.get_ident()
        seen: dict = {}

        class FakeReranker:
            def rerank(self, request) -> list[dict]:
                seen["thread"] = threading.get_ident()
                seen["passages"] = len(request.passages)
                return [{**p, "score": 1.0 - p["id"] / 100} for p in request.passages]

        with (
            patch.object(embedder, "_reranker", FakeReranker()),
            patch.object(embedder, "RERANK_CANDIDATES", 10),
        ):
            reranked = await embedder._rerank_hits("q", _hits(15), n_results=5)

        self.assertNotEqual(seen["thread"], loop_thread)
        self.assertEqual(seen["passages"], 10)
        self.assertEqual(len(reranked), 5)

    async def test_first_use_loads_the_model_off_the_event_loop(self) -> None:
        loop_thread = threading.get_ident()
        load_thread: dict = {}

        def fake_ranker(**_kwargs):
            load_thread["id"] = threading.get_ident()
            ranker = MagicMock()
            ranker.rerank.return_value = []
            return ranker

        with (
            patch.object(embedder, "_reranker", None),
            patch("flashrank.Ranker", side_effect=fake_ranker),
        ):
            await embedder._rerank_hits("q", _hits(3), n_results=3)

        self.assertNotEqual(load_thread["id"], loop_thread)

    async def test_reranker_failure_keeps_fused_order(self) -> None:
        hits = _hits(8)
        with patch.object(embedder, "_get_reranker", side_effect=OSError("download failed")):
            reranked = await embedder._rerank_hits("q", hits, n_results=5)

        self.assertEqual(reranked, hits[:5])


if __name__ == "__main__":
    unittest.main()
