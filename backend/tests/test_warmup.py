"""Startup warm-up: BM25 index and reranker load in the background, not on the first query."""

import asyncio
import contextlib
import io
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main
from core import config
from services import embedder

CHUNKS = {
    "ids": ["c0"],
    "documents": ["rarekeyword text"],
    "metadatas": [{"doc_id": "doc-a", "user_id": "user-a", "doc_name": "a.pdf",
                   "page_number": 1, "image_filename": ""}],
}


class BM25StateTestCase(unittest.TestCase):
    """Starts each test with an empty, unbuilt index and restores the real one after."""

    def setUp(self) -> None:
        for name, value in (("_bm25_index", None), ("_bm25_records", []),
                            ("_bm25_ready", False), ("_reranker", None)):
            p = patch.object(embedder, name, value)
            p.start()
            self.addCleanup(p.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        self.output = quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)


class WarmUpTests(BM25StateTestCase):
    def test_builds_index_and_loads_reranker_and_logs_durations(self) -> None:
        collection = MagicMock()
        collection.get.return_value = CHUNKS
        with (
            patch.object(embedder, "HYBRID_SEARCH_ENABLED", True),
            patch.object(embedder, "RERANK_ENABLED", True),
            patch.object(embedder, "collection", collection),
            patch("flashrank.Ranker") as ranker_cls,
        ):
            embedder.warm_up()

        self.assertTrue(embedder._bm25_ready)
        self.assertIsNotNone(embedder._bm25_index)
        ranker_cls.assert_called_once()
        log = self.output.getvalue()
        self.assertRegex(log, r"\[warmup\] bm25 index ready in \d+\.\d\ds")
        self.assertRegex(log, r"\[warmup\] reranker loaded in \d+\.\d\ds")
        self.assertRegex(log, r"\[warmup\] done in \d+\.\d\ds")

    def test_skips_the_reranker_when_reranking_is_off(self) -> None:
        collection = MagicMock()
        collection.get.return_value = CHUNKS
        with (
            patch.object(embedder, "HYBRID_SEARCH_ENABLED", True),
            patch.object(embedder, "RERANK_ENABLED", False),
            patch.object(embedder, "collection", collection),
            patch("flashrank.Ranker") as ranker_cls,
        ):
            embedder.warm_up()

        self.assertTrue(embedder._bm25_ready)
        ranker_cls.assert_not_called()

    def test_does_nothing_without_hybrid_search(self) -> None:
        # Neither the index nor the reranker is used on the vector-only path.
        collection = MagicMock()
        with (
            patch.object(embedder, "HYBRID_SEARCH_ENABLED", False),
            patch.object(embedder, "RERANK_ENABLED", True),
            patch.object(embedder, "collection", collection),
            patch("flashrank.Ranker") as ranker_cls,
        ):
            embedder.warm_up()

        collection.get.assert_not_called()
        ranker_cls.assert_not_called()

    def test_failures_are_logged_not_raised(self) -> None:
        collection = MagicMock()
        collection.get.side_effect = ConnectionError("chroma down")
        with (
            patch.object(embedder, "HYBRID_SEARCH_ENABLED", True),
            patch.object(embedder, "RERANK_ENABLED", True),
            patch.object(embedder, "collection", collection),
            patch("flashrank.Ranker", side_effect=OSError("download failed")),
        ):
            embedder.warm_up()

        self.assertFalse(embedder._bm25_ready)  # the first query retries
        log = self.output.getvalue()
        self.assertIn("[warmup] bm25 index failed: ConnectionError: chroma down", log)
        self.assertIn("[warmup] reranker failed: OSError: download failed", log)

    def test_reranker_uses_the_persistent_cache_dir(self) -> None:
        with patch("flashrank.Ranker") as ranker_cls:
            embedder._get_reranker()
        self.assertEqual(ranker_cls.call_args.kwargs["cache_dir"], config.FLASHRANK_CACHE_DIR)
        self.assertFalse(config.FLASHRANK_CACHE_DIR.startswith("/tmp"))


class RebuildConcurrencyTests(BM25StateTestCase):
    def test_query_during_warm_up_waits_for_that_build(self) -> None:
        started, release = threading.Event(), threading.Event()
        collection = MagicMock()

        def slow_get(**_kwargs):
            started.set()
            release.wait(5)
            return CHUNKS

        collection.get.side_effect = slow_get
        with patch.object(embedder, "collection", collection):
            warm = threading.Thread(target=embedder._rebuild_bm25_index)
            warm.start()
            self.assertTrue(started.wait(5))
            results: list = []
            query = threading.Thread(
                target=lambda: results.append(embedder._search_bm25("rarekeyword", 5, "user-a", None))
            )
            query.start()
            release.set()
            warm.join(5)
            query.join(5)

        collection.get.assert_called_once()
        self.assertEqual([h["chunk_id"] for h in results[0]], ["c0"])

    def test_upload_during_rebuild_discards_the_stale_index(self) -> None:
        collection = MagicMock()

        def get_then_upload(**_kwargs):
            embedder._invalidate_bm25_index()  # a new document lands mid-read
            return CHUNKS

        collection.get.side_effect = get_then_upload
        with patch.object(embedder, "collection", collection):
            embedder._rebuild_bm25_index()

        self.assertFalse(embedder._bm25_ready)
        self.assertIsNone(embedder._bm25_index)


class SearchOffLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_keyword_search_runs_off_the_event_loop(self) -> None:
        loop_thread = threading.get_ident()
        seen: dict = {}

        def fake_bm25(*_args, **_kwargs):
            seen["thread"] = threading.get_ident()
            return []

        collection = MagicMock()
        collection.count.return_value = 1
        collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        with (
            patch.object(embedder, "HYBRID_SEARCH_ENABLED", True),
            patch.object(embedder, "RERANK_ENABLED", False),
            patch.object(embedder, "collection", collection),
            patch.object(embedder, "get_embedding", AsyncMock(return_value=[0.0])),
            patch.object(embedder, "_search_bm25", fake_bm25),
        ):
            await embedder.search_or_raise("q", user_id="user-a")

        self.assertNotEqual(seen["thread"], loop_thread)


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_does_not_wait_for_warm_up(self) -> None:
        loop_thread = threading.get_ident()
        running, release = threading.Event(), threading.Event()
        seen: dict = {}

        def slow_warm_up() -> None:
            seen["thread"] = threading.get_ident()
            running.set()
            release.wait(5)

        with (
            patch.object(main, "connect", AsyncMock()),
            patch.object(main, "ensure_schema", AsyncMock()),
            patch.object(main, "disconnect", AsyncMock()),
            patch.object(main.storage, "get_storage"),
            patch.object(main.embedder, "warm_up", slow_warm_up),
        ):
            try:
                started = time.perf_counter()
                async with main.lifespan(main.app):
                    # Startup finished while warm-up is still blocked.
                    self.assertLess(time.perf_counter() - started, 1)
                    self.assertTrue(await asyncio.to_thread(running.wait, 5))
                    self.assertFalse(release.is_set())
            finally:
                release.set()

        self.assertNotEqual(seen["thread"], loop_thread)


if __name__ == "__main__":
    unittest.main()
