"""Focused tests for the opt-in Chroma + BM25 retrieval path."""

import unittest
from unittest.mock import AsyncMock, patch

from services import embedder


class FakeCollection:
    def __init__(self) -> None:
        self.records = [
            (
                "a-vector",
                "vector matching text",
                {
                    "doc_id": "doc-a",
                    "user_id": "user-a",
                    "doc_name": "a.pdf",
                    "page_number": 1,
                    "image_filename": "a.jpg",
                },
            ),
            (
                "a-keyword",
                "rarekeyword appears only in User A's document",
                {
                    "doc_id": "doc-a",
                    "user_id": "user-a",
                    "doc_name": "a.pdf",
                    "page_number": 2,
                    "image_filename": "a-2.jpg",
                },
            ),
            (
                "b-keyword",
                "rarekeyword appears only in User B's document",
                {
                    "doc_id": "doc-b",
                    "user_id": "user-b",
                    "doc_name": "b.pdf",
                    "page_number": 1,
                    "image_filename": "b.jpg",
                },
            ),
        ]
        self.get_calls = 0

    def count(self) -> int:
        return len(self.records)

    def query(self, **_kwargs: object) -> dict:
        # The fake vector result deliberately includes one overlapping chunk.
        self.last_query_kwargs = _kwargs
        vector_records = [self.records[0], self.records[1]]
        return {
            "ids": [[record[0] for record in vector_records]],
            "documents": [[record[1] for record in vector_records]],
            "metadatas": [[record[2] for record in vector_records]],
            "distances": [[0.1, 0.2]],
        }

    def get(self, **_kwargs: object) -> dict:
        self.get_calls += 1
        return {
            "ids": [record[0] for record in self.records],
            "documents": [record[1] for record in self.records],
            "metadatas": [record[2] for record in self.records],
        }


class HybridSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.collection = FakeCollection()
        self.collection_patch = patch.object(
            embedder,
            "collection",
            self.collection,
        )
        self.embedding_patch = patch.object(
            embedder,
            "get_embedding",
            AsyncMock(return_value=[0.1]),
        )
        self.collection_patch.start()
        self.embedding_patch.start()
        embedder._invalidate_bm25_index()

    def tearDown(self) -> None:
        self.collection_patch.stop()
        self.embedding_patch.stop()
        embedder._invalidate_bm25_index()

    async def test_flag_disabled_keeps_chroma_only_behavior(self) -> None:
        with patch.object(embedder, "HYBRID_SEARCH_ENABLED", False):
            hits = await embedder.search(
                "rarekeyword",
                user_id="user-a",
                doc_id="doc-a",
            )

        self.assertEqual(
            [hit["chunk_id"] for hit in hits],
            ["a-vector", "a-keyword"],
        )
        self.assertEqual(self.collection.get_calls, 0)

    async def test_rrf_combines_vector_and_bm25_results(self) -> None:
        with patch.object(embedder, "HYBRID_SEARCH_ENABLED", True):
            hits = await embedder.search(
                "rarekeyword",
                user_id="user-a",
                doc_id="doc-a",
            )

        self.assertEqual(
            [hit["chunk_id"] for hit in hits],
            ["a-keyword", "a-vector"],
        )
        self.assertEqual(self.collection.get_calls, 1)

    def test_rrf_boosts_a_chunk_ranked_by_both_retrievers(self) -> None:
        vector_hits = [
            {"chunk_id": "vector-only"},
            {"chunk_id": "shared"},
        ]
        bm25_hits = [
            {"chunk_id": "shared"},
            {"chunk_id": "keyword-only"},
        ]

        fused = embedder._fuse_with_rrf(
            vector_hits,
            bm25_hits,
            n_results=3,
        )

        self.assertEqual(
            [hit["chunk_id"] for hit in fused],
            ["shared", "vector-only", "keyword-only"],
        )

    async def test_bm25_never_returns_another_users_chunks(self) -> None:
        with patch.object(embedder, "HYBRID_SEARCH_ENABLED", True):
            hits = await embedder.search(
                "rarekeyword",
                user_id="user-a",
                doc_id="doc-a",
            )

        self.assertTrue(hits)
        self.assertTrue(
            all(hit["doc_id"] != "doc-b" for hit in hits)
        )
        self.assertEqual(
            self.collection.last_query_kwargs["where"],
            {
                "$and": [
                    {"user_id": {"$eq": "user-a"}},
                    {"doc_id": {"$eq": "doc-a"}},
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
