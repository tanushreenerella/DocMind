"""Groq calls run off the event loop, so concurrent requests overlap instead of queueing."""

import asyncio
import contextlib
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services import classifier, rag

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _chunk(query: str) -> dict:
    return {
        "chunk_id": f"{query}-0", "chunk_text": f"evidence for {query}", "doc_name": "a.pdf",
        "page_number": 1, "image_filename": "", "relevance_score": 0.9, "doc_id": "doc-a", "user_id": USER,
    }


class OverlappingGroq:
    """A sync client whose calls only return once ``parties`` of them are in flight at once.

    Run one at a time (a call blocking the event loop), the first call can never
    meet the second, the barrier times out, and the call fails.
    """

    def __init__(self, content: str, parties: int = 2) -> None:
        self.content = content
        self.barrier = threading.Barrier(parties, timeout=2)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        self.barrier.wait()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_basic_chats_overlap(self) -> None:
        with (
            patch.object(rag, "_client", OverlappingGroq("Covered [SOURCE 1].")),
            patch.object(rag, "search", AsyncMock(side_effect=lambda q, **_: [_chunk(q)])),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = await asyncio.wait_for(
                asyncio.gather(
                    rag.answer_query("first?", [], user_id=USER),
                    rag.answer_query("second?", [], user_id=USER),
                ),
                timeout=5,
            )

        for result in results:
            self.assertEqual((result["error"], result["has_answer"]), (None, True))
            self.assertEqual(result["answer"], "Covered [a.pdf, p.1].")

    async def test_two_classifications_overlap(self) -> None:
        reply = json.dumps({"document_type": "invoice"})
        with (
            patch.object(classifier, "_client", OverlappingGroq(reply)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = await asyncio.wait_for(
                asyncio.gather(
                    classifier.classify_document("text one", "one.pdf"),
                    classifier.classify_document("text two", "two.pdf"),
                ),
                timeout=5,
            )

        # The fallback (document_type "other") would mean a call failed at the barrier.
        self.assertEqual([r["document_type"] for r in results], ["invoice", "invoice"])


if __name__ == "__main__":
    unittest.main()
