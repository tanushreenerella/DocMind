"""Upload processing writes to object storage; chunk deletion is owner-scoped."""

import asyncio
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from core import database
from routers import upload
from services import embedder, storage
from tests.fakes import FakeStorage

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC = "11111111-2222-3333-4444-555555555555"


class _ProcessingTestCase(unittest.TestCase):
    """Runs the real _process_document against fake storage; no tests of its own."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "temp"))
        self.file_path = os.path.join(self.root, "temp", f"{DOC}_report.pdf")
        with open(self.file_path, "wb") as f:
            f.write(b"%PDF-1.4 report")
        self.storage = FakeStorage()
        self.storage.put(storage.pdf_key(USER, DOC), b"%PDF-1.4 stored original", "application/pdf")
        self.statuses: list[dict] = []

        async def set_status(doc_id: str, **fields) -> None:
            self.statuses.append(fields)

        self.patches = [
            patch.object(storage, "_storage", self.storage),
            patch.object(upload, "STORAGE_PATH", self.root),
            patch.object(upload, "_set_status", set_status),
            patch.object(upload, "classify_document", AsyncMock(return_value={})),
            patch.object(upload, "index_document", AsyncMock()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    async def _parse_two_pages(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
        os.makedirs(image_dir, exist_ok=True)
        pages = []
        for n in (1, 2):
            name = f"{doc_id}_p{n}.jpg" if n == 1 else ""  # page 2 failed to render
            if name:
                with open(os.path.join(image_dir, name), "wb") as f:
                    f.write(b"\xff\xd8\xff page 1")
            pages.append({
                "page_number": n, "text": f"page {n}",
                "image_path": os.path.join(image_dir, name) if name else "",
                "image_filename": name, "has_tables": False, "tables": [],
            })
        return pages

    def _process(self) -> None:
        asyncio.run(upload._process_document(DOC, USER, self.file_path, "report.pdf"))


class ProcessDocumentTests(_ProcessingTestCase):
    def test_page_images_go_to_storage_and_temp_files_are_removed(self) -> None:
        with patch.object(upload, "parse_document", self._parse_two_pages):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "complete")
        self.assertEqual(
            self.storage.objects[storage.page_image_key(USER, DOC, f"{DOC}_p1.jpg")],
            (b"\xff\xd8\xff page 1", "image/jpeg"),
        )
        # The original was stored by the upload itself; processing never rewrites it.
        self.assertEqual(
            self.storage.objects[storage.pdf_key(USER, DOC)],
            (b"%PDF-1.4 stored original", "application/pdf"),
        )
        self.assertEqual(len(self.storage.keys()), 2)  # nothing for the unrendered page
        self.assertEqual(os.listdir(os.path.join(self.root, "temp")), [])

    def test_failed_parse_marks_error_and_still_cleans_up(self) -> None:
        with patch.object(upload, "parse_document", AsyncMock(side_effect=RuntimeError("bad pdf"))):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "error")
        self.assertEqual(os.listdir(os.path.join(self.root, "temp")), [])
        # The original stays stored so the user can still open or delete it.
        self.assertEqual(self.storage.keys(), [storage.pdf_key(USER, DOC)])

    def test_storage_failure_marks_error(self) -> None:
        broken = MagicMock()
        broken.put.side_effect = ConnectionError("R2 unreachable")
        with (
            patch.object(storage, "_storage", broken),
            patch.object(upload, "parse_document", self._parse_two_pages),
        ):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "error")
        self.assertIn("R2 unreachable", self.statuses[-1]["error_message"])
        self.assertEqual(os.listdir(os.path.join(self.root, "temp")), [])


class ProcessingOutcomeTests(_ProcessingTestCase):
    """complete / partial / error, decided by which pages' OCR failed."""

    def _parse_with_failures(self, failed: set[int], total: int = 3):
        async def parse(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
            return [
                {
                    "page_number": n, "text": "" if n in failed else f"page {n}",
                    "image_path": "", "image_filename": "", "has_tables": False,
                    "tables": [], "ocr_failed": n in failed,
                }
                for n in range(1, total + 1)
            ]
        return parse

    def test_no_failed_pages_is_complete(self) -> None:
        with patch.object(upload, "parse_document", self._parse_with_failures(set())):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "complete")
        self.assertEqual(self.statuses[-1]["failed_pages"], [])

    def test_some_failed_pages_is_partial_and_still_indexed(self) -> None:
        with patch.object(upload, "parse_document", self._parse_with_failures({1, 3})):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "partial")
        self.assertEqual(self.statuses[-1]["failed_pages"], [1, 3])
        upload.index_document.assert_awaited_once()

    def test_every_page_failed_is_an_error_with_nothing_indexed(self) -> None:
        output = io.StringIO()
        with (
            patch.object(upload, "parse_document", self._parse_with_failures({1, 2, 3})),
            contextlib.redirect_stdout(output),
        ):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "error")
        upload.index_document.assert_not_awaited()
        # The real reason, with traceback, is logged server-side.
        self.assertIn("OCR failed on all 3 pages", output.getvalue())
        self.assertIn("Traceback", output.getvalue())


class ChromaWriteRetryTests(unittest.IsolatedAsyncioTestCase):
    async def _index(self, upsert: MagicMock) -> None:
        page = {"page_number": 1, "text": "some text", "image_filename": ""}
        with (
            patch.object(embedder, "collection", MagicMock(upsert=upsert)),
            patch.object(embedder, "get_embeddings", AsyncMock(return_value=[[0.1]])),
            patch.object(embedder, "CHROMA_RETRY_BASE_SECONDS", 0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await embedder.index_document(DOC, USER, "a.pdf", [page], {})

    async def test_transport_errors_are_retried(self) -> None:
        upsert = MagicMock(side_effect=[httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), None])
        await self._index(upsert)

        self.assertEqual(upsert.call_count, 3)
        # Every attempt writes the same batch (same ids), so retries are idempotent.
        self.assertEqual(len({tuple(c.kwargs["ids"]) for c in upsert.call_args_list}), 1)

    async def test_gives_up_after_three_attempts(self) -> None:
        upsert = MagicMock(side_effect=httpx.ConnectError("refused"))
        with self.assertRaises(httpx.ConnectError):
            await self._index(upsert)
        self.assertEqual(upsert.call_count, 3)

    async def test_non_transport_errors_are_not_retried(self) -> None:
        upsert = MagicMock(side_effect=Exception('{"error":"InvalidArgument"}'))
        with self.assertRaises(Exception):
            await self._index(upsert)
        self.assertEqual(upsert.call_count, 1)

    async def test_backoff_doubles(self) -> None:
        upsert = MagicMock(side_effect=[httpx.ConnectError("x"), httpx.ConnectError("x"), None])
        sleep = AsyncMock()
        page = {"page_number": 1, "text": "some text", "image_filename": ""}
        with (
            patch.object(embedder, "collection", MagicMock(upsert=upsert)),
            patch.object(embedder, "get_embeddings", AsyncMock(return_value=[[0.1]])),
            patch.object(embedder.asyncio, "sleep", sleep),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await embedder.index_document(DOC, USER, "a.pdf", [page], {})

        self.assertEqual([c.args[0] for c in sleep.await_args_list], [1.0, 2.0])


class SchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_updates_are_idempotent_and_applied(self) -> None:
        pool = MagicMock(execute=AsyncMock())
        with patch.object(database, "get_pool", return_value=pool):
            await database.ensure_schema()

        statements = [c.args[0] for c in pool.execute.await_args_list]
        self.assertEqual(statements, list(database.SCHEMA_UPDATES))
        for statement in statements:
            self.assertIn("IF NOT EXISTS", statement)
        self.assertTrue(any("failed_pages" in st for st in statements))
        for column in ("hidden_text_pages", "hidden_text_removed", "suspicious"):
            self.assertTrue(any(column in st for st in statements), column)


class DeleteChunksTests(unittest.TestCase):
    def test_delete_filters_on_document_and_owner_and_resets_bm25(self) -> None:
        fake_collection = MagicMock()
        with (
            patch.object(embedder, "collection", fake_collection),
            patch.object(embedder, "_bm25_ready", True),
        ):
            embedder.delete_document_chunks(DOC, USER)
            bm25_ready_after = embedder._bm25_ready

        fake_collection.delete.assert_called_once_with(
            where={"$and": [{"doc_id": {"$eq": DOC}}, {"user_id": {"$eq": USER}}]}
        )
        self.assertFalse(bm25_ready_after)


if __name__ == "__main__":
    unittest.main()
