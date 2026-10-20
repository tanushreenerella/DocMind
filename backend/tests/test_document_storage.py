"""Upload processing writes to object storage; chunk deletion is owner-scoped."""

import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from routers import upload
from services import embedder, storage
from tests.fakes import FakeStorage

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC = "11111111-2222-3333-4444-555555555555"


class ProcessDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "temp"))
        self.file_path = os.path.join(self.root, "temp", f"{DOC}_report.pdf")
        with open(self.file_path, "wb") as f:
            f.write(b"%PDF-1.4 report")
        self.storage = FakeStorage()
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

    def test_pdf_and_page_images_go_to_storage_and_temp_files_are_removed(self) -> None:
        with patch.object(upload, "parse_document", self._parse_two_pages):
            self._process()

        self.assertEqual(self.statuses[-1]["status"], "complete")
        self.assertEqual(
            self.storage.objects[storage.pdf_key(USER, DOC)],
            (b"%PDF-1.4 report", "application/pdf"),
        )
        self.assertEqual(
            self.storage.objects[storage.page_image_key(USER, DOC, f"{DOC}_p1.jpg")],
            (b"\xff\xd8\xff page 1", "image/jpeg"),
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
