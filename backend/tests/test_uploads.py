"""Direct uploads: presign → PUT (R2, or the local-disk route) → complete."""

import os
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import tests.test_cross_user_isolation as iso
from core import config
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from main import app
from routers import upload
from services import parser, storage
from tests import PROXY_HEADERS
from tests.fakes import FakeStorage

USER = iso.USER_B


def make_pdf(pages: int) -> bytes:
    """A minimal but valid PDF with ``pages`` blank pages (correct xref offsets)."""
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
        *[b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"] * pages,
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class PageCountTests(unittest.TestCase):
    def test_counts_pages_of_a_real_pdf(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(make_pdf(3))
        try:
            self.assertEqual(parser.count_pdf_pages(f.name), 3)
        finally:
            os.remove(f.name)

    def test_garbage_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 but nothing else")
        try:
            with self.assertRaises(Exception):
                parser.count_pdf_pages(f.name)
        finally:
            os.remove(f.name)


class _UploadApiTestCase(unittest.TestCase):
    direct_uploads = False

    def setUp(self) -> None:
        limiter.reset()
        self.db = iso.FakeDB()
        self.storage = FakeStorage(direct_uploads=self.direct_uploads)
        self.temp_root = tempfile.mkdtemp()
        self.parse = AsyncMock(return_value=[{
            "page_number": 1, "text": "hello", "image_path": "", "image_filename": "",
            "has_tables": False, "tables": [],
        }])
        self.patches = [
            patch(f"{module}.get_pool", return_value=self.db)
            for module in ("core.auth", "routers.upload", "routers.documents")
        ] + [
            patch.object(storage, "_storage", self.storage),
            patch.object(upload, "STORAGE_PATH", self.temp_root),
            patch.object(upload, "parse_document", self.parse),
            patch.object(upload, "classify_document", AsyncMock(return_value={})),
            patch.object(upload, "index_document", AsyncMock()),
        ]
        for p in self.patches:
            p.start()
        token = create_access_token(USER, "b@example.com")
        self.client = TestClient(app, headers={
            **PROXY_HEADERS,
            "Origin": config.ALLOWED_ORIGINS[0],
            "Cookie": f"{ACCESS_TOKEN_COOKIE}={token}",
        })

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.temp_root, ignore_errors=True)
        limiter.reset()

    def presign(self, filename="report.pdf", size=1000, content_type="application/pdf"):
        return self.client.post("/api/uploads/presign", json={
            "filename": filename, "size": size, "content_type": content_type,
        })

    def ticket(self, data: bytes, filename="report.pdf") -> dict:
        response = self.presign(filename, len(data))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def complete(self, doc_id: str):
        return self.client.post(f"/api/uploads/{doc_id}/complete")

    def row(self, doc_id: str) -> dict | None:
        return next((r for r in self.db.tables["documents"] if r["doc_id"] == doc_id), None)

    def store(self, doc_id: str, data: bytes) -> None:
        self.storage.put(storage.pdf_key(USER, doc_id), data, "application/pdf")

    def temp_files(self) -> list[str]:
        temp = os.path.join(self.temp_root, "temp")
        return os.listdir(temp) if os.path.isdir(temp) else []


class PresignTests(_UploadApiTestCase):
    def test_local_disk_gets_the_api_upload_route(self) -> None:
        response = self.presign("My Report.pdf", 1234)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["upload_url"], f"/api/uploads/{body['doc_id']}/file")
        self.assertEqual(body["method"], "PUT")
        self.assertEqual(body["headers"], {"Content-Type": "application/pdf"})
        self.assertEqual(body["filename"], "My_Report.pdf")
        row = self.row(body["doc_id"])
        self.assertEqual((row["user_id"], row["status"], row["doc_name"]), (USER, "uploading", "My_Report.pdf"))

    def test_rejects_non_pdfs(self) -> None:
        for filename, content_type in (
            ("notes.txt", "application/pdf"),
            ("scan.png", "image/png"),
            ("report.pdf", "application/octet-stream"),
        ):
            with self.subTest(filename=filename, content_type=content_type):
                response = self.presign(filename, 100, content_type)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"], "Only PDF files are supported")
        self.assertEqual(self.db.tables["documents"], [])

    def test_rejects_empty_and_oversized_files(self) -> None:
        self.assertEqual(self.presign(size=0).status_code, 400)

        response = self.presign(size=config.MAX_UPLOAD_BYTES + 1)
        self.assertEqual(response.status_code, 413)
        self.assertIn(f"the limit is {config.MAX_UPLOAD_MB} MB", response.json()["detail"])
        self.assertEqual(self.db.tables["documents"], [])

    def test_exactly_the_limit_is_allowed(self) -> None:
        self.assertEqual(self.presign(size=config.MAX_UPLOAD_BYTES).status_code, 200)

    def test_requires_login(self) -> None:
        anonymous = TestClient(app, headers={**PROXY_HEADERS, "Origin": config.ALLOWED_ORIGINS[0]})
        response = anonymous.post("/api/uploads/presign", json={
            "filename": "a.pdf", "size": 10, "content_type": "application/pdf",
        })
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.db.tables["documents"], [])


class R2PresignTests(_UploadApiTestCase):
    direct_uploads = True

    def test_r2_gets_a_presigned_url_for_the_users_original_pdf_key(self) -> None:
        body = self.ticket(b"x" * 10)

        key = storage.pdf_key(USER, body["doc_id"])
        self.assertEqual(
            body["upload_url"],
            f"https://r2.example/{key}?ct=application/pdf&expires=600&sig=x",
        )
        self.assertEqual(body["expires_in"], 600)

    def test_local_put_route_is_disabled_with_r2(self) -> None:
        body = self.ticket(b"%PDF-1.4")
        response = self.client.put(
            f"/api/uploads/{body['doc_id']}/file",
            content=b"%PDF-1.4", headers={"Content-Type": "application/pdf"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.storage.keys(), [])

    def test_complete_processes_the_object_the_browser_put_in_r2(self) -> None:
        pdf = make_pdf(2)
        body = self.ticket(pdf)
        self.store(body["doc_id"], pdf)  # what the browser's PUT would have done

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["page_count"], 2)
        self.assertEqual(self.row(body["doc_id"])["status"], "complete")
        # Processing read the stored bytes (via a temp copy it then removed).
        parsed_path = self.parse.await_args.args[0]
        self.assertTrue(parsed_path.endswith(f"{body['doc_id']}_report.pdf"))
        self.assertEqual(self.temp_files(), [])


class LocalPutTests(_UploadApiTestCase):
    def test_put_stores_the_original(self) -> None:
        body = self.ticket(b"%PDF-1.4 data")
        response = self.client.put(body["upload_url"], content=b"%PDF-1.4 data", headers=body["headers"])

        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(
            self.storage.objects[storage.pdf_key(USER, body["doc_id"])],
            (b"%PDF-1.4 data", "application/pdf"),
        )

    def test_put_over_the_limit_is_413_and_stores_nothing(self) -> None:
        body = self.ticket(b"%PDF")
        with patch.object(upload, "MAX_UPLOAD_BYTES", 10):
            response = self.client.put(body["upload_url"], content=b"%PDF" + b"x" * 20, headers=body["headers"])

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.storage.keys(), [])

    def test_put_needs_the_pdf_content_type(self) -> None:
        body = self.ticket(b"%PDF")
        response = self.client.put(body["upload_url"], content=b"%PDF", headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.storage.keys(), [])

    def test_put_after_complete_is_refused(self) -> None:
        pdf = make_pdf(1)
        body = self.ticket(pdf)
        self.client.put(body["upload_url"], content=pdf, headers=body["headers"])
        self.assertEqual(self.complete(body["doc_id"]).status_code, 200)

        response = self.client.put(body["upload_url"], content=b"%PDF-1.4 swapped", headers=body["headers"])

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.storage.objects[storage.pdf_key(USER, body["doc_id"])][0], pdf)

    def test_malformed_doc_id_is_404(self) -> None:
        response = self.client.put("/api/uploads/not-a-uuid/file", content=b"%PDF",
                                   headers={"Content-Type": "application/pdf"})
        self.assertEqual(response.status_code, 404)


class CompleteTests(_UploadApiTestCase):
    def test_full_local_flow_processes_the_document(self) -> None:
        pdf = make_pdf(3)
        body = self.ticket(pdf)
        self.assertEqual(
            self.client.put(body["upload_url"], content=pdf, headers=body["headers"]).status_code, 204
        )

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"doc_id": body["doc_id"], "filename": "report.pdf", "page_count": 3})
        self.assertEqual(self.row(body["doc_id"])["status"], "complete")
        self.parse.assert_awaited_once()
        self.assertEqual(self.temp_files(), [])

    def test_second_complete_is_409_and_starts_nothing(self) -> None:
        pdf = make_pdf(1)
        body = self.ticket(pdf)
        self.store(body["doc_id"], pdf)
        self.assertEqual(self.complete(body["doc_id"]).status_code, 200)

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 409)
        self.parse.assert_awaited_once()
        self.assertIsNotNone(self.row(body["doc_id"]))

    def test_nothing_uploaded_is_400_and_the_upload_stays_pending(self) -> None:
        body = self.ticket(b"%PDF")

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.row(body["doc_id"])["status"], "uploading")

    def test_stored_file_over_the_limit_is_413_and_discarded(self) -> None:
        # The presigned PUT can't bound the size, so a client can lie in presign.
        body = self.ticket(b"%PDF")
        self.store(body["doc_id"], b"%PDF" + b"x" * 50)

        with patch.object(upload, "MAX_UPLOAD_BYTES", 20):
            response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 413)
        self.assertIn("the limit is", response.json()["detail"])
        self.assertEqual(self.storage.keys(), [])
        self.assertIsNone(self.row(body["doc_id"]))
        self.parse.assert_not_awaited()

    def test_wrong_magic_bytes_is_400_and_discarded(self) -> None:
        body = self.ticket(b"PK\x03\x04")
        self.store(body["doc_id"], b"PK\x03\x04 a zip renamed to .pdf")

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "This file is not a PDF")
        self.assertEqual(self.storage.keys(), [])
        self.assertIsNone(self.row(body["doc_id"]))
        self.assertEqual(self.temp_files(), [])
        self.parse.assert_not_awaited()

    def test_unreadable_pdf_is_400_and_discarded(self) -> None:
        body = self.ticket(b"%PDF")
        self.store(body["doc_id"], b"%PDF-1.4 truncated garbage")

        response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 400)
        self.assertIn("could not be read", response.json()["detail"])
        self.assertIsNone(self.row(body["doc_id"]))
        self.assertEqual(self.temp_files(), [])

    def test_too_many_pages_is_422_with_a_clear_message(self) -> None:
        pdf = make_pdf(4)
        body = self.ticket(pdf)
        self.store(body["doc_id"], pdf)

        with patch.object(upload, "MAX_PAGES", 3):
            response = self.complete(body["doc_id"])

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "This PDF has 4 pages; the limit is 3. Split it into smaller files and upload them separately.",
        )
        self.assertEqual(self.storage.keys(), [])
        self.assertIsNone(self.row(body["doc_id"]))
        self.assertEqual(self.temp_files(), [])
        self.parse.assert_not_awaited()

    def test_exactly_max_pages_is_accepted(self) -> None:
        pdf = make_pdf(3)
        body = self.ticket(pdf)
        self.store(body["doc_id"], pdf)

        with patch.object(upload, "MAX_PAGES", 3):
            self.assertEqual(self.complete(body["doc_id"]).status_code, 200)

    def test_storage_outage_keeps_the_upload_for_a_retry(self) -> None:
        pdf = make_pdf(1)
        body = self.ticket(pdf)
        self.store(body["doc_id"], pdf)

        # Let the app's 500 handler answer instead of re-raising into the test.
        self.client = TestClient(app, headers=self.client.headers, raise_server_exceptions=False)
        with patch.object(self.storage, "download", MagicMock(side_effect=ConnectionError("R2 down"))):
            response = self.complete(body["doc_id"])
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.row(body["doc_id"])["status"], "uploading")
        self.assertEqual(self.temp_files(), [])

        self.assertEqual(self.complete(body["doc_id"]).status_code, 200)

    def test_unknown_doc_is_404(self) -> None:
        self.assertEqual(self.complete(str(uuid.uuid4())).status_code, 404)

    def test_pending_upload_can_be_deleted(self) -> None:
        body = self.ticket(b"%PDF")
        self.store(body["doc_id"], b"%PDF-1.4")

        self.assertEqual(self.client.delete(f"/api/documents/{body['doc_id']}").status_code, 204)

        self.assertEqual(self.storage.keys(), [])
        self.assertEqual(self.complete(body["doc_id"]).status_code, 404)


class R2StorageUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.store = storage.R2Storage(self.client, "bucket")

    def test_presign_signs_a_put_for_the_key_and_content_type(self) -> None:
        self.client.generate_presigned_url.return_value = "https://signed"

        url = self.store.presign_put("users/u/d/original.pdf", "application/pdf", 600)

        self.assertEqual(url, "https://signed")
        self.client.generate_presigned_url.assert_called_once_with(
            "put_object",
            Params={"Bucket": "bucket", "Key": "users/u/d/original.pdf", "ContentType": "application/pdf"},
            ExpiresIn=600,
        )

    def test_size_heads_the_object(self) -> None:
        self.client.head_object.return_value = {"ContentLength": 1234}
        self.assertEqual(self.store.size("users/u/d/original.pdf"), 1234)
        self.client.head_object.assert_called_once_with(Bucket="bucket", Key="users/u/d/original.pdf")

    def test_size_of_missing_object_is_none(self) -> None:
        self.client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        self.assertIsNone(self.store.size("users/u/d/original.pdf"))

    def test_size_reraises_other_errors(self) -> None:
        self.client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
        with self.assertRaises(ClientError):
            self.store.size("users/u/d/original.pdf")

    def test_download_writes_to_the_path(self) -> None:
        self.store.download("users/u/d/original.pdf", "/tmp/x.pdf")
        self.client.download_file.assert_called_once_with("bucket", "users/u/d/original.pdf", "/tmp/x.pdf")


class LocalStorageUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.store = storage.LocalStorage(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_size_and_download(self) -> None:
        key = "users/u/d/original.pdf"
        self.assertIsNone(self.store.size(key))
        self.store.put(key, b"%PDF-1.4 data", "application/pdf")

        self.assertEqual(self.store.size(key), 13)
        target = os.path.join(self.root, "copy.pdf")
        self.store.download(key, target)
        with open(target, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 data")

    def test_no_presigned_urls(self) -> None:
        self.assertFalse(self.store.direct_uploads)
        with self.assertRaises(NotImplementedError):
            self.store.presign_put("users/u/d/original.pdf", "application/pdf", 600)


if __name__ == "__main__":
    unittest.main()
