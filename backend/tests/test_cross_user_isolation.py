"""Cross-user isolation: user B uploads a document; user A must not reach it anywhere."""

import os
import re
import shutil
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core import config
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from main import app
from services import storage
from tests import PROXY_HEADERS
from tests.fakes import FakeStorage

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
B_FILENAME = "b-private-contract.pdf"
B_PDF = b"%PDF-1.4 user B's confidential bytes"
B_IMAGE = b"\xff\xd8\xff user B's page image"


class FakeDB:
    """In-memory stand-in for the users/documents/chat_history tables.

    WHERE clauses are evaluated from the query text itself (``col = $n``,
    ``col = 'literal'`` and ``col IN ('a', 'b')``), so a route that forgets its ``user_id`` filter really
    does see other users' rows here, just as it would against Postgres.
    """

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {
            "users": [
                {"id": uid, "email": f"{uid[0]}@example.com", "full_name": uid[0].upper()}
                for uid in (USER_A, USER_B)
            ],
            "documents": [],
            "chat_history": [],
        }

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.split())

    def _select(self, query: str, args: tuple) -> list[dict]:
        q = self._normalize(query)
        table = re.search(r"FROM (\w+)", q).group(1)
        where = q.split(" WHERE ", 1)[1].split(" ORDER BY ")[0] if " WHERE " in q else ""
        params = re.findall(r"(\w+) = \$(\d+)", where)
        literals = re.findall(r"(\w+) = '([^']*)'", where)
        in_lists = [
            (col, re.findall(r"'([^']*)'", values))
            for col, values in re.findall(r"(\w+) IN \(([^)]*)\)", where)
        ]
        return [
            row
            for row in self.tables[table]
            if all(row.get(col) == args[int(n) - 1] for col, n in params)
            and all(row.get(col) == value for col, value in literals)
            and all(row.get(col) in values for col, values in in_lists)
        ]

    async def fetchrow(self, query: str, *args: object):
        rows = self._select(query, args)
        return rows[0] if rows else None

    async def fetch(self, query: str, *args: object):
        return self._select(query, args)

    async def fetchval(self, query: str, *args: object):
        rows = self._select(query, args)
        if "COUNT(*)" in query:
            return len(rows)
        if "SELECT 1 " in self._normalize(query):
            return 1 if rows else None
        raise AssertionError(f"unexpected fetchval: {query}")

    async def execute(self, query: str, *args: object):
        q = self._normalize(query)
        if q.startswith("INSERT INTO documents"):
            doc_id, user_id, doc_name = args
            status, stage = re.search(r"VALUES \(\$1, \$2, \$3, '(\w+)', '(\w+)', 0\)", q).groups()
            self.tables["documents"].append({
                "doc_id": doc_id, "user_id": user_id, "doc_name": doc_name,
                "status": status, "stage": stage, "progress": 0,
                "document_type": None, "sensitivity_level": None, "summary": None,
                "page_count": None, "error_message": None,
            })
        elif q.startswith("UPDATE documents SET"):
            set_part, where = q[len("UPDATE documents SET "):].split(" WHERE ", 1)
            changes = {col: args[int(n) - 1] for col, n in re.findall(r"(\w+) = \$(\d+)", set_part)}
            changes.update(re.findall(r"(\w+) = '([^']*)'", set_part))
            rows = self._select(f"SELECT * FROM documents WHERE {where}", args)
            for row in rows:
                row.update(changes)
            return f"UPDATE {len(rows)}"
        elif q.startswith("INSERT INTO chat_history"):
            user_id, doc_id, content = args
            role = re.search(r"'(user|assistant)'", q).group(1)
            self.tables["chat_history"].append({
                "user_id": user_id, "doc_id": doc_id, "role": role,
                "content": content, "created_at": datetime.now(timezone.utc),
            })
        elif q.startswith("DELETE FROM"):
            table = q.split()[2]
            doomed = self._select(q.replace("DELETE FROM", "SELECT * FROM", 1), args)
            self.tables[table] = [r for r in self.tables[table] if r not in doomed]
        else:
            raise AssertionError(f"unexpected execute: {query}")


class CrossUserIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        self.db = FakeDB()
        self.storage = FakeStorage()
        self.temp_root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.temp_root, "temp"))

        async def parse(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
            # Stand-in for pdfplumber: one page with a rendered image on disk.
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"{doc_id}_p1.jpg"
            with open(os.path.join(image_dir, image_filename), "wb") as f:
                f.write(B_IMAGE)
            return [{
                "page_number": 1, "text": "confidential terms",
                "image_path": os.path.join(image_dir, image_filename),
                "image_filename": image_filename, "has_tables": False, "tables": [],
            }]

        self.answer = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.agentic = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.timeline = AsyncMock(
            return_value={"briefing": "b", "citations": [], "has_timeline": True}
        )
        self.delete_chunks = MagicMock()
        self.patches = [
            patch(f"{module}.get_pool", return_value=self.db)
            for module in ("core.auth", "routers.chat", "routers.upload", "routers.documents")
        ] + [
            patch.object(storage, "_storage", self.storage),
            patch("routers.upload.STORAGE_PATH", self.temp_root),
            patch("routers.upload.parse_document", parse),
            # The test PDFs are just a header; real page counting is tested elsewhere.
            patch("routers.upload.count_pdf_pages", return_value=1),
            patch("routers.upload.classify_document", AsyncMock(return_value={})),
            patch("routers.upload.index_document", AsyncMock()),
            patch("routers.chat.answer_query", self.answer),
            patch("routers.chat.answer_agentic_query", self.agentic),
            patch("routers.documents.build_evidence_timeline", self.timeline),
            patch("routers.documents.delete_document_chunks", self.delete_chunks),
        ]
        for p in self.patches:
            p.start()

        self.client_a = self._client(USER_A)
        self.client_b = self._client(USER_B)

        self.doc_id = self._upload(self.client_b, B_FILENAME, B_PDF)
        self.assertEqual(self.db.tables["documents"][0]["status"], "complete")
        self.b_prefix = storage.document_prefix(USER_B, self.doc_id)
        self.assertEqual(len(self.storage.keys(self.b_prefix)), 2)

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.temp_root, ignore_errors=True)
        limiter.reset()

    def _presign(self, client: TestClient, filename: str, data: bytes) -> dict:
        response = client.post("/api/uploads/presign", json={
            "filename": filename, "size": len(data), "content_type": "application/pdf",
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _upload(self, client: TestClient, filename: str, data: bytes) -> str:
        """presign → PUT (local-disk route) → complete; returns the doc_id."""
        ticket = self._presign(client, filename, data)
        put = client.put(ticket["upload_url"], content=data, headers=ticket["headers"])
        self.assertEqual(put.status_code, 204, put.text)
        complete = client.post(f"/api/uploads/{ticket['doc_id']}/complete")
        self.assertEqual(complete.status_code, 200, complete.text)
        return ticket["doc_id"]

    @staticmethod
    def _client(user_id: str) -> TestClient:
        token = create_access_token(user_id, f"{user_id[0]}@example.com")
        return TestClient(app, headers={
            **PROXY_HEADERS,
            "Origin": config.ALLOWED_ORIGINS[0],
            "Cookie": f"{ACCESS_TOKEN_COOKIE}={token}",
        })

    def _doc_endpoints(self, doc_id: str) -> dict:
        """Every route that takes a doc_id (or a filename derived from one)."""
        chat_body = {"question": "What does it say?", "conversation_history": [], "doc_id": doc_id}
        return {
            "chat": ("POST", "/api/chat", chat_body),
            "agentic chat": ("POST", "/api/chat/agentic", chat_body),
            "chat history": ("GET", f"/api/chat/history/{doc_id}", None),
            "status": ("GET", f"/api/status/{doc_id}", None),
            "pdf": ("GET", f"/api/pdf/{doc_id}", None),
            "page image": ("GET", f"/api/page-image/{doc_id}_p1.jpg", None),
            "timeline": ("POST", f"/api/documents/{doc_id}/evidence-timeline", None),
        }

    @staticmethod
    def _send(client: TestClient, method: str, path: str, body: dict | None):
        return client.request(method, path, json=body)

    def test_owner_can_reach_every_endpoint(self) -> None:
        # Control: proves the refusals below come from ownership, not a broken setup.
        for name, (method, path, body) in self._doc_endpoints(self.doc_id).items():
            with self.subTest(endpoint=name):
                response = self._send(self.client_b, method, path, body)
                self.assertEqual(response.status_code, 200, response.text)

        self.assertEqual(self.client_b.get(f"/api/pdf/{self.doc_id}").content, B_PDF)
        self.assertEqual(
            self.client_b.get(f"/api/page-image/{self.doc_id}_p1.jpg").content, B_IMAGE
        )

    def test_file_responses_are_never_shared_caches(self) -> None:
        pdf = self.client_b.get(f"/api/pdf/{self.doc_id}")
        image = self.client_b.get(f"/api/page-image/{self.doc_id}_p1.jpg")

        self.assertEqual(pdf.headers["cache-control"], "private, no-store")
        self.assertEqual(image.headers["cache-control"], "private, max-age=300")
        self.assertEqual(image.headers["content-type"], "image/jpeg")
        self.assertEqual(pdf.headers["content-type"], "application/pdf")

    def test_other_user_is_refused_on_every_endpoint(self) -> None:
        for name, (method, path, body) in self._doc_endpoints(self.doc_id).items():
            with self.subTest(endpoint=name):
                response = self._send(self.client_a, method, path, body)
                self.assertIn(response.status_code, (403, 404), response.text)
                self.assertNotIn(B_FILENAME.encode(), response.content)
                self.assertNotIn(b"confidential", response.content)
                self.assertNotIn(b"page image", response.content)

        # Nothing downstream ran for A, and A's attempts left no history on B's doc.
        self.answer.assert_not_awaited()
        self.agentic.assert_not_awaited()
        self.timeline.assert_not_awaited()
        self.assertEqual(self.db.tables["chat_history"], [])

    def test_foreign_doc_is_indistinguishable_from_unknown_doc(self) -> None:
        unknown = str(uuid.uuid4())
        foreign = self._doc_endpoints(self.doc_id)
        for name, (method, path, body) in self._doc_endpoints(unknown).items():
            with self.subTest(endpoint=name):
                expected = self._send(self.client_a, method, path, body)
                actual = self._send(self.client_a, *foreign[name])
                self.assertEqual(
                    (actual.status_code, actual.json()), (expected.status_code, expected.json())
                )

    def test_document_list_and_count_exclude_other_users(self) -> None:
        listed_a = self.client_a.get("/api/documents")
        self.assertEqual(listed_a.status_code, 200)
        self.assertEqual(listed_a.json(), {"documents": []})
        self.assertEqual(self.client_a.get("/api/documents/count").json(), {"count": 0})

        listed_b = self.client_b.get("/api/documents").json()["documents"]
        self.assertEqual([d["doc_id"] for d in listed_b], [self.doc_id])
        self.assertEqual(self.client_b.get("/api/documents/count").json(), {"count": 1})

    # ── status values ───────────────────────────────────────────────────────

    def _set_doc(self, **fields) -> None:
        self.db.tables["documents"][0].update(fields)

    def test_partial_documents_are_listed_and_usable(self) -> None:
        self._set_doc(status="partial", failed_pages=[2, 5])

        listed = self.client_b.get("/api/documents").json()["documents"]
        self.assertEqual([d["doc_id"] for d in listed], [self.doc_id])
        self.assertEqual(self.client_b.get("/api/documents/count").json(), {"count": 1})
        timeline = self.client_b.post(f"/api/documents/{self.doc_id}/evidence-timeline")
        self.assertEqual(timeline.status_code, 200)

        status = self.client_b.get(f"/api/status/{self.doc_id}").json()
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["failed_pages"], [2, 5])
        self.assertIsNotNone(status["classification"])

    def test_unfinished_and_failed_documents_are_not_listed(self) -> None:
        for state in ("queued", "processing", "error"):
            with self.subTest(status=state):
                self._set_doc(status=state)
                self.assertEqual(self.client_b.get("/api/documents").json(), {"documents": []})
                self.assertEqual(self.client_b.get("/api/documents/count").json(), {"count": 0})

    def test_status_never_returns_the_internal_error(self) -> None:
        self._set_doc(
            status="error", stage="error",
            error_message="ConnectError: https://internal-host:8443 refused (key=abc123)",
        )

        response = self.client_b.get(f"/api/status/{self.doc_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")
        self.assertNotIn("error", response.json())
        self.assertNotIn(b"internal-host", response.content)
        self.assertNotIn(b"abc123", response.content)
        self.assertEqual(response.json()["failed_pages"], [])

    # ── delete ──────────────────────────────────────────────────────────────

    def test_other_user_cannot_delete(self) -> None:
        response = self.client_a.delete(f"/api/documents/{self.doc_id}")

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(B_FILENAME.encode(), response.content)
        # Nothing of B's was touched...
        self.delete_chunks.assert_not_called()
        self.assertEqual(len(self.storage.keys(self.b_prefix)), 2)
        self.assertEqual(len(self.db.tables["documents"]), 1)
        # ...and B can still use it.
        self.assertEqual(self.client_b.get(f"/api/pdf/{self.doc_id}").content, B_PDF)

    def test_delete_of_foreign_doc_looks_like_unknown_doc(self) -> None:
        unknown = self.client_a.delete(f"/api/documents/{uuid.uuid4()}")
        foreign = self.client_a.delete(f"/api/documents/{self.doc_id}")
        self.assertEqual(
            (foreign.status_code, foreign.json()), (unknown.status_code, unknown.json())
        )

    def test_owner_delete_removes_chunks_files_and_rows(self) -> None:
        chat_body = {"question": "q", "conversation_history": [], "doc_id": self.doc_id}
        self.client_b.post("/api/chat", json=chat_body)
        self.assertEqual(len(self.db.tables["chat_history"]), 2)

        response = self.client_b.delete(f"/api/documents/{self.doc_id}")

        self.assertEqual(response.status_code, 204, response.text)
        self.delete_chunks.assert_called_once_with(self.doc_id, USER_B)
        self.assertEqual(self.storage.keys(), [])
        self.assertEqual(self.db.tables["documents"], [])
        self.assertEqual(self.db.tables["chat_history"], [])
        for method, path, body in self._doc_endpoints(self.doc_id).values():
            self.assertIn(self._send(self.client_b, method, path, body).status_code, (403, 404))
        self.assertEqual(self.client_b.get("/api/documents").json(), {"documents": []})
        # A second delete is a plain 404.
        self.assertEqual(self.client_b.delete(f"/api/documents/{self.doc_id}").status_code, 404)

    def test_delete_leaves_other_documents_alone(self) -> None:
        second = self._upload(self.client_b, "second.pdf", b"%PDF-1.4 second")

        self.assertEqual(self.client_b.delete(f"/api/documents/{self.doc_id}").status_code, 204)

        self.assertEqual(self.storage.keys(self.b_prefix), [])
        self.assertEqual(len(self.storage.keys(storage.document_prefix(USER_B, second))), 2)
        self.assertEqual(
            self.client_b.get(f"/api/pdf/{second}").content, b"%PDF-1.4 second"
        )

    def test_delete_while_processing_is_refused(self) -> None:
        self.db.tables["documents"][0]["status"] = "processing"

        response = self.client_b.delete(f"/api/documents/{self.doc_id}")

        self.assertEqual(response.status_code, 409)
        self.delete_chunks.assert_not_called()
        self.assertEqual(len(self.storage.keys(self.b_prefix)), 2)

    # ── pending uploads ─────────────────────────────────────────────────────

    def test_other_user_cannot_fill_or_complete_a_pending_upload(self) -> None:
        ticket = self._presign(self.client_b, "pending.pdf", b"%PDF-1.4 pending")
        doc_id = ticket["doc_id"]
        unknown = str(uuid.uuid4())

        for path, send in (
            ("/api/uploads/{}/file", lambda p: self.client_a.put(
                p, content=b"%PDF-1.4 from A", headers={"Content-Type": "application/pdf"})),
            ("/api/uploads/{}/complete", lambda p: self.client_a.post(p)),
        ):
            with self.subTest(path=path):
                foreign = send(path.format(doc_id))
                expected = send(path.format(unknown))
                self.assertEqual(foreign.status_code, 404, foreign.text)
                self.assertEqual(
                    (foreign.status_code, foreign.json()), (expected.status_code, expected.json())
                )

        # Nothing landed under either user, and B's upload is still waiting for B.
        self.assertEqual(self.storage.keys(storage.document_prefix(USER_B, doc_id)), [])
        self.assertEqual(self.storage.keys(f"users/{USER_A}/"), [])
        put = self.client_b.put(ticket["upload_url"], content=b"%PDF-1.4 pending", headers=ticket["headers"])
        self.assertEqual(put.status_code, 204)

    def test_owner_history_is_not_visible_to_other_user(self) -> None:
        chat_body = {"question": "secret question", "conversation_history": [], "doc_id": self.doc_id}
        self.assertEqual(self.client_b.post("/api/chat", json=chat_body).status_code, 200)
        history_b = self.client_b.get(f"/api/chat/history/{self.doc_id}").json()["history"]
        self.assertEqual(len(history_b), 2)

        response = self.client_a.get(f"/api/chat/history/{self.doc_id}")
        self.assertIn(response.status_code, (403, 404))
        self.assertNotIn(b"secret question", response.content)


if __name__ == "__main__":
    unittest.main()
