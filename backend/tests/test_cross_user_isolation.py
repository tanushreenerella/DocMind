"""Cross-user isolation: user B uploads a document; user A must not reach it anywhere."""

import os
import re
import shutil
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core import config
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from main import app
from tests import PROXY_HEADERS

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
B_FILENAME = "b-private-contract.pdf"
B_PDF = b"%PDF-1.4 user B's confidential bytes"
B_IMAGE = b"\xff\xd8\xff user B's page image"


class FakeDB:
    """In-memory stand-in for the users/documents/chat_history tables.

    WHERE clauses are evaluated from the query text itself (``col = $n`` and
    ``col = 'literal'``), so a route that forgets its ``user_id`` filter really
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
        return [
            row
            for row in self.tables[table]
            if all(row.get(col) == args[int(n) - 1] for col, n in params)
            and all(row.get(col) == value for col, value in literals)
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
            self.tables["documents"].append({
                "doc_id": doc_id, "user_id": user_id, "doc_name": doc_name,
                "status": "queued", "stage": "queued", "progress": 0,
                "document_type": None, "sensitivity_level": None, "summary": None,
                "page_count": None, "error_message": None,
            })
        elif q.startswith("UPDATE documents SET"):
            doc_id, *values = args
            columns = re.findall(r"(\w+) = \$\d+", q.split(" WHERE ")[0])
            for row in self.tables["documents"]:
                if row["doc_id"] == doc_id:
                    row.update(zip(columns, values))
        elif q.startswith("INSERT INTO chat_history"):
            user_id, doc_id, content = args
            role = re.search(r"'(user|assistant)'", q).group(1)
            self.tables["chat_history"].append({
                "user_id": user_id, "doc_id": doc_id, "role": role,
                "content": content, "created_at": datetime.now(timezone.utc),
            })
        else:
            raise AssertionError(f"unexpected execute: {query}")


class CrossUserIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        self.db = FakeDB()
        self.storage = tempfile.mkdtemp()
        self.images = os.path.join(self.storage, "page_images")
        os.makedirs(os.path.join(self.storage, "temp"))
        os.makedirs(self.images)

        async def process(doc_id: str, user_id: str, file_path: str, filename: str) -> None:
            # Stand-in for parse/classify/index: leave the same artifacts behind.
            pdf_dir = os.path.join(self.storage, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(pdf_dir, f"{doc_id}.pdf"))
            with open(os.path.join(self.images, f"{doc_id}_p1.jpg"), "wb") as f:
                f.write(B_IMAGE)
            await self.db.execute(
                "UPDATE documents SET status = $2, stage = $3, progress = $4 WHERE doc_id = $1",
                doc_id, "complete", "complete", 100,
            )

        self.answer = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.agentic = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.timeline = AsyncMock(
            return_value={"briefing": "b", "citations": [], "has_timeline": True}
        )
        self.patches = [
            patch(f"{module}.get_pool", return_value=self.db)
            for module in ("core.auth", "routers.chat", "routers.upload", "routers.documents")
        ] + [
            patch("routers.upload.STORAGE_PATH", self.storage),
            patch("routers.upload.IMAGES_PATH", self.images),
            patch("routers.upload._process_document", process),
            patch("routers.chat.answer_query", self.answer),
            patch("routers.chat.answer_agentic_query", self.agentic),
            patch("routers.documents.build_evidence_timeline", self.timeline),
        ]
        for p in self.patches:
            p.start()

        self.client_a = self._client(USER_A)
        self.client_b = self._client(USER_B)

        upload = self.client_b.post(
            "/api/upload", files={"files": (B_FILENAME, B_PDF, "application/pdf")}
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        self.doc_id = upload.json()["jobs"][0]["doc_id"]
        self.assertEqual(self.db.tables["documents"][0]["status"], "complete")

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.storage, ignore_errors=True)
        limiter.reset()

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
