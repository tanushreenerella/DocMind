"""Personal access tokens: create/list/revoke, Bearer auth, and never storing the secret."""

import hashlib
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core import auth as core_auth
from core import config, database, token_budget
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import limiter
from core.token_budget import MemoryStore
from main import app
from tests import PROXY_HEADERS

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TOKEN_FORMAT = re.compile(r"dm_[A-Za-z0-9_-]{43}")


class FakeTokenDB:
    """users plus personal_access_tokens, answering exactly the queries the code makes."""

    def __init__(self) -> None:
        self.users = {u: {"id": u, "email": f"{u[:1]}@example.com", "full_name": ""} for u in (USER_A, USER_B)}
        self.tokens: list[dict] = []
        # Every argument any query was given, to prove the secret never reaches the DB.
        self.all_args: list[object] = []
        self._clock = datetime(2026, 10, 27, 9, 0, tzinfo=timezone.utc)

    def _now(self) -> datetime:
        self._clock += timedelta(seconds=1)
        return self._clock

    @staticmethod
    def _q(query: str) -> str:
        return " ".join(query.split())

    async def fetchrow(self, query: str, *args):
        self.all_args.extend(args)
        q = self._q(query)
        if q.startswith("SELECT id, email, full_name FROM users WHERE id = $1"):
            return self.users.get(args[0])
        if q.startswith("INSERT INTO personal_access_tokens"):
            token_id, user_id, name, token_hash, prefix = args
            row = {
                "id": token_id, "user_id": user_id, "name": name, "token_hash": token_hash,
                "prefix": prefix, "created_at": self._now(), "last_used_at": None, "revoked_at": None,
            }
            self.tokens.append(row)
            return row
        if "FROM personal_access_tokens t JOIN users u ON u.id::text = t.user_id" in q:
            assert "t.revoked_at IS NULL" in q
            for t in self.tokens:
                if t["token_hash"] == args[0] and t["revoked_at"] is None and t["user_id"] in self.users:
                    return {"token_id": t["id"], **self.users[t["user_id"]]}
            return None
        raise AssertionError(f"unexpected fetchrow: {q}")

    async def fetch(self, query: str, *args):
        self.all_args.extend(args)
        q = self._q(query)
        assert q.startswith("SELECT id, name, prefix, created_at, last_used_at, revoked_at FROM personal_access_tokens WHERE user_id = $1"), q
        assert "token_hash" not in q
        return sorted((t for t in self.tokens if t["user_id"] == args[0]), key=lambda t: t["created_at"], reverse=True)

    async def fetchval(self, query: str, *args):
        self.all_args.extend(args)
        q = self._q(query)
        if q.startswith("SELECT 1 FROM personal_access_tokens WHERE id = $1 AND user_id = $2"):
            return 1 if any(t["id"] == args[0] and t["user_id"] == args[1] for t in self.tokens) else None
        if q.startswith("SELECT 1 FROM documents"):
            return 1
        raise AssertionError(f"unexpected fetchval: {q}")

    async def execute(self, query: str, *args):
        self.all_args.extend(args)
        q = self._q(query)
        if q.startswith("UPDATE personal_access_tokens SET last_used_at = now() WHERE id = $1"):
            for t in self.tokens:
                if t["id"] == args[0]:
                    t["last_used_at"] = self._now()
        elif q.startswith("UPDATE personal_access_tokens SET revoked_at = now()"):
            assert "AND user_id = $2 AND revoked_at IS NULL" in q
            for t in self.tokens:
                if t["id"] == args[0] and t["user_id"] == args[1] and t["revoked_at"] is None:
                    t["revoked_at"] = self._now()
        elif q.startswith("INSERT INTO chat_history"):
            pass
        else:
            raise AssertionError(f"unexpected execute: {q}")


def _session(user_id: str) -> dict:
    return {"Cookie": f"{ACCESS_TOKEN_COOKIE}={create_access_token(user_id, f'{user_id[:1]}@example.com')}"}


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class _TokenTestCase(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        self.db = FakeTokenDB()
        self.patches = [
            patch("core.auth.get_pool", return_value=self.db),
            patch("routers.tokens.get_pool", return_value=self.db),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app, headers={**PROXY_HEADERS, "Origin": config.ALLOWED_ORIGINS[0]})
        # What a script sends: no Origin header at all.
        self.script = TestClient(app, headers=PROXY_HEADERS)

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        limiter.reset()

    def _create(self, name: str = "CI", user: str = USER_A) -> dict:
        response = self.client.post("/api/tokens", json={"name": name}, headers=_session(user))
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()


class CreateTokenTests(_TokenTestCase):
    def test_returns_the_token_once_in_the_expected_format(self) -> None:
        response = self.client.post("/api/tokens", json={"name": "  laptop script  "}, headers=_session(USER_A))

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        body = response.json()
        self.assertRegex(body["token"], TOKEN_FORMAT)
        self.assertEqual(body["name"], "laptop script")
        self.assertEqual(body["prefix"], body["token"][:11])
        self.assertIsNone(body["last_used_at"])
        self.assertIsNone(body["revoked_at"])
        self.assertNotIn("token_hash", body)

    def test_token_is_never_stored_in_plaintext(self) -> None:
        token = self._create()["token"]

        (row,) = self.db.tokens
        self.assertEqual(row["token_hash"], hashlib.sha256(token.encode()).hexdigest())
        for value in row.values():
            self.assertNotEqual(value, token)
            if isinstance(value, str):
                self.assertNotIn(token[len(row["prefix"]):], value)  # not even the secret part
        # Nor did any query ever receive it, including authenticating with it.
        self.client.get("/api/auth/me", headers=_bearer(token))
        self.assertNotIn(token, [a for a in self.db.all_args if isinstance(a, str)])

    def test_each_token_is_unique_and_random(self) -> None:
        tokens = {core_auth.generate_personal_access_token() for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        self.assertTrue(all(TOKEN_FORMAT.fullmatch(t) for t in tokens))

    def test_name_is_required_and_bounded(self) -> None:
        for name in ("", "   ", "x" * 101):
            with self.subTest(name=name[:5]):
                response = self.client.post("/api/tokens", json={"name": name}, headers=_session(USER_A))
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.db.tokens, [])


class ListTokenTests(_TokenTestCase):
    def test_lists_own_tokens_newest_first_without_secrets(self) -> None:
        first = self._create("first")
        second = self._create("second")
        self._create("someone else's", user=USER_B)

        response = self.client.get("/api/tokens", headers=_session(USER_A))

        self.assertEqual(response.status_code, 200)
        listed = response.json()["tokens"]
        self.assertEqual([t["name"] for t in listed], ["second", "first"])
        self.assertEqual(
            set(listed[0]), {"id", "name", "prefix", "created_at", "last_used_at", "revoked_at"}
        )
        for secret in (first["token"], second["token"]):
            self.assertNotIn(secret, response.text)
        for row in self.db.tokens:
            self.assertNotIn(row["token_hash"], response.text)


class BearerAuthTests(_TokenTestCase):
    def test_bearer_token_authenticates_and_records_last_use(self) -> None:
        token = self._create()["token"]

        response = self.client.get("/api/auth/me", headers=_bearer(token))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], USER_A)
        listed = self.client.get("/api/tokens", headers=_session(USER_A)).json()["tokens"]
        self.assertIsNotNone(listed[0]["last_used_at"])

    def test_revoked_token_is_rejected(self) -> None:
        created = self._create()
        self.assertEqual(self.client.get("/api/auth/me", headers=_bearer(created["token"])).status_code, 200)

        revoke = self.client.delete(f"/api/tokens/{created['id']}", headers=_session(USER_A))

        self.assertEqual(revoke.status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me", headers=_bearer(created["token"])).status_code, 401)
        listed = self.client.get("/api/tokens", headers=_session(USER_A)).json()["tokens"]
        self.assertIsNotNone(listed[0]["revoked_at"])

    def test_unknown_malformed_and_jwt_bearers_are_rejected(self) -> None:
        self._create()
        jwt = create_access_token(USER_A, "a@example.com")
        for bearer in ("dm_" + "A" * 43, "dm_", "not-a-token", jwt, ""):
            with self.subTest(bearer=bearer[:12]):
                self.assertEqual(self.client.get("/api/auth/me", headers=_bearer(bearer)).status_code, 401)

    def test_a_bad_bearer_is_not_rescued_by_a_valid_cookie(self) -> None:
        headers = {**_session(USER_A), **_bearer("dm_" + "A" * 43)}
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 401)

    def test_token_of_a_deleted_user_is_rejected(self) -> None:
        token = self._create()["token"]
        del self.db.users[USER_A]
        self.assertEqual(self.client.get("/api/auth/me", headers=_bearer(token)).status_code, 401)

    def test_tokens_cannot_manage_tokens(self) -> None:
        created = self._create()
        bearer = _bearer(created["token"])

        self.assertEqual(self.script.get("/api/tokens", headers=bearer).status_code, 401)
        self.assertEqual(self.script.post("/api/tokens", json={"name": "x"}, headers=bearer).status_code, 401)
        self.assertEqual(self.script.delete(f"/api/tokens/{created['id']}", headers=bearer).status_code, 401)
        self.assertEqual(len(self.db.tokens), 1)
        self.assertIsNone(self.db.tokens[0]["revoked_at"])


class RevokeTokenTests(_TokenTestCase):
    def test_revoking_twice_is_a_no_op(self) -> None:
        created = self._create()
        self.client.delete(f"/api/tokens/{created['id']}", headers=_session(USER_A))
        revoked_at = self.db.tokens[0]["revoked_at"]

        again = self.client.delete(f"/api/tokens/{created['id']}", headers=_session(USER_A))

        self.assertEqual(again.status_code, 204)
        self.assertEqual(self.db.tokens[0]["revoked_at"], revoked_at)

    def test_cannot_revoke_another_users_token(self) -> None:
        created = self._create(user=USER_B)

        response = self.client.delete(f"/api/tokens/{created['id']}", headers=_session(USER_A))

        self.assertEqual(response.status_code, 404)
        self.assertIsNone(self.db.tokens[0]["revoked_at"])
        self.assertEqual(self.client.get("/api/auth/me", headers=_bearer(created["token"])).status_code, 200)

    def test_unknown_token_is_404(self) -> None:
        self.assertEqual(self.client.delete("/api/tokens/nope", headers=_session(USER_A)).status_code, 404)


class OriginCheckTests(_TokenTestCase):
    """Scripts send no Origin; a token exempts them from the cookie CSRF check, nothing else does."""

    def setUp(self) -> None:
        super().setUp()
        answer = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.chat_patches = [
            patch("routers.chat.get_pool", return_value=self.db),
            patch("routers.chat.answer_query", answer),
            patch.object(token_budget, "_stores", [MemoryStore()]),
        ]
        for p in self.chat_patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.chat_patches:
            p.stop()
        super().tearDown()

    def _chat(self, headers: dict):
        return self.script.post("/api/chat", json={"question": "What is the total?"}, headers=headers)

    def test_post_with_a_token_and_no_origin_works(self) -> None:
        response = self._chat(_bearer(self._create()["token"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")

    def test_cookie_post_without_origin_is_still_refused(self) -> None:
        self.assertEqual(self._chat(_session(USER_A)).status_code, 403)

    def test_non_token_bearer_without_origin_is_still_refused(self) -> None:
        jwt = create_access_token(USER_A, "a@example.com")
        self.assertEqual(self._chat({**_session(USER_A), **_bearer(jwt)}).status_code, 403)

    def test_invalid_token_without_origin_gets_no_further_than_401(self) -> None:
        self.assertEqual(self._chat({**_session(USER_A), **_bearer("dm_" + "A" * 43)}).status_code, 401)


class SchemaTests(unittest.TestCase):
    def test_table_keeps_a_hash_and_no_token_column(self) -> None:
        (create,) = [s for s in database.SCHEMA_UPDATES if "TABLE IF NOT EXISTS personal_access_tokens" in s]
        columns = {
            part.split()[0]
            for part in create.split("(", 1)[1].rsplit(")", 1)[0].split(",")
        }
        self.assertEqual(
            columns,
            {"id", "user_id", "name", "token_hash", "prefix", "created_at", "last_used_at", "revoked_at"},
        )
        self.assertIn("token_hash TEXT NOT NULL UNIQUE", create)


if __name__ == "__main__":
    unittest.main()
