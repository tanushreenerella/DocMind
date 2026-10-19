"""Rate limiting: per-route limits, key selection, Retry-After and storage fallback."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from limits import parse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core import config
from core.auth import ACCESS_TOKEN_COOKIE, create_access_token
from core.rate_limit import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_key,
)
from main import app

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _amount(limit: str) -> int:
    return parse(limit).amount


def _cookie(user_id: str) -> str:
    return f"{ACCESS_TOKEN_COOKIE}={create_access_token(user_id, f'{user_id}@example.com')}"


class FakePool:
    async def fetchrow(self, query: str, *args: object):
        if "FROM users WHERE id" in query:
            return {"id": args[0], "email": f"{args[0]}@example.com", "full_name": ""}
        if query.startswith("SELECT id FROM users WHERE email"):
            return {"id": USER_A}  # signup: email taken -> 409
        return None  # login: unknown email -> 401

    execute = AsyncMock()
    fetchval = AsyncMock(return_value=0)


class RateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        pool = FakePool()
        groq = MagicMock()
        groq.audio.transcriptions.create.return_value = "hello"
        answer = AsyncMock(return_value={"answer": "ok", "citations": [], "has_answer": True})
        self.patches = [
            patch("core.auth.get_pool", return_value=pool),
            patch("routers.auth.get_pool", return_value=pool),
            patch("routers.chat.get_pool", return_value=pool),
            patch("routers.upload.get_pool", return_value=pool),
            patch("routers.documents.get_pool", return_value=pool),
            patch("routers.chat.answer_query", answer),
            patch("routers.chat.answer_agentic_query", answer),
            patch("routers.chat._get_groq", return_value=groq),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app, headers={"Origin": config.ALLOWED_ORIGINS[0]})

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        limiter.reset()

    def _login(self, headers: dict | None = None):
        return self.client.post(
            "/api/auth/login",
            data={"username": "nobody@example.com", "password": "wrong-password"},
            headers=headers or {},
        )

    def _exhaust(self, send, limit: str) -> None:
        for i in range(_amount(limit)):
            self.assertNotEqual(send().status_code, 429, f"request {i + 1} was limited")

    def _assert_limited(self, response, max_retry_after: int = 60) -> None:
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Too many requests, try again shortly"})
        retry_after = int(response.headers["Retry-After"])
        self.assertGreaterEqual(retry_after, 1)
        self.assertLessEqual(retry_after, max_retry_after)

    # ── per-route limits ────────────────────────────────────────────────────

    def test_login_and_signup_limits(self) -> None:
        self._exhaust(self._login, config.RATE_LIMIT_AUTH)
        self._assert_limited(self._login())

        signup = lambda: self.client.post(  # noqa: E731
            "/api/auth/signup", json={"email": "x@example.com", "password": "long-enough"}
        )
        self._exhaust(signup, config.RATE_LIMIT_AUTH)
        self._assert_limited(signup())

    def test_authenticated_route_limits(self) -> None:
        cookie = {"Cookie": _cookie(USER_A)}
        chat_body = {"question": "hi", "conversation_history": []}
        routes = {
            "chat": (
                config.RATE_LIMIT_CHAT,
                lambda: self.client.post("/api/chat", json=chat_body, headers=cookie),
                60,
            ),
            "agentic": (
                config.RATE_LIMIT_AGENTIC_CHAT,
                lambda: self.client.post("/api/chat/agentic", json=chat_body, headers=cookie),
                60,
            ),
            "voice": (
                config.RATE_LIMIT_VOICE,
                lambda: self.client.post(
                    "/api/transcribe", json={"audio_base64": "aGk="}, headers=cookie
                ),
                60,
            ),
            "upload": (
                config.RATE_LIMIT_UPLOAD,
                lambda: self.client.post(
                    "/api/upload",
                    files={"files": ("notes.txt", b"not a pdf", "text/plain")},
                    headers=cookie,
                ),
                3600,
            ),
        }
        for name, (limit, send, max_retry_after) in routes.items():
            with self.subTest(route=name):
                first = send()
                self.assertIn(first.status_code, (200, 400), first.text)
                for _ in range(_amount(limit) - 1):
                    send()
                self._assert_limited(send(), max_retry_after)

    def test_default_limit_applies_to_undecorated_routes(self) -> None:
        cookie = {"Cookie": _cookie(USER_A)}
        send = lambda: self.client.get("/api/documents/count", headers=cookie)  # noqa: E731
        self._exhaust(send, f"{config.RATE_LIMIT_PER_MINUTE}/minute")
        self._assert_limited(send())

    def test_health_is_exempt(self) -> None:
        for _ in range(config.RATE_LIMIT_PER_MINUTE + 5):
            self.assertEqual(self.client.get("/health").status_code, 200)

    # ── key selection ───────────────────────────────────────────────────────

    def test_users_have_separate_buckets(self) -> None:
        body = {"question": "hi", "conversation_history": []}
        send_a = lambda: self.client.post(  # noqa: E731
            "/api/chat", json=body, headers={"Cookie": _cookie(USER_A)}
        )
        self._exhaust(send_a, config.RATE_LIMIT_CHAT)
        self._assert_limited(send_a())

        response_b = self.client.post("/api/chat", json=body, headers={"Cookie": _cookie(USER_B)})
        self.assertEqual(response_b.status_code, 200)

    def test_first_forwarded_ip_is_the_key(self) -> None:
        via_proxy_1 = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
        via_proxy_2 = {"X-Forwarded-For": "203.0.113.7, 10.0.0.2"}
        self._exhaust(lambda: self._login(via_proxy_1), config.RATE_LIMIT_AUTH)

        # Same client IP through a different proxy hop shares the bucket.
        self._assert_limited(self._login(via_proxy_2))
        # A different client IP, and a direct (no-XFF) caller, do not.
        self.assertEqual(self._login({"X-Forwarded-For": "198.51.100.9"}).status_code, 401)
        self.assertEqual(self._login().status_code, 401)

    def test_logged_in_user_is_keyed_by_user_not_ip(self) -> None:
        shared_ip = {"X-Forwarded-For": "203.0.113.7"}
        self._exhaust(lambda: self._login(shared_ip), config.RATE_LIMIT_AUTH)

        with_cookie = {**shared_ip, "Cookie": _cookie(USER_A)}
        self.assertEqual(self._login(with_cookie).status_code, 401)

    def test_invalid_cookie_falls_back_to_ip(self) -> None:
        shared_ip = {"X-Forwarded-For": "203.0.113.7"}
        self._exhaust(lambda: self._login(shared_ip), config.RATE_LIMIT_AUTH)

        forged = {**shared_ip, "Cookie": f"{ACCESS_TOKEN_COOKIE}=not-a-jwt"}
        self._assert_limited(self._login(forged))

    def test_key_falls_back_to_peer_ip(self) -> None:
        request = Request({
            "type": "http",
            "headers": [],
            "client": ("192.0.2.5", 1234),
        })
        self.assertEqual(rate_limit_key(request), "ip:192.0.2.5")


class StorageTests(unittest.TestCase):
    def test_app_limiter_uses_configured_storage_with_memory_fallback(self) -> None:
        self.assertEqual(limiter._storage_uri, config.RATE_LIMIT_STORAGE_URI)
        self.assertTrue(limiter._in_memory_fallback_enabled)

    def test_unreachable_storage_falls_back_to_memory(self) -> None:
        # Nothing listens on port 1, so every storage call fails.
        fallback_limiter = Limiter(
            key_func=lambda: "client",
            storage_uri="redis://127.0.0.1:1/0",
            in_memory_fallback_enabled=True,
        )
        mini = FastAPI()
        mini.state.limiter = fallback_limiter
        mini.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        mini.add_middleware(SlowAPIMiddleware)

        @mini.get("/ping")
        @fallback_limiter.limit("2/minute")
        async def ping(request: Request) -> dict:
            return {"ok": True}

        client = TestClient(mini)
        statuses = [client.get("/ping").status_code for _ in range(3)]
        self.assertEqual(statuses, [200, 200, 429])


if __name__ == "__main__":
    unittest.main()
