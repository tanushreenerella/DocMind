"""Cookie-based auth and the Origin check on state-changing requests."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import auth as core_auth
from core.rate_limit import limiter
from main import app
from tests import PROXY_HEADERS

USER_ID = "11111111-1111-1111-1111-111111111111"
EMAIL = "user@example.com"
PASSWORD = "correct-horse"


class FakePool:
    """Answers the handful of queries the auth routes make."""

    def __init__(self) -> None:
        self.hashed_password = core_auth.hash_password(PASSWORD)

    async def fetchrow(self, query: str, *args: object):
        row = {
            "id": USER_ID,
            "email": EMAIL,
            "full_name": "Test User",
            "hashed_password": self.hashed_password,
        }
        if query.startswith("SELECT id FROM users WHERE email"):
            return None  # signup: email not taken yet
        if "WHERE email" in query:
            return row if args[0] == EMAIL else None
        if "WHERE id" in query:
            return row if args[0] == USER_ID else None
        if query.strip().startswith("INSERT INTO users"):
            return row
        raise AssertionError(f"unexpected query: {query}")


class AuthCookieTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests log in repeatedly; keep them clear of the login rate limit.
        limiter.reset()
        pool = FakePool()
        self.patches = [
            patch("routers.auth.get_pool", return_value=pool),
            patch("core.auth.get_pool", return_value=pool),
            patch.object(core_auth, "COOKIE_SECURE", True),
        ]
        for p in self.patches:
            p.start()
        self.origin = core_auth.ALLOWED_ORIGINS[0]
        # https so the client's cookie jar sends back the Secure cookie.
        self.client = TestClient(
            app,
            base_url="https://testserver",
            headers={**PROXY_HEADERS, "Origin": self.origin},
        )

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def _login(self):
        return self.client.post(
            "/api/auth/login",
            data={"username": EMAIL, "password": PASSWORD},
        )

    def _auth_cookie_header(self, response) -> str:
        cookies = [
            c for c in response.headers.get_list("set-cookie")
            if c.startswith(f"{core_auth.ACCESS_TOKEN_COOKIE}=")
        ]
        self.assertEqual(len(cookies), 1)
        return cookies[0]

    def test_login_sets_httponly_cookie_and_omits_token(self) -> None:
        response = self._login()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access_token", response.json())
        cookie = self._auth_cookie_header(response).lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertIn("secure", cookie)
        self.assertIn(f"max-age={core_auth.JWT_EXPIRE_MINUTES * 60}", cookie)
        self.assertIn("path=/", cookie)

    def test_secure_flag_follows_cookie_secure(self) -> None:
        with patch.object(core_auth, "COOKIE_SECURE", False):
            response = self._login()

        self.assertNotIn("secure", self._auth_cookie_header(response).lower())

    def test_signup_sets_cookie_and_omits_token(self) -> None:
        response = self.client.post(
            "/api/auth/signup",
            json={"email": EMAIL, "password": PASSWORD, "full_name": "Test User"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json(),
            {"user_id": USER_ID, "email": EMAIL, "full_name": "Test User"},
        )
        self.assertIn("httponly", self._auth_cookie_header(response).lower())

    def test_wrong_password_sets_no_cookie(self) -> None:
        response = self.client.post(
            "/api/auth/login",
            data={"username": EMAIL, "password": "wrong-password"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get_list("set-cookie"), [])

    def test_me_reads_the_cookie(self) -> None:
        self._login()
        response = self.client.get("/api/auth/me")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], USER_ID)

    def test_me_without_cookie_is_401(self) -> None:
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_bearer_header_is_no_longer_accepted(self) -> None:
        token = core_auth.create_access_token(USER_ID, EMAIL)
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 401)

    def test_logout_clears_the_cookie(self) -> None:
        self._login()
        response = self.client.post("/api/auth/logout")

        self.assertEqual(response.status_code, 204)
        cookie = self._auth_cookie_header(response).lower()
        self.assertIn("max-age=0", cookie)
        self.assertIn("httponly", cookie)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_post_from_foreign_origin_is_403(self) -> None:
        response = self.client.post(
            "/api/auth/login",
            data={"username": EMAIL, "password": PASSWORD},
            headers={"Origin": "https://evil.example"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers.get_list("set-cookie"), [])

    def test_post_without_origin_is_403(self) -> None:
        client = TestClient(app, base_url="https://testserver", headers=PROXY_HEADERS)
        for method in ("POST", "PUT", "DELETE"):
            with self.subTest(method=method):
                response = client.request(method, "/api/auth/logout")
                self.assertEqual(response.status_code, 403)

    def test_get_is_not_origin_checked(self) -> None:
        response = self.client.get(
            "/health",
            headers={"Origin": "https://evil.example"},
        )

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
