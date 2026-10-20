"""Only the Next.js proxy (holder of PROXY_SECRET) may reach /api."""

import unittest

from fastapi.testclient import TestClient

from core import config
from core.rate_limit import limiter
from main import app
from tests import PROXY_HEADERS


class ProxySecretTests(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        self.origin = {"Origin": config.ALLOWED_ORIGINS[0]}
        self.client = TestClient(app)

    def tearDown(self) -> None:
        limiter.reset()

    def test_api_without_secret_is_403(self) -> None:
        for method, path in (("GET", "/api/debug-ip"), ("POST", "/api/auth/logout")):
            with self.subTest(method=method, path=path):
                response = self.client.request(method, path, headers=self.origin)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json(), {"detail": "Forbidden"})

    def test_api_with_wrong_secret_is_403(self) -> None:
        for secret in ("wrong", "", PROXY_HEADERS["x-proxy-secret"] + "x"):
            with self.subTest(secret=secret):
                response = self.client.get("/api/debug-ip", headers={"x-proxy-secret": secret})
                self.assertEqual(response.status_code, 403)

    def test_api_with_secret_is_allowed(self) -> None:
        response = self.client.post("/api/auth/logout", headers={**PROXY_HEADERS, **self.origin})
        self.assertEqual(response.status_code, 204)

    def test_health_needs_no_secret(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_rejected_requests_spend_no_rate_limit_budget(self) -> None:
        for _ in range(config.RATE_LIMIT_PER_MINUTE + 5):
            self.client.get("/api/debug-ip", headers={"x-client-ip": "203.0.113.7"})

        response = self.client.get(
            "/api/debug-ip", headers={**PROXY_HEADERS, "x-client-ip": "203.0.113.7"}
        )
        self.assertEqual(response.status_code, 200)

    def test_debug_ip_reports_client_ip_header(self) -> None:
        response = self.client.get(
            "/api/debug-ip",
            headers={
                **PROXY_HEADERS,
                "x-client-ip": "203.0.113.7",
                "x-forwarded-for": "198.51.100.9",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["x_client_ip"], "203.0.113.7")
        self.assertEqual(response.json()["xff"], "198.51.100.9")


if __name__ == "__main__":
    unittest.main()
