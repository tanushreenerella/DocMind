"""MCP server at /api/mcp: auth, per-user scoping, tool errors, and per-user rate limiting."""

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core import auth as core_auth
from core import token_budget
from core.auth import create_access_token
from core.rate_limit import limiter, rate_limit_key
from core.token_budget import MemoryStore, record_usage
from main import app
from services.errors import AIUnavailable, SearchUnavailable, error_response
from tests import PROXY_HEADERS
from tests import test_personal_tokens as pat_tests

USER_A, USER_B = pat_tests.USER_A, pat_tests.USER_B
DOC_A = "11111111-1111-1111-1111-111111111111"
DOC_B = "22222222-2222-2222-2222-222222222222"
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class FakeMcpDB(pat_tests.FakeTokenDB):
    """The token DB plus a documents table, and the cached token->user lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.documents = [
            {"doc_id": DOC_A, "user_id": USER_A, "doc_name": "a.pdf", "document_type": "report",
             "sensitivity_level": "internal", "suspicious": False, "status": "complete"},
            {"doc_id": DOC_B, "user_id": USER_B, "doc_name": "b.pdf", "document_type": "invoice",
             "sensitivity_level": "internal", "suspicious": False, "status": "complete"},
        ]
        self.token_user_lookups = 0

    def issue(self, user_id: str) -> str:
        token = core_auth.generate_personal_access_token()
        self.tokens.append({
            "id": f"tok-{len(self.tokens)}", "user_id": user_id, "name": "mcp",
            "token_hash": core_auth.hash_personal_access_token(token), "prefix": token[:11],
            "created_at": datetime.now(timezone.utc), "last_used_at": None, "revoked_at": None,
        })
        return token

    async def fetch(self, query: str, *args):
        q = self._q(query)
        if q.startswith("SELECT doc_id, doc_name, document_type, sensitivity_level, suspicious FROM documents"):
            return [d for d in self.documents if d["user_id"] == args[0]]
        return await super().fetch(query, *args)

    async def fetchval(self, query: str, *args):
        q = self._q(query)
        if q.startswith("SELECT 1 FROM documents WHERE doc_id = $1 AND user_id = $2"):
            return 1 if any(d["doc_id"] == args[0] and d["user_id"] == args[1] for d in self.documents) else None
        if q.startswith("SELECT t.user_id FROM personal_access_tokens t JOIN users u"):
            self.token_user_lookups += 1
            for t in self.tokens:
                if t["token_hash"] == args[0] and t["revoked_at"] is None and t["user_id"] in self.users:
                    return t["user_id"]
            return None
        return await super().fetchval(query, *args)


def _hit(doc_id: str, text: str, page: int = 1) -> dict:
    return {"doc_id": doc_id, "doc_name": "a.pdf", "page_number": page, "chunk_text": text,
            "image_filename": "", "relevance_score": 0.9}


class _McpTestCase(unittest.TestCase):
    def setUp(self) -> None:
        limiter.reset()
        core_auth._token_user_cache.clear()
        self.db = FakeMcpDB()
        self.memory = MemoryStore()

        async def answer(question, history, *args, **kwargs) -> dict:
            await record_usage(SimpleNamespace(usage=SimpleNamespace(total_tokens=250)))
            return {
                "answer": "The total is $5 [a.pdf, p.2].",
                "has_answer": True,
                "error": None,
                "citations": [{"doc_name": "a.pdf", "page_number": 2, "image_filename": "a_p2.jpg",
                               "doc_id": DOC_A, "excerpt": "Total: $5..."}],
            }

        self.answer = AsyncMock(side_effect=answer)
        self.agentic = AsyncMock(side_effect=answer)
        self.search = AsyncMock(return_value=[_hit(DOC_A, "Total: $5")])
        self.patches = [
            patch("core.auth.get_pool", return_value=self.db),
            patch("routers.documents.get_pool", return_value=self.db),
            patch("routers.chat.get_pool", return_value=self.db),
            patch("routers.chat.answer_query", self.answer),
            patch("routers.chat.answer_agentic_query", self.agentic),
            patch("routers.mcp.search", self.search),
            patch.object(token_budget, "_stores", [self.memory]),
            patch.object(token_budget, "DAILY_TOKEN_BUDGET", 1000),
        ]
        for p in self.patches:
            p.start()
        # An MCP client is a script: proxy secret (added by the proxy), no Origin, no cookie.
        self.client = TestClient(app, headers=PROXY_HEADERS)
        self.token_a = self.db.issue(USER_A)

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        limiter.reset()
        core_auth._token_user_cache.clear()

    def rpc(self, method: str, params: dict | None = None, *, token: str | None = None, headers: dict | None = None):
        token = self.token_a if token is None else token
        all_headers = {**MCP_HEADERS, **({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})}
        body = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})}
        return self.client.post("/api/mcp", headers=all_headers, json=body)

    def call(self, tool: str, token: str | None = None, **arguments) -> dict:
        response = self.rpc("tools/call", {"name": tool, "arguments": arguments}, token=token)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["result"]

    def ok(self, tool: str, **arguments) -> dict:
        result = self.call(tool, **arguments)
        self.assertFalse(result.get("isError"), result)
        return json.loads(result["content"][0]["text"])

    def error(self, tool: str, **arguments) -> str:
        """A tool error's message, minus the prefix the SDK adds to every one."""
        result = self.call(tool, **arguments)
        self.assertTrue(result.get("isError"), result)
        text = result["content"][0]["text"]
        prefix = f"Error executing tool {tool}: "
        self.assertTrue(text.startswith(prefix), text)
        return text.removeprefix(prefix)


class ProtocolTests(_McpTestCase):
    def test_initialize_returns_json_not_sse(self) -> None:
        response = self.rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"},
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertEqual(response.json()["result"]["serverInfo"]["name"], "DocMind")
        # Stateless: no session for the client to carry.
        self.assertNotIn("mcp-session-id", response.headers)

    def test_tools_are_listed_with_their_limits(self) -> None:
        response = self.rpc("tools/list")

        self.assertEqual(response.status_code, 200)
        tools = {t["name"]: t for t in response.json()["result"]["tools"]}
        self.assertEqual(set(tools), {"list_documents", "search_documents", "ask_documents"})
        search_props = tools["search_documents"]["inputSchema"]["properties"]
        self.assertEqual((search_props["k"]["minimum"], search_props["k"]["maximum"]), (1, 10))
        self.assertEqual(tools["search_documents"]["inputSchema"]["required"], ["query"])
        self.assertEqual(tools["ask_documents"]["inputSchema"]["properties"]["mode"]["enum"], ["basic", "agentic"])
        self.assertEqual(tools["ask_documents"]["inputSchema"]["required"], ["question"])


class AuthTests(_McpTestCase):
    def test_missing_token_is_401_with_bearer_challenge(self) -> None:
        response = self.rpc("tools/list", token="")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_invalid_tokens_are_401(self) -> None:
        jwt = create_access_token(USER_A, "a@example.com")
        for token in ("dm_" + "A" * 43, "dm_", "not-a-token", jwt):
            with self.subTest(token=token[:10]):
                self.assertEqual(self.rpc("tools/list", token=token).status_code, 401)

    def test_session_cookie_is_not_enough(self) -> None:
        cookie = {"Cookie": f"{core_auth.ACCESS_TOKEN_COOKIE}={create_access_token(USER_A, 'a@example.com')}"}
        self.assertEqual(self.rpc("tools/list", token="", headers=cookie).status_code, 401)

    def test_revoked_token_is_401_at_once(self) -> None:
        self.assertEqual(self.rpc("tools/list").status_code, 200)
        self.db.tokens[0]["revoked_at"] = datetime.now(timezone.utc)
        # Even though the rate-limit cache still maps it to USER_A.
        self.assertEqual(self.rpc("tools/list").status_code, 401)

    def test_proxy_secret_is_still_required(self) -> None:
        bare = TestClient(app)
        response = bare.post("/api/mcp", headers={**MCP_HEADERS, "Authorization": f"Bearer {self.token_a}"},
                             json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(response.status_code, 403)

    def test_use_is_recorded_on_the_token(self) -> None:
        self.rpc("tools/list")
        self.assertIsNotNone(self.db.tokens[0]["last_used_at"])


class ToolScopingTests(_McpTestCase):
    def test_list_documents_shows_only_the_token_users_documents(self) -> None:
        listed = self.ok("list_documents")
        self.assertEqual([d["doc_id"] for d in listed["documents"]], [DOC_A])

        token_b = self.db.issue(USER_B)
        result = self.call("list_documents", token=token_b)
        self.assertEqual([d["doc_id"] for d in json.loads(result["content"][0]["text"])["documents"]], [DOC_B])

    def test_search_is_scoped_bounded_and_trimmed(self) -> None:
        self.search.return_value = [_hit(DOC_A, "word " * 400, page=3), _hit(DOC_A, "short")]

        results = self.ok("search_documents", query="  totals  ", k=2)["results"]

        self.search.assert_awaited_once_with("totals", 2, None, user_id=USER_A)
        self.assertEqual(results[0]["page"], 3)
        self.assertEqual(results[0]["doc_name"], "a.pdf")
        self.assertLessEqual(len(results[0]["excerpt"]), 501)
        self.assertTrue(results[0]["excerpt"].endswith("…"))
        self.assertEqual(results[1]["excerpt"], "short")

    def test_search_rejects_bad_arguments(self) -> None:
        self.assertIn("10", self.error("search_documents", query="x", k=11))
        self.assertIn("1", self.error("search_documents", query="x", k=0))
        self.assertIn("[invalid_arguments] Question cannot be empty", self.error("search_documents", query="   "))
        self.search.assert_not_awaited()

    def test_foreign_doc_id_is_rejected(self) -> None:
        for tool, args in (("search_documents", {"query": "x"}), ("ask_documents", {"question": "x"})):
            with self.subTest(tool):
                self.assertEqual(self.error(tool, doc_id=DOC_B, **args), "[forbidden] Document access denied")
        self.assertEqual(self.error("ask_documents", question="x", doc_id="no-such-doc"),
                         "[forbidden] Document access denied")
        self.search.assert_not_awaited()
        self.answer.assert_not_awaited()

    def test_own_doc_id_is_passed_through(self) -> None:
        self.ok("search_documents", query="x", doc_id=DOC_A)
        self.search.assert_awaited_once_with("x", 5, DOC_A, user_id=USER_A)
        self.ok("ask_documents", question="x", doc_id=DOC_A)
        self.assertEqual(self.answer.await_args.kwargs, {"doc_id": DOC_A, "user_id": USER_A})


class AskTests(_McpTestCase):
    def test_basic_answer_with_citations_charged_to_the_token_user(self) -> None:
        answered = self.ok("ask_documents", question="What is the total?")

        self.assertEqual(answered["answer"], "The total is $5 [a.pdf, p.2].")
        self.assertTrue(answered["has_answer"])
        self.assertEqual(answered["citations"], [{"doc_id": DOC_A, "doc_name": "a.pdf", "page": 2, "excerpt": "Total: $5..."}])
        self.answer.assert_awaited_once()
        self.agentic.assert_not_awaited()
        self.assertEqual(self.answer.await_args.args[:2], ("What is the total?", []))
        self.assertEqual(self.memory._tokens, {(USER_A, token_budget._today()): 250})

    def test_agentic_mode_uses_the_agentic_pipeline(self) -> None:
        self.ok("ask_documents", question="Compare both invoices", mode="agentic")
        self.agentic.assert_awaited_once()
        self.answer.assert_not_awaited()

    def test_unknown_mode_is_rejected(self) -> None:
        self.error("ask_documents", question="x", mode="turbo")
        self.answer.assert_not_awaited()

    def test_daily_budget_limit_is_a_tool_error(self) -> None:
        self.memory._tokens[(USER_A, token_budget._today())] = 1000

        message = self.error("ask_documents", question="What is the total?")

        self.assertEqual(message, "[daily_budget_exceeded] Daily AI usage limit reached, resets at midnight UTC")
        self.answer.assert_not_awaited()
        # Another user's token is unaffected.
        token_b = self.db.issue(USER_B)
        self.assertFalse(self.call("ask_documents", token=token_b, question="x").get("isError"))

    def test_outages_are_tool_errors_with_their_code(self) -> None:
        self.answer.side_effect = None
        self.answer.return_value = error_response(AIUnavailable())
        self.assertEqual(
            self.error("ask_documents", question="x"),
            "[ai_unavailable] The AI service is busy or unavailable right now. Please try again shortly.",
        )

        self.search.side_effect = SearchUnavailable()
        self.assertTrue(self.error("search_documents", query="x").startswith("[search_unavailable] "))

    def test_chat_rate_limit_applies_per_user(self) -> None:
        with patch("routers.mcp.RATE_LIMIT_CHAT", "2/minute"):
            self.ok("ask_documents", question="one")
            self.ok("ask_documents", question="two")
            self.assertTrue(self.error("ask_documents", question="three").startswith("[rate_limited] "))
            # Counted per user, not per token or connection.
            token_a2 = self.db.issue(USER_A)
            self.assertTrue(self.call("ask_documents", token=token_a2, question="four")["isError"])
        self.assertEqual(self.answer.await_count, 2)


class RateLimitKeyTests(_McpTestCase):
    def test_token_requests_share_one_bucket_per_user_across_ips(self) -> None:
        for _ in range(30):  # the default limit, RATE_LIMIT_PER_MINUTE
            self.assertEqual(self.rpc("tools/list", headers={"x-client-ip": "198.51.100.1"}).status_code, 200)

        refused = self.rpc("tools/list", headers={"x-client-ip": "203.0.113.9"})

        self.assertEqual(refused.status_code, 429)
        # A different user is unaffected.
        token_b = self.db.issue(USER_B)
        self.assertEqual(self.rpc("tools/list", token=token_b, headers={"x-client-ip": "203.0.113.9"}).status_code, 200)

    def test_token_to_user_is_cached_for_60_seconds(self) -> None:
        start = core_auth.time.monotonic()
        with patch.object(core_auth.time, "monotonic", return_value=start):
            self.rpc("tools/list")
            self.rpc("tools/list")
        self.assertEqual(self.db.token_user_lookups, 1)

        with patch.object(core_auth.time, "monotonic", return_value=start + 61):
            self.rpc("tools/list")
        self.assertEqual(self.db.token_user_lookups, 2)

    def test_invalid_tokens_are_limited_by_ip(self) -> None:
        bogus = "dm_" + "B" * 43
        for _ in range(30):
            self.rpc("tools/list", token=bogus, headers={"x-client-ip": "198.51.100.7"})
        # Same IP, made-up token: still the IP's bucket, now full.
        self.assertEqual(self.rpc("tools/list", token="dm_" + "C" * 43, headers={"x-client-ip": "198.51.100.7"}).status_code, 429)
        # A valid token from that IP has its own (user) bucket.
        self.assertEqual(self.rpc("tools/list", headers={"x-client-ip": "198.51.100.7"}).status_code, 200)

    def test_key_prefers_the_validated_token_user(self) -> None:
        request = SimpleNamespace(
            state=SimpleNamespace(token_user_id=USER_B), cookies={}, headers={"x-client-ip": "198.51.100.1"}
        )
        self.assertEqual(rate_limit_key(request), f"user:{USER_B}")
        request.state = SimpleNamespace()
        self.assertEqual(rate_limit_key(request), "ip:198.51.100.1")

    def test_token_lookup_failure_falls_back_to_ip_and_still_authenticates(self) -> None:
        with patch.object(core_auth, "cached_token_user_id", AsyncMock(side_effect=ConnectionError("db"))):
            self.assertEqual(self.rpc("tools/list").status_code, 200)


if __name__ == "__main__":
    unittest.main()
