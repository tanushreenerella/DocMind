"""MCP server at /api/mcp: the user's documents as tools for Claude Desktop and other MCP clients.

Streamable HTTP in stateless mode with plain JSON responses (no SSE, no
sessions), so it works through the Next.js/Vercel proxy like any other /api
route. Every request needs "Authorization: Bearer dm_..." (a personal access
token; the session cookie is not accepted), and every tool works only on that
token's user.

Tools reuse the app's own code paths: the documents listing, owner-filtered
search, and chat's answer_for_user (daily token budget, document ownership,
outage codes). A refusal or outage becomes an MCP tool error (isError) carrying
a code in brackets; the SDK prefixes it, so clients see e.g.
"Error executing tool ask_documents: [daily_budget_exceeded] Daily AI usage ...".
"""

from typing import Annotated, Any, Literal

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import Field, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from core.auth import MCP_PATH, PAT_PREFIX, _bearer_token, user_from_personal_access_token
from core.config import RATE_LIMIT_AGENTIC_CHAT, RATE_LIMIT_CHAT
from core.rate_limit import hit_user_limit
from routers import documents
from routers.chat import ChatRequest, _require_owned_document, answer_for_user
from services.embedder import search
from services.errors import ServiceUnavailable

__all__ = ["MCP_PATH", "mcp", "mcp_endpoint"]

MAX_SEARCH_RESULTS = 10
# Search excerpts are trimmed to about this many characters, at a word boundary.
EXCERPT_CHARS = 500

mcp = FastMCP(
    "DocMind",
    instructions=(
        "Search and ask questions about the documents this user uploaded to DocMind. "
        "Call list_documents for doc_ids; search_documents for raw passages; "
        "ask_documents for a cited answer."
    ),
    stateless_http=True,
    json_response=True,
    # The SDK logs several INFO lines per request; ours are enough.
    log_level="WARNING",
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _user_id(ctx: Context) -> str:
    request = ctx.request_context.request
    user = getattr(getattr(request, "state", None), "mcp_user", None)
    if not user:
        # Only reachable if a tool runs outside McpEndpoint, which always sets it.
        raise ToolError("[unauthorized] No authenticated user")
    return user["id"]


def _tool_error(exc: HTTPException) -> ToolError:
    code = {403: "forbidden", 429: "daily_budget_exceeded"}.get(exc.status_code, "error")
    return ToolError(f"[{code}] {exc.detail}")


def _validation_error(exc: ValidationError) -> ToolError:
    messages = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
    return ToolError(f"[invalid_arguments] {messages}")


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit] + "…"


# ── tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
async def list_documents(ctx: Context) -> dict[str, Any]:
    """List the user's indexed documents (doc_id, name, type, sensitivity, suspicious flag)."""
    return await documents.list_documents(current_user={"id": _user_id(ctx)})


@mcp.tool()
async def search_documents(
    ctx: Context,
    query: Annotated[str, Field(description="What to look for.")],
    doc_id: Annotated[str | None, Field(description="Only search this document (from list_documents).")] = None,
    k: Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS, description="How many passages to return.")] = 5,
) -> dict[str, Any]:
    """Find the passages most relevant to a query in the user's documents, with document name and page."""
    user_id = _user_id(ctx)
    try:
        query = ChatRequest(question=query).question
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    try:
        await _require_owned_document(doc_id, user_id)
    except HTTPException as exc:
        raise _tool_error(exc) from exc
    try:
        hits = await search(query, k, doc_id, user_id=user_id)
    except ServiceUnavailable as exc:
        raise ToolError(f"[{exc.code}] {exc.message}") from exc
    return {
        "results": [
            {
                "doc_id": hit["doc_id"],
                "doc_name": hit["doc_name"],
                "page": hit["page_number"],
                "excerpt": _excerpt(hit["chunk_text"]),
            }
            for hit in hits[:k]
        ]
    }


@mcp.tool()
async def ask_documents(
    ctx: Context,
    question: Annotated[str, Field(description="The question, in plain language.")],
    doc_id: Annotated[str | None, Field(description="Only use this document (from list_documents).")] = None,
    mode: Annotated[
        Literal["basic", "agentic"],
        Field(description="basic: one retrieval pass. agentic: plans sub-questions; slower, for multi-part questions."),
    ] = "basic",
) -> dict[str, Any]:
    """Answer a question from the user's documents, with citations (document, page, excerpt)."""
    user_id = _user_id(ctx)
    try:
        question = ChatRequest(question=question).question
    except ValidationError as exc:
        raise _validation_error(exc) from exc

    agentic = mode == "agentic"
    # The same per-minute caps as the chat endpoints (counted separately from them).
    limit = RATE_LIMIT_AGENTIC_CHAT if agentic else RATE_LIMIT_CHAT
    retry_after = hit_user_limit(limit, f"mcp-ask-{mode}", user_id)
    if retry_after is not None:
        raise ToolError(f"[rate_limited] Too many requests, try again in {retry_after} s")

    try:
        result = await answer_for_user(user_id, question, [], doc_id, agentic=agentic)
    except HTTPException as exc:
        raise _tool_error(exc) from exc
    if result.get("error"):
        raise ToolError(f"[{result['error']}] {result['answer']}")
    return {
        "answer": result["answer"],
        "has_answer": result.get("has_answer", False),
        "citations": [
            {
                "doc_id": c.get("doc_id"),
                "doc_name": c["doc_name"],
                "page": c["page_number"],
                "excerpt": c.get("excerpt", ""),
            }
            for c in result.get("citations", [])
        ],
    }


# ── HTTP endpoint ───────────────────────────────────────────────────────────

def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"detail": "A personal access token is required: Authorization: Bearer dm_..."},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


class McpEndpoint:
    """ASGI endpoint: authenticate the token, then hand the request to the MCP SDK.

    A session manager per request: in stateless mode each request gets its own
    transport anyway, and this avoids depending on app lifespan (a mounted or
    routed sub-app's lifespan never runs, and a manager can only run once).
    """

    # slowapi names routes by __module__.__name__; without one it would skip the default limit.
    __name__ = "mcp_endpoint"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        token = _bearer_token(request)
        if not token or not token.startswith(PAT_PREFIX):
            await _unauthorized()(scope, receive, send)
            return
        try:
            user = await user_from_personal_access_token(token)
        except HTTPException:
            await _unauthorized()(scope, receive, send)
            return
        # Tools read it back from ctx.request_context.request.state.
        request.state.mcp_user = user

        manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server,
            json_response=mcp.settings.json_response,
            stateless=mcp.settings.stateless_http,
        )
        async with manager.run():
            await manager.handle_request(scope, receive, send)


mcp_endpoint = McpEndpoint()
