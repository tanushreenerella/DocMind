import base64
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from groq import Groq
from pydantic import BaseModel, field_validator

from core.auth import get_current_user
from core.config import (
    GROQ_API_KEY,
    RATE_LIMIT_AGENTIC_CHAT,
    RATE_LIMIT_CHAT,
    RATE_LIMIT_VOICE,
)
from core.database import get_pool
from core.rate_limit import limiter
from core.token_budget import charged_to, ensure_within_budget
from services.agentic_rag import answer_agentic_query
from services.rag import answer_query

router = APIRouter()
_groq_client: Groq | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"user", "assistant", "system"}:
            raise ValueError("Invalid role")
        return v


class ChatRequest(BaseModel):
    question: str
    conversation_history: List[Message] = []
    doc_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 2000:
            raise ValueError("Question too long (max 2000 chars)")
        return v


class VoiceRequest(BaseModel):
    audio_base64: str


async def _require_owned_document(
    doc_id: str | None,
    user_id: str,
) -> None:
    """Reject document IDs not owned by the caller without revealing why."""
    if not doc_id:
        return

    pool = get_pool()
    owned_document = await pool.fetchval(
        "SELECT 1 FROM documents WHERE doc_id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    if not owned_document:
        # Keep this identical for unknown and foreign document IDs.
        raise HTTPException(
            status_code=403,
            detail="Document access denied",
        )


async def answer_for_user(
    user_id: str,
    question: str,
    history: list[dict],
    doc_id: str | None,
    *,
    agentic: bool,
) -> dict:
    """The answer pipeline with every check the chat endpoints apply, charged to ``user_id``.

    Raises HTTPException 429 (daily token budget) or 403 (doc_id not the
    caller's) before any AI work. An outage comes back as the usual result with
    ``error`` set. Shared by /chat, /chat/agentic and the MCP ask_documents tool.
    """
    await ensure_within_budget(user_id)
    await _require_owned_document(doc_id, user_id)
    with charged_to(user_id):
        if agentic:
            return await answer_agentic_query(question, history, user_id=user_id, doc_id=doc_id)
        return await answer_query(question, history, doc_id=doc_id, user_id=user_id)


@router.post("/chat")
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    pool = get_pool()
    history = [{"role": m.role, "content": m.content} for m in body.conversation_history]
    result = await answer_for_user(current_user["id"], body.question, history, body.doc_id, agentic=False)
    if result.get("error"):
        # An outage notice isn't part of the conversation; the user will ask again.
        return result

    # Persist both turns to chat_history
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'user', $3)",
        current_user["id"],
        body.doc_id or "",
        body.question,
    )
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'assistant', $3)",
        current_user["id"],
        body.doc_id or "",
        result["answer"],
    )

    return result


@router.post("/chat/agentic")
@limiter.limit(RATE_LIMIT_AGENTIC_CHAT)
async def chat_agentic(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Phase-1 LangGraph endpoint; preserves the existing chat response contract."""
    pool = get_pool()
    history = [{"role": m.role, "content": m.content} for m in body.conversation_history]
    result = await answer_for_user(current_user["id"], body.question, history, body.doc_id, agentic=True)
    if result.get("error"):
        return result

    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'user', $3)",
        current_user["id"],
        body.doc_id or "",
        body.question,
    )
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'assistant', $3)",
        current_user["id"],
        body.doc_id or "",
        result["answer"],
    )

    return result


@router.get("/chat/history/{doc_id}")
async def get_history(doc_id: str, current_user: dict = Depends(get_current_user)):
    await _require_owned_document(doc_id, current_user["id"])
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT role, content, created_at
        FROM chat_history
        WHERE user_id = $1 AND doc_id = $2
        ORDER BY created_at ASC
        """,
        current_user["id"],
        doc_id,
    )
    return {
        "history": [
            {"role": r["role"], "content": r["content"], "timestamp": r["created_at"].isoformat()}
            for r in rows
        ]
    }


@router.post("/transcribe")
@limiter.limit(RATE_LIMIT_VOICE)
async def transcribe_voice(
    request: Request,
    body: VoiceRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        audio_bytes = base64.b64decode(body.audio_base64)
        transcription = _get_groq().audio.transcriptions.create(
            file=("audio.webm", audio_bytes, "audio/webm"),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        return {"transcript": transcription}
    except Exception as exc:
        raise HTTPException(500, "Transcription failed") from exc
