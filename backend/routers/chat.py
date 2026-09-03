import base64
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from groq import Groq
from pydantic import BaseModel, field_validator

from core.auth import get_current_user
from core.config import GROQ_API_KEY
from core.database import get_pool
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


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    pool = get_pool()
    await _require_owned_document(
        request.doc_id,
        current_user["id"],
    )
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    result = await answer_query(
        request.question,
        history,
        doc_id=request.doc_id,
        user_id=current_user["id"],
    )

    # Persist both turns to chat_history
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'user', $3)",
        current_user["id"],
        request.doc_id or "",
        request.question,
    )
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'assistant', $3)",
        current_user["id"],
        request.doc_id or "",
        result["answer"],
    )

    return result


@router.post("/chat/agentic")
async def chat_agentic(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Phase-1 LangGraph endpoint; preserves the existing chat response contract."""
    pool = get_pool()
    await _require_owned_document(
        request.doc_id,
        current_user["id"],
    )
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    result = await answer_agentic_query(
        request.question,
        history,
        user_id=current_user["id"],
        doc_id=request.doc_id,
    )

    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'user', $3)",
        current_user["id"],
        request.doc_id or "",
        request.question,
    )
    await pool.execute(
        "INSERT INTO chat_history (user_id, doc_id, role, content) VALUES ($1, $2, 'assistant', $3)",
        current_user["id"],
        request.doc_id or "",
        result["answer"],
    )

    return result


@router.get("/chat/history/{doc_id}")
async def get_history(doc_id: str, current_user: dict = Depends(get_current_user)):
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
async def transcribe_voice(
    request: VoiceRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        transcription = _get_groq().audio.transcriptions.create(
            file=("audio.webm", audio_bytes, "audio/webm"),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        return {"transcript": transcription}
    except Exception as exc:
        raise HTTPException(500, "Transcription failed") from exc
