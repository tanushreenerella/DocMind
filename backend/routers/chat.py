import base64
from typing import List

from fastapi import APIRouter, HTTPException
from groq import Groq
from pydantic import BaseModel, field_validator

from core.config import GROQ_API_KEY
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


@router.post("/chat")
async def chat(request: ChatRequest) -> dict:
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    result = await answer_query(request.question, history, doc_id=request.doc_id)
    return result


@router.post("/transcribe")
async def transcribe_voice(request: VoiceRequest) -> dict:
    """Transcribe audio via Groq Whisper."""
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
