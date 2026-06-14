import asyncio
from google import genai
from core.config import GEMINI_API_KEY

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def get_embedding(text: str) -> list[float]:
    result = await asyncio.to_thread(
        _get_client().models.embed_content,
        model="text-embedding-004",
        contents=text,
    )
    return list(result.embeddings[0].values)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    result = await asyncio.to_thread(
        _get_client().models.embed_content,
        model="text-embedding-004",
        contents=texts,
    )
    return [list(e.values) for e in result.embeddings]
