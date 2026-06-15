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
        model="models/gemini-embedding-001",
        contents=text,
    )
    return list(result.embeddings[0].values)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for attempt in range(6):
            try:
                result = await asyncio.to_thread(
                    _get_client().models.embed_content,
                    model="models/gemini-embedding-001",
                    contents=batch,
                )
                all_embeddings.extend([list(e.values) for e in result.embeddings])
                print(f"[embeddings] batch {batch_num}/{total_batches} done")
                # Small polite delay — only between batches, not after the last one
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.3)
                break
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err.upper()
                if is_rate_limit and attempt < 5:
                    wait = min(30 * (attempt + 1), 120)  # 30s, 60s, 90s, 120s, 120s
                    print(f"[embeddings] rate limited — waiting {wait}s (attempt {attempt + 1}/6)")
                    await asyncio.sleep(wait)
                else:
                    print(f"[embeddings] error on batch {batch_num}: {e}")
                    raise

    return all_embeddings
