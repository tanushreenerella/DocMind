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
    all_embeddings = []
    batch_size = 10
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        for attempt in range(3):  # retry up to 3 times
            try:
                result = await asyncio.to_thread(
                    _get_client().models.embed_content,
                    model="models/gemini-embedding-001",
                    contents=batch,
                )
                all_embeddings.extend([list(e.values) for e in result.embeddings])
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 60 * (attempt + 1)  # 60s, 120s
                    print(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise
        
        await asyncio.sleep(2)
    
    return all_embeddings