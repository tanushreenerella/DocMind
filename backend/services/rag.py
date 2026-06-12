"""
Agentic RAG: retrieves relevant chunks, synthesizes grounded answers with citations.
"""
from groq import Groq

from backend.core.config import settings
from backend.services.embedder import search

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

_SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the provided document sources.

Rules:
- Cite your sources inline using [SOURCE N] notation matching the provided sources
- If the sources don't contain enough information, say so clearly — do NOT hallucinate
- Be concise and precise
- If multiple sources confirm the same point, cite all of them
- Never invent facts not present in the sources"""


async def answer_query(
    question: str,
    conversation_history: list[dict],
    doc_id: str | None = None,
) -> dict:
    """
    Returns:
    {
      "answer": str,
      "citations": [{"doc_name": str, "page_number": int, "image_filename": str, "excerpt": str}],
      "has_answer": bool
    }
    """
    hits = await search(question, n_results=6, doc_id=doc_id)

    if not hits:
        return {
            "answer": "I don't have any documents in my knowledge base yet. Please upload relevant documents first.",
            "citations": [],
            "has_answer": False,
        }

    relevant_hits = [h for h in hits if h["relevance_score"] > 0.1]

    if not relevant_hits:
        return {
            "answer": "I couldn't find information relevant enough to answer this question confidently. The documents I have may not cover this topic.",
            "citations": [],
            "has_answer": False,
        }

    context_parts = []
    for i, hit in enumerate(relevant_hits):
        context_parts.append(
            f"[SOURCE {i+1}: {hit['doc_name']}, Page {hit['page_number']}]\n{hit['chunk_text']}"
        )
    context = "\n\n".join(context_parts)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    for msg in conversation_history[-6:]:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {question}",
        }
    )

    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
    )

    answer: str = response.choices[0].message.content.strip()

    # Collect citations for sources actually referenced in the answer
    citations: list[dict] = []
    seen: set[tuple] = set()
    for i, hit in enumerate(relevant_hits):
        if f"[SOURCE {i+1}]" in answer:
            key = (hit["doc_name"], hit["page_number"])
            if key not in seen:
                seen.add(key)
                citations.append(
                    {
                        "doc_name": hit["doc_name"],
                        "page_number": hit["page_number"],
                        "image_filename": hit["image_filename"],
                        "excerpt": hit["chunk_text"][:200] + "...",
                    }
                )

    # Replace [SOURCE N] markers with readable inline references
    for i, hit in enumerate(relevant_hits):
        answer = answer.replace(
            f"[SOURCE {i+1}]",
            f"[{hit['doc_name']}, p.{hit['page_number']}]",
        )

    return {
        "answer": answer,
        "citations": citations,
        "has_answer": True,
    }
