"""Evidence-first timeline generation for a single document."""
import asyncio
import json
import re

from groq import Groq

from core.config import GROQ_API_KEY
from services.embedder import search

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _parse_json(raw: str) -> dict:
    """Accept a JSON response even if a model accidentally wraps it in a fence."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(raw)


async def build_evidence_timeline(doc_id: str, *, user_id: str) -> dict:
    """Return important dates, obligations and amounts with source-page evidence."""
    queries = [
        "dates deadlines milestones renewals effective dates",
        "amounts totals payments fees financial figures",
        "obligations actions risks exceptions next steps",
    ]
    hits: list[dict] = []
    seen_chunks: set[str] = set()
    for query in queries:
        for hit in await search(
            query,
            n_results=6,
            doc_id=doc_id,
            user_id=user_id,
        ):
            key = hit["chunk_text"]
            if key not in seen_chunks:
                seen_chunks.add(key)
                hits.append(hit)

    if not hits:
        return {
            "briefing": "I could not find enough indexed evidence to build a timeline.",
            "citations": [],
            "has_timeline": False,
        }

    sources = "\n\n".join(
        f"[SOURCE {i + 1}, page {hit['page_number']}]\n{hit['chunk_text']}"
        for i, hit in enumerate(hits)
    )
    prompt = f"""You are an evidence analyst. Build a concise decision timeline using ONLY the sources below.

Return valid JSON only, matching exactly:
{{
  "headline": "one sentence, or 'No material timeline found'",
  "events": [
    {{"when": "date or timeframe exactly as stated", "event": "fact", "why_it_matters": "brief impact", "sources": [1]}}
  ],
  "risks_or_open_questions": [{{"item": "brief item", "sources": [1]}}]
}}

Rules:
- Do not infer dates, amounts, obligations, or risks.
- Every event and risk must contain one or more valid source numbers.
- Sort dated events chronologically when their dates make that possible.
- Use an empty list when evidence is absent.

Sources:
{sources}"""
    try:
        response = await asyncio.to_thread(
            _get_client().chat.completions.create,
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
            reasoning_format="hidden",
        )
        data = _parse_json(response.choices[0].message.content or "{}")
    except (Exception, json.JSONDecodeError) as exc:
        print(f"[timeline error] {exc}")
        return {
            "briefing": "The evidence analysis service is temporarily unavailable.",
            "citations": [],
            "has_timeline": False,
        }

    valid_source_ids = set(range(1, len(hits) + 1))

    def valid_sources(values: object) -> list[int]:
        if not isinstance(values, list):
            return []
        return [value for value in values if isinstance(value, int) and value in valid_source_ids]

    events = data.get("events", []) if isinstance(data.get("events"), list) else []
    risks = data.get("risks_or_open_questions", []) if isinstance(data.get("risks_or_open_questions"), list) else []
    used_sources: set[int] = set()
    lines = [str(data.get("headline") or "Evidence timeline")]

    for event in events[:12]:
        if not isinstance(event, dict):
            continue
        refs = valid_sources(event.get("sources"))
        if not refs or not event.get("event"):
            continue
        used_sources.update(refs)
        pages = ", ".join(str(hits[source - 1]["page_number"]) for source in refs)
        lines.append(f"• {event.get('when', 'Undated')}: {event['event']} — {event.get('why_it_matters', '')} (p. {pages})")

    risk_lines = []
    for risk in risks[:6]:
        if not isinstance(risk, dict):
            continue
        refs = valid_sources(risk.get("sources"))
        if not refs or not risk.get("item"):
            continue
        used_sources.update(refs)
        pages = ", ".join(str(hits[source - 1]["page_number"]) for source in refs)
        risk_lines.append(f"• {risk['item']} (p. {pages})")
    if risk_lines:
        lines.extend(["", "Risks / open questions:", *risk_lines])

    citations = [
        {
            "doc_name": hits[source - 1]["doc_name"],
            "page_number": hits[source - 1]["page_number"],
            "image_filename": hits[source - 1]["image_filename"],
            "doc_id": hits[source - 1]["doc_id"],
            "excerpt": hits[source - 1]["chunk_text"][:240] + "...",
        }
        for source in sorted(used_sources)
    ]
    return {
        "briefing": "\n".join(lines),
        "citations": citations,
        "has_timeline": bool(used_sources),
    }
