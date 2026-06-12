"""
Classifies a document using Groq LLaMA-3.
Returns structured JSON with document type, sensitivity, entities, summary.
"""
import json

from groq import Groq

from core.config import settings

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

CLASSIFICATION_SCHEMA = """
{
  "document_type": "invoice | report | contract | academic_paper | handwritten_note | medical_record | legal | financial | technical_manual | other",
  "primary_topic": "string (e.g. 'quarterly sales figures', 'patient diagnosis')",
  "content_characteristics": {
    "has_tables": boolean,
    "has_handwriting": boolean,
    "has_images": boolean,
    "is_scanned": boolean,
    "language": "string",
    "estimated_formality": "formal | informal | mixed"
  },
  "sensitivity_level": "public | internal | confidential | restricted",
  "sensitivity_reason": "string explaining why this sensitivity level was assigned",
  "key_entities": ["list of important names, dates, organizations mentioned"],
  "summary": "2-3 sentence summary of the document"
}
"""

_FALLBACK: dict = {
    "document_type": "other",
    "primary_topic": "unknown",
    "sensitivity_level": "internal",
    "sensitivity_reason": "Auto-classification failed",
    "summary": "Classification could not be completed.",
    "content_characteristics": {
        "has_tables": False,
        "has_handwriting": False,
        "has_images": False,
        "is_scanned": False,
        "language": "unknown",
        "estimated_formality": "mixed",
    },
    "key_entities": [],
}


async def classify_document(text: str, filename: str) -> dict:
    prompt = f"""You are a document classification expert. Analyze this document and return ONLY a JSON object matching this schema exactly:

{CLASSIFICATION_SCHEMA}

Document filename: {filename}
Document text (first 3000 chars):
{text[:3000]}

Return ONLY the JSON object, no markdown, no explanation."""

    try:
        response = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except (json.JSONDecodeError, Exception):
        return _FALLBACK.copy()
