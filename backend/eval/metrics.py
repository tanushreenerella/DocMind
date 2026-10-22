"""Pure scoring helpers: question validation, ranking metrics, percentiles, declines.

Relevance is page-level: a retrieved chunk is relevant when it comes from the
question's document and its page is in ``expected_pages``. A page usually
holds several chunks, so recall and nDCG count each expected page once (the
first chunk from it); MRR uses the first relevant chunk of any kind.
"""

import math
import re
from typing import Any, Iterable

QUESTION_TYPES = ("factual", "multi_part", "unanswerable")
QUESTION_FIELDS = ("question", "doc_name", "expected_pages", "reference_answer", "type")


class QuestionError(ValueError):
    pass


def validate_questions(items: Any) -> list[dict]:
    """Check questions.json's shape; returns the items, or raises QuestionError listing every problem."""
    if not isinstance(items, list):
        raise QuestionError("questions file must contain a JSON list")
    problems: list[str] = []
    for index, item in enumerate(items, start=1):
        where = f"question {index}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object")
            continue
        missing = [field for field in QUESTION_FIELDS if field not in item]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        for field in ("question", "doc_name", "reference_answer"):
            if not isinstance(item[field], str) or not item[field].strip():
                problems.append(f"{where}: {field} must be a non-empty string")
        if item["type"] not in QUESTION_TYPES:
            problems.append(f"{where}: type must be one of {', '.join(QUESTION_TYPES)}")
        pages = item["expected_pages"]
        if not isinstance(pages, list) or not all(
            isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages
        ):
            problems.append(f"{where}: expected_pages must be a list of page numbers (1-based)")
        elif item["type"] == "unanswerable" and pages:
            problems.append(f"{where}: unanswerable questions must have expected_pages []")
        elif item["type"] != "unanswerable" and not pages:
            problems.append(f"{where}: {item['type']} questions need at least one expected page")
    if problems:
        raise QuestionError("invalid questions:\n  " + "\n  ".join(problems))
    return items


# ── retrieval ───────────────────────────────────────────────────────────────

def is_relevant(hit: dict, doc_name: str, expected_pages: Iterable[int]) -> bool:
    return hit.get("doc_name") == doc_name and hit.get("page_number") in set(expected_pages)


def recall_at_k(hits: list[dict], doc_name: str, expected_pages: list[int], k: int) -> float:
    """Share of the expected pages that appear in the top ``k`` chunks."""
    expected = set(expected_pages)
    found = {h["page_number"] for h in hits[:k] if is_relevant(h, doc_name, expected)}
    return len(found) / len(expected)


def mrr_at_k(hits: list[dict], doc_name: str, expected_pages: list[int], k: int) -> float:
    for rank, hit in enumerate(hits[:k], start=1):
        if is_relevant(hit, doc_name, expected_pages):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[dict], doc_name: str, expected_pages: list[int], k: int) -> float:
    """Binary gain for the first chunk from each expected page; later ones from the same page gain 0."""
    expected = set(expected_pages)
    seen: set[int] = set()
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        page = hit.get("page_number")
        if is_relevant(hit, doc_name, expected) and page not in seen:
            seen.add(page)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(expected)) + 1))
    return dcg / ideal


# ── aggregation ─────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None and math.isfinite(v)]
    return sum(present) / len(present) if present else None


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ── declines ────────────────────────────────────────────────────────────────

# Phrases the answer models use when the sources don't cover the question.
_DECLINE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:don't|do not|doesn't|does not|didn't|did not|couldn't|could not|can't|cannot|unable to)\b"
        r"[^.?!]{0,80}\b(?:find|contain|include|mention|provide|cover|specify|state|answer|determine|say)",
        r"\bnot (?:mentioned|specified|provided|covered|stated|included|available|found|addressed)\b",
        r"\bno (?:information|mention|details|data|reference|relevant)\b",
        r"\b(?:insufficient|not enough) (?:information|context|detail)",
        r"\bdon't have any documents\b",
    )
]


def decline_reason(answer: str | None, has_answer: bool | None) -> str | None:
    """Why ``answer`` counts as declining to answer, or None if it answers.

    has_answer=False is the pipelines' own "I can't answer" signal; otherwise
    the text is matched against common refusal phrasing (a heuristic: the
    reason is saved per question so it can be checked by hand).
    """
    if has_answer is False:
        return "has_answer=false"
    for pattern in _DECLINE_PATTERNS:
        match = pattern.search(answer or "")
        if match:
            return f"matched: {match.group(0)!r}"
    return None
