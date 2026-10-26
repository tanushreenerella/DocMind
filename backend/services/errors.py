"""Outages the answer pipelines report to the user instead of answering.

They are raised where the failure happens (a Groq call, a vector search) and
turned into an ``error_response`` at the pipeline's entry point, so an outage
never reads like "no documents" or "insufficient evidence".
"""


class ServiceUnavailable(Exception):
    """A dependency failed; the user should simply try again shortly."""

    code = "service_unavailable"
    message = "Something went wrong on our side. Please try again shortly."


class AIUnavailable(ServiceUnavailable):
    """A Groq call failed: 429, 5xx, timeout, connection error or an empty reply."""

    code = "ai_unavailable"
    message = "The AI service is busy or unavailable right now. Please try again shortly."


class SearchUnavailable(ServiceUnavailable):
    """Document search failed (Chroma or the embedding service), so nothing was searched."""

    code = "search_unavailable"
    message = (
        "Document search is unavailable right now, so I couldn't look through your "
        "documents. Please try again shortly."
    )


class AnswerTimeout(ServiceUnavailable):
    """The whole answer took longer than its deadline."""

    code = "timeout"
    message = "That took longer than expected. Please try again shortly."


def error_response(exc: ServiceUnavailable) -> dict:
    """The chat response for an outage: the usual shape plus an ``error`` code."""
    return {"answer": exc.message, "citations": [], "has_answer": False, "error": exc.code}
