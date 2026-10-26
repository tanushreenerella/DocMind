"""What one user has indexed in Chroma: documents, their ids and page texts."""

from collections import defaultdict
from dataclasses import dataclass, field

from services import embedder


@dataclass
class IndexedDoc:
    doc_name: str
    doc_id: str
    # page number -> chunks in order
    chunks: dict[int, list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))

    @property
    def pages(self) -> list[int]:
        return sorted(self.chunks)

    def page_text(self, page: int) -> str:
        """The page's text, rebuilt from its chunks (dropping the chunk overlap)."""
        ordered = [text for _, text in sorted(self.chunks[page])]
        if not ordered:
            return ""
        rebuilt = ordered[0]
        for text in ordered[1:]:
            overlap = _overlap(rebuilt, text)
            rebuilt += text[overlap:] if overlap else " " + text
        return rebuilt


def _overlap(previous: str, chunk: str) -> int:
    """How many leading chars of ``chunk`` repeat the end of ``previous``.

    Chunks indexed before sentence chunking overlap by exactly CHUNK_OVERLAP
    chars, mid-word. Newer ones overlap by up to CHUNK_OVERLAP chars starting
    at a word boundary, or not at all (e.g. when a long sentence follows).
    """
    if len(chunk) > embedder.CHUNK_OVERLAP and previous.endswith(chunk[: embedder.CHUNK_OVERLAP]):
        return embedder.CHUNK_OVERLAP
    for size in range(min(embedder.CHUNK_OVERLAP, len(previous), len(chunk)), 0, -1):
        starts_a_word = size == len(previous) or previous[-size - 1].isspace()
        ends_a_unit = size == len(chunk) or chunk[size].isspace()
        if starts_a_word and ends_a_unit and previous.endswith(chunk[:size]):
            return size
    return 0


def load_user_docs(user_id: str) -> dict[str, IndexedDoc]:
    """Every document ``user_id`` has in Chroma, by doc_name.

    Raises if two of the user's documents share a name: questions refer to
    documents by name, so that would be ambiguous.
    """
    stored = embedder.collection.get(
        where={"user_id": {"$eq": user_id}},
        include=["documents", "metadatas"],
    )
    docs: dict[str, IndexedDoc] = {}
    for text, meta in zip(stored.get("documents") or [], stored.get("metadatas") or []):
        name, doc_id = meta["doc_name"], meta["doc_id"]
        doc = docs.setdefault(name, IndexedDoc(name, doc_id))
        if doc.doc_id != doc_id:
            raise RuntimeError(
                f"user has two documents named {name!r} ({doc.doc_id}, {doc_id}); "
                "rename or delete one so questions can refer to it unambiguously"
            )
        doc.chunks[int(meta["page_number"])].append((int(meta.get("chunk_index", 0)), text))
    return docs
