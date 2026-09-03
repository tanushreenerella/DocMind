import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from core.config import CHROMA_PATH
from core.config import HYBRID_SEARCH_ENABLED
from services.embeddings import get_embedding, get_embeddings
import re

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Monitoring only: suspicious content remains searchable, but is recorded so
# uploads containing likely prompt-injection attempts can be investigated.
_PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "disregard previous instructions",
    "system prompt",
    "you are now",
)

_BM25_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60
_bm25_index: BM25Okapi | None = None
_bm25_records: list[dict] = []
_bm25_ready = False

# Collection has no embedding_function — we supply embeddings explicitly
# so no local model is loaded and no ONNX runtime is needed.
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    # Disable/avoid sending telemetry from the Chroma client to prevent
    # compatibility issues with local telemetry integrations.
    settings=Settings(anonymized_telemetry=True),
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def _tokenize_for_bm25(text: str) -> list[str]:
    return _BM25_TOKEN_PATTERN.findall(text.casefold())


def _invalidate_bm25_index() -> None:
    """Ensure the next hybrid query includes chunks indexed since the cache."""
    global _bm25_index, _bm25_records, _bm25_ready
    _bm25_index = None
    _bm25_records = []
    _bm25_ready = False


def _rebuild_bm25_index() -> None:
    """Recreate the in-memory keyword index from persisted Chroma chunks."""
    global _bm25_index, _bm25_records, _bm25_ready

    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents", []) or []
    metadatas = stored.get("metadatas", []) or []
    chunk_ids = stored.get("ids", []) or []

    _bm25_records = [
        {
            "chunk_id": chunk_ids[index] if index < len(chunk_ids) else "unknown",
            "chunk_text": document,
            "metadata": metadata or {},
            "tokens": _tokenize_for_bm25(document),
        }
        for index, (document, metadata) in enumerate(
            zip(documents, metadatas)
        )
    ]
    corpus = [record["tokens"] for record in _bm25_records]
    _bm25_index = BM25Okapi(corpus) if corpus else None
    _bm25_ready = True
    print(f"[bm25] rebuilt keyword index from {len(_bm25_records)} Chroma chunks")


def _warn_on_prompt_injection(
    chunk_text: str,
    metadata: dict,
    chunk_id: str,
) -> None:
    matched_pattern = next(
        (
            pattern
            for pattern in _PROMPT_INJECTION_PATTERNS
            if pattern in chunk_text.casefold()
        ),
        None,
    )
    if matched_pattern:
        print(
            "[prompt-injection warning] "
            f"doc_id={metadata.get('doc_id', 'unknown')!r} "
            f"chunk_id={chunk_id!r} "
            f"pattern={matched_pattern!r}"
        )


def _make_hit(
    chunk_text: str,
    metadata: dict,
    chunk_id: str,
    relevance_score: float,
) -> dict:
    _warn_on_prompt_injection(chunk_text, metadata, chunk_id)
    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "doc_name": metadata["doc_name"],
        "page_number": metadata["page_number"],
        "image_filename": metadata["image_filename"],
        "relevance_score": relevance_score,
        "doc_id": metadata["doc_id"],
    }


def _search_bm25(
    query: str,
    n_results: int,
    user_id: str,
    doc_id: str | None,
) -> list[dict]:
    """Return keyword matches after applying the same ownership filters."""
    if not _bm25_ready:
        _rebuild_bm25_index()
    if _bm25_index is None:
        return []

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return []

    scores = _bm25_index.get_scores(query_tokens)
    candidates = [
        (score, record)
        for score, record in zip(scores, _bm25_records)
        # BM25's IDF can be negative for very common terms, so use token
        # membership to distinguish genuine keyword matches from zero-score
        # non-matches before applying the BM25 ordering.
        if any(token in record["tokens"] for token in query_tokens)
        and record["metadata"].get("user_id") == user_id
        and (doc_id is None or record["metadata"].get("doc_id") == doc_id)
    ]
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    # BM25 scores are not comparable with Chroma cosine scores. RRF uses
    # ranks instead, while 1.0 keeps existing RAG's relevance threshold for
    # BM25-only hits.
    return [
        _make_hit(
            record["chunk_text"],
            record["metadata"],
            record["chunk_id"],
            relevance_score=1.0,
        )
        for _, record in candidates[:n_results]
    ]


def _fuse_with_rrf(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    n_results: int,
) -> list[dict]:
    """Fuse two rankings using reciprocal rank fusion without reranking."""
    scores: dict[str, float] = {}
    hits_by_id: dict[str, dict] = {}
    best_rank: dict[str, int] = {}

    for ranking in (vector_hits, bm25_hits):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1 / (_RRF_K + rank))
            hits_by_id.setdefault(chunk_id, hit)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_rank[chunk_id],
            chunk_id,
        ),
    )
    return [hits_by_id[chunk_id] for chunk_id in ordered_ids[:n_results]]


def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


async def index_document(
    doc_id: str,
    user_id: str,
    doc_name: str,
    pages: list[dict],
    classification: dict,
) -> None:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for page in pages:
        if not page["text"].strip():
            continue
        chunks = chunk_text(page["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_id}_p{page['page_number']}_c{i}")
            documents.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "doc_name": doc_name,
                    "page_number": page["page_number"],
                    "image_filename": page["image_filename"],
                    "sensitivity_level": classification.get("sensitivity_level", "internal"),
                    "document_type": classification.get("document_type", "other"),
                    "chunk_index": i,
                }
            )

    if not ids:
        return

    BATCH_SIZE = 20
    for i in range(0, len(ids), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_embeddings = await get_embeddings(batch_docs)
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    _invalidate_bm25_index()


async def search(
    query: str,
    n_results: int = 5,
    doc_id: str | None = None,
    *,
    user_id: str,
) -> list[dict]:
    """Search only chunks owned by ``user_id`` (and optionally one document)."""
    try:
        if not user_id:
            return []

        total = collection.count()
        if total == 0:
            return []
        safe_n = min(n_results, total)

        query_embedding = await get_embedding(query)

        where_clauses = [{"user_id": {"$eq": user_id}}]
        if doc_id:
            where_clauses.append({"doc_id": {"$eq": doc_id}})

        # Chroma applies every metadata predicate server-side. Existing chunks
        # without user_id do not match and therefore cannot be returned.
        where_filter = (
            {"$and": where_clauses}
            if len(where_clauses) > 1
            else where_clauses[0]
        )
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": safe_n,
            "include": ["documents", "metadatas", "distances"],
        }
        query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        hits: list[dict] = []
        if results["documents"] and results["documents"][0]:
            chunk_ids = results.get("ids", [[]])[0] or []
            for index, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                chunk_id = (
                    chunk_ids[index]
                    if index < len(chunk_ids)
                    else "unknown"
                )
                hits.append(
                    _make_hit(
                        doc,
                        meta,
                        chunk_id,
                        relevance_score=1 - dist,
                    )
                )

        if not HYBRID_SEARCH_ENABLED:
            return hits

        bm25_hits = _search_bm25(
            query,
            n_results=n_results,
            user_id=user_id,
            doc_id=doc_id,
        )
        return _fuse_with_rrf(
            hits,
            bm25_hits,
            n_results=n_results,
        )
    except Exception as exc:
        print(f"[search error] {exc}")
        return []


async def get_document_count() -> int:
    try:
        return collection.count()
    except Exception:
        return 0


async def list_indexed_documents() -> list[dict]:
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        seen: dict[str, dict] = {}
        for m in all_meta:
            doc_id = m.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": m.get("doc_name", ""),
                    "document_type": m.get("document_type", "other"),
                    "sensitivity_level": m.get("sensitivity_level", "internal"),
                }
        return list(seen.values())
    except Exception:
        return []
