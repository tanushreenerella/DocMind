import asyncio
import functools
import os
import threading
import chromadb
import httpx
import snowballstemmer
from rank_bm25 import BM25Okapi
from core.config import CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
from core.config import HYBRID_SEARCH_ENABLED, RERANK_ENABLED
from core.config import RERANK_CANDIDATES, RERANK_MODEL
from services.embeddings import get_embedding, get_embeddings
from services.errors import SearchUnavailable
import re

CHUNK_SIZE = 500
CHROMA_WRITE_ATTEMPTS = 3
CHROMA_RETRY_BASE_SECONDS = 1.0
# Network-level failures only. Chroma's HTTP error responses (bad request,
# quota) arrive as other exception types and would fail the same way again.
_TRANSIENT_CHROMA_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError)
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

# Words, plus terms joined by "-" or "/" kept whole: IL-6, COVID-19, tools/list.
_BM25_TOKEN_PATTERN = re.compile(r"\w+(?:[-/]\w+)*", re.UNICODE)
_BM25_JOINERS = re.compile(r"[-/]")
# Standard English stop words (NLTK's list, minus contraction fragments).
_BM25_STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())
_porter = snowballstemmer.stemmer("porter")
# Stemmer objects keep per-call state; the index can be rebuilt off the event loop.
_porter_lock = threading.Lock()
_RRF_K = 60
_bm25_index: BM25Okapi | None = None
_bm25_records: list[dict] = []
_bm25_ready = False
_reranker = None
_reranker_lock = threading.Lock()

def _validate_rerank_model(model_name: str) -> None:
    """Fail fast on names FlashRank doesn't ship (Ranker would raise KeyError)."""
    from flashrank.Config import model_file_map

    if model_name not in model_file_map:
        raise RuntimeError(
            f"RERANK_MODEL={model_name!r} is not a FlashRank model; "
            f"choose one of: {', '.join(sorted(model_file_map))}"
        )


# The model loads lazily, so check the name at startup rather than on the
# first query.
if RERANK_ENABLED:
    _validate_rerank_model(RERANK_MODEL)


def _get_reranker():
    """Load the cross-encoder on first use; blocking, so call it off the event loop.

    FlashRank runs it on ONNX Runtime (no torch/transformers), which keeps
    memory low enough for a 512MB instance. A cold cache downloads the model.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from flashrank import Ranker

                print(f"[rerank] loading {RERANK_MODEL}")
                _reranker = Ranker(
                    model_name=RERANK_MODEL,
                    cache_dir=os.environ.get("FLASHRANK_CACHE_DIR", "/tmp/flashrank"),
                )
    return _reranker

# Collection has no embedding_function — we supply embeddings explicitly
# so no local model is loaded and no ONNX runtime is needed.
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    # Chroma 1.x replaced the legacy metadata={"hnsw:space": ...} with
    # `configuration`, which is the form Chroma documents for both local and
    # Cloud. Distance is immutable once the collection exists.
    configuration={"hnsw": {"space": "cosine"}},
)


def _assert_cosine_space(chroma_collection) -> None:
    """Fail fast unless the collection really uses cosine distance.

    relevance_score = 1 - distance (and the RAG relevance thresholds built on
    it) is only meaningful for cosine. If the server ignored the requested
    space, or the collection was created earlier with another one, a wrong
    metric would silently corrupt every score, so refuse to start instead.
    """
    index_config = chroma_collection.configuration_json or {}
    index = index_config.get("hnsw") or index_config.get("spann") or {}
    space = index.get("space")
    if space != "cosine":
        raise RuntimeError(
            f"Chroma collection {chroma_collection.name!r} reports distance "
            f"space {space!r}, expected 'cosine'. Delete the collection in "
            "the Chroma Cloud dashboard (distance cannot be changed after "
            "creation), re-upload documents, and restart."
        )


_assert_cosine_space(collection)


@functools.lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    """Porter stem, except words with digits (p53, 2020s, h2o), which stay exact."""
    if any(char.isdigit() for char in word):
        return word
    with _porter_lock:
        return _porter.stemWord(word)


def _tokenize_for_bm25(text: str) -> list[str]:
    """BM25 terms only (embeddings always see the raw text).

    Lowercased; a joined term such as "IL-6" or "tools/list" yields the whole
    term and each part ("il-6", "il", "6"), so either form of a query matches.
    Stop words are dropped; everything else is Porter-stemmed unless it has
    digits ("proteins" -> "protein", but "p53" stays "p53").
    """
    tokens: list[str] = []
    for term in _BM25_TOKEN_PATTERN.findall(text.casefold()):
        parts = [part for part in _BM25_JOINERS.split(term) if part]
        if len(parts) > 1:
            # Rebuild the whole term from stemmed parts, keeping its separators.
            separators = _BM25_JOINERS.findall(term)
            whole = _stem(parts[0]) + "".join(
                sep + _stem(part) for sep, part in zip(separators, parts[1:])
            )
            tokens.append(whole)
        tokens.extend(_stem(part) for part in parts if part not in _BM25_STOP_WORDS)
    return tokens


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


async def _rerank_hits(
    query: str,
    hits: list[dict],
    n_results: int,
) -> list[dict]:
    """Rerank the leading RRF candidates with the shared cross-encoder."""
    candidates = hits[:RERANK_CANDIDATES]
    print(
        "[rerank] BEFORE:",
        [(h["doc_name"], h["page_number"], round(h["relevance_score"], 3)) for h in candidates[:3]],
    )

    from flashrank import RerankRequest

    request = RerankRequest(
        query=query,
        passages=[
            {"id": index, "text": hit["chunk_text"]}
            for index, hit in enumerate(candidates)
        ],
    )
    # Model loading (first call) and scoring are both CPU/IO-bound, so run them
    # in a worker thread. FlashRank returns passages sorted best first.
    try:
        ranked = await asyncio.to_thread(lambda: _get_reranker().rerank(request))
    except Exception as exc:
        # Keep the fused order rather than failing the whole search.
        print(f"[rerank] failed, using fused order: {exc}")
        return hits[:n_results]
    reranked = [
        {**candidates[item["id"]], "relevance_score": float(item["score"])}
        for item in ranked
    ]

    print(
        "[rerank] AFTER: ",
        [(h["doc_name"], h["page_number"], round(h["relevance_score"], 3)) for h in reranked[:3]],
    )

    return reranked[:n_results]


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
# After . ! ? (and any closing quotes/brackets, group 1), whitespace, then
# something that can start a sentence. Abbreviations ("e.g. The") may split
# too; that only makes the units finer.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])([\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9])")
Span = tuple[int, int]


def _strip_span(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_long(text: str, start: int, end: int) -> list[Span]:
    """Cut a span over CHUNK_SIZE at a line break, else a space, else anywhere."""
    pieces: list[Span] = []
    while end - start > CHUNK_SIZE:
        window = text[start:start + CHUNK_SIZE]
        cut = window.rfind("\n")
        if cut < CHUNK_SIZE // 4:
            cut = max(window.rfind(" "), window.rfind("\t"))
        if cut < CHUNK_SIZE // 4:
            cut = CHUNK_SIZE
        if span := _strip_span(text, start, start + cut):
            pieces.append(span)
        start += cut
    if span := _strip_span(text, start, end):
        pieces.append(span)
    return pieces


def _sentence_units(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, ends_paragraph) for every sentence, each at most CHUNK_SIZE long."""
    units: list[tuple[int, int, bool]] = []
    paragraph_start = 0
    breaks = [m.span() for m in _PARAGRAPH_BREAK.finditer(text)] + [(len(text), len(text))]
    for paragraph_end, next_start in breaks:
        spans: list[Span] = []
        sentence_start = paragraph_start
        for match in _SENTENCE_BREAK.finditer(text, paragraph_start, paragraph_end):
            if span := _strip_span(text, sentence_start, match.end(1)):
                spans.extend(_split_long(text, *span))
            sentence_start = match.end()
        if span := _strip_span(text, sentence_start, paragraph_end):
            spans.extend(_split_long(text, *span))
        units.extend((s, e, i == len(spans) - 1) for i, (s, e) in enumerate(spans))
        paragraph_start = next_start
    return units


def _overlap_start(text: str, units: list[tuple[int, int, bool]], chunk: Span, next_unit: Span) -> int:
    """Where the next chunk starts: up to CHUNK_OVERLAP chars back into ``chunk``,
    at a sentence start if one is that close, else at a word start."""
    chunk_start, chunk_end = chunk
    floor = max(chunk_start + 1, chunk_end - CHUNK_OVERLAP)
    candidates = [s for s, _, _ in units if floor <= s < chunk_end]
    if not candidates:
        candidates = [
            p for p in range(floor, chunk_end) if text[p - 1].isspace() and not text[p].isspace()
        ]
    for start in candidates:  # earliest first: the longest overlap that still fits
        if next_unit[1] - start <= CHUNK_SIZE:
            return start
    return next_unit[0]


def chunk_text(text: str) -> list[str]:
    """Split a page into chunks of whole sentences, about CHUNK_SIZE chars each.

    Each chunk is an exact slice of ``text``. Chunks end at sentence ends, and
    at a paragraph end once at least half full; a sentence longer than
    CHUNK_SIZE is cut at a line break or space. Consecutive chunks overlap by
    up to CHUNK_OVERLAP chars, starting at a sentence or word boundary.
    """
    units = _sentence_units(text)
    chunks: list[str] = []
    i = 0
    start = units[0][0] if units else 0
    while i < len(units):
        j = i
        while j + 1 < len(units):
            if units[j + 1][1] - start > CHUNK_SIZE:
                break
            if units[j][2] and units[j][1] - start >= CHUNK_SIZE // 2:
                break  # a paragraph ends here and the chunk is substantial
            j += 1
        end = units[j][1]
        chunks.append(text[start:end])
        i = j + 1
        if i < len(units):
            start = _overlap_start(text, units, (start, end), units[i][:2])
    return chunks


async def _write_chunks_with_retry(**batch) -> None:
    """Write one batch, retrying transport errors with exponential backoff.

    Uses upsert, not add: a request that times out may still have been applied,
    and chunk ids are deterministic, so a retried upsert just rewrites the same rows.
    """
    for attempt in range(1, CHROMA_WRITE_ATTEMPTS + 1):
        try:
            await asyncio.to_thread(collection.upsert, **batch)
            return
        except _TRANSIENT_CHROMA_ERRORS as exc:
            if attempt == CHROMA_WRITE_ATTEMPTS:
                raise
            delay = CHROMA_RETRY_BASE_SECONDS * 2 ** (attempt - 1)
            print(
                f"[chroma] write attempt {attempt}/{CHROMA_WRITE_ATTEMPTS} failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)


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
        await _write_chunks_with_retry(
            ids=ids[i : i + BATCH_SIZE],
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    _invalidate_bm25_index()


def delete_document_chunks(doc_id: str, user_id: str) -> None:
    """Remove every chunk of one document; filtered on the owner too, never doc_id alone."""
    collection.delete(
        where={"$and": [{"doc_id": {"$eq": doc_id}}, {"user_id": {"$eq": user_id}}]}
    )
    _invalidate_bm25_index()


def _vector_search(
    query_embedding: list[float],
    n_results: int,
    user_id: str,
    doc_id: str | None,
) -> list[dict]:
    """Nearest chunks by cosine similarity, owner-filtered server-side. Blocking."""
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
        "n_results": n_results,
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
    return hits


async def search_or_raise(
    query: str,
    n_results: int = 5,
    doc_id: str | None = None,
    *,
    user_id: str,
) -> list[dict]:
    """``search``, but failures raise the underlying error (used by eval)."""
    if not user_id:
        return []

    total = collection.count()
    if total == 0:
        return []
    safe_n = min(
        max(n_results, RERANK_CANDIDATES),
        total,
    ) if HYBRID_SEARCH_ENABLED and RERANK_ENABLED else min(n_results, total)

    query_embedding = await get_embedding(query)
    hits = _vector_search(query_embedding, safe_n, user_id, doc_id)

    if not HYBRID_SEARCH_ENABLED:
        return hits

    bm25_hits = _search_bm25(
        query,
        n_results=safe_n,
        user_id=user_id,
        doc_id=doc_id,
    )
    fused_hits = _fuse_with_rrf(
        hits,
        bm25_hits,
        n_results=safe_n,
    )
    if RERANK_ENABLED:
        print("[rerank] cross-encoder reranking triggered")
        return await _rerank_hits(query, fused_hits, n_results)
    return fused_hits[:n_results]


async def search(
    query: str,
    n_results: int = 5,
    doc_id: str | None = None,
    *,
    user_id: str,
) -> list[dict]:
    """Search only chunks owned by ``user_id`` (and optionally one document).

    Raises SearchUnavailable when Chroma (or the query embedding) fails, so an
    outage isn't mistaken for having no documents.
    """
    try:
        return await search_or_raise(query, n_results, doc_id, user_id=user_id)
    except Exception as exc:
        print(f"[search error] {type(exc).__name__}: {exc}")
        raise SearchUnavailable() from exc


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
