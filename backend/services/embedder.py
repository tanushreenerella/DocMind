import chromadb
from chromadb.config import Settings
from core.config import CHROMA_PATH
from services.embeddings import get_embedding, get_embeddings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Collection has no embedding_function — we supply embeddings explicitly
# so no local model is loaded and no ONNX runtime is needed.
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


async def index_document(
    doc_id: str, doc_name: str, pages: list[dict], classification: dict
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


async def search(query: str, n_results: int = 5, doc_id: str | None = None) -> list[dict]:
    try:
        total = collection.count()
        if total == 0:
            return []
        safe_n = min(n_results, total)

        query_embedding = await get_embedding(query)

        where_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": safe_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        hits: list[dict] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append(
                    {
                        "chunk_text": doc,
                        "doc_name": meta["doc_name"],
                        "page_number": meta["page_number"],
                        "image_filename": meta["image_filename"],
                        "relevance_score": 1 - dist,
                        "doc_id": meta["doc_id"],
                    }
                )
        return hits
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
