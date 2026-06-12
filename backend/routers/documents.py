from fastapi import APIRouter
from services.embedder import collection
from services.embedder import get_document_count

router = APIRouter()

@router.get("/documents")
async def list_documents():
    results = collection.get(include=["metadatas"])
    
    seen = {}
    for meta in results["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "doc_name": meta["doc_name"],
                "document_type": meta.get("document_type", "other"),
                "sensitivity_level": meta.get("sensitivity_level", "internal"),
            }
    
    return {"documents": list(seen.values())}

@router.get("/documents/count")
async def document_count() -> dict:
    count = await get_document_count()
    return {"chunk_count": count}
