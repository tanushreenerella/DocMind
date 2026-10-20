import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status
from core.auth import get_current_user
from core.database import get_pool
from services import storage
from services.embedder import delete_document_chunks
from services.timeline import build_evidence_timeline

router = APIRouter()


@router.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT doc_id, doc_name, document_type, sensitivity_level
        FROM documents
        WHERE user_id = $1 AND status = 'complete'
        ORDER BY created_at DESC
        """,
        current_user["id"],
    )
    return {
        "documents": [
            {
                "doc_id": r["doc_id"],
                "doc_name": r["doc_name"],
                "document_type": r["document_type"] or "other",
                "sensitivity_level": r["sensitivity_level"] or "internal",
            }
            for r in rows
        ]
    }


@router.get("/documents/count")
async def document_count(current_user: dict = Depends(get_current_user)):
    pool = get_pool()
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE user_id = $1 AND status = 'complete'",
        current_user["id"],
    )
    return {"count": count}


@router.post("/documents/{doc_id}/evidence-timeline")
async def evidence_timeline(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Produce an evidence-backed timeline only after verifying document ownership."""
    pool = get_pool()
    document = await pool.fetchrow(
        "SELECT doc_id FROM documents WHERE doc_id = $1 AND user_id = $2 AND status = 'complete'",
        doc_id,
        current_user["id"],
    )
    if not document:
        raise HTTPException(404, "Document not found")
    return await build_evidence_timeline(
        doc_id,
        user_id=current_user["id"],
    )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document's chunks, stored files and rows, owner only.

    External data goes first and the DB row last, so a failure part-way leaves
    the row in place and the (idempotent) delete can simply be retried.
    """
    user_id = current_user["id"]
    pool = get_pool()
    document = await pool.fetchrow(
        "SELECT status FROM documents WHERE doc_id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    if not document:
        raise HTTPException(404, "Document not found")
    if document["status"] in ("queued", "processing"):
        # The background job would re-create chunks and files after we delete them.
        raise HTTPException(409, "Document is still processing; try again when it finishes")

    await asyncio.to_thread(delete_document_chunks, doc_id, user_id)
    await asyncio.to_thread(storage.delete_prefix, storage.document_prefix(user_id, doc_id))
    await pool.execute(
        "DELETE FROM chat_history WHERE user_id = $1 AND doc_id = $2", user_id, doc_id
    )
    await pool.execute(
        "DELETE FROM documents WHERE doc_id = $1 AND user_id = $2", doc_id, user_id
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
