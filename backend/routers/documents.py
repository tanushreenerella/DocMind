from fastapi import APIRouter, Depends
from core.auth import get_current_user
from core.database import get_pool

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
