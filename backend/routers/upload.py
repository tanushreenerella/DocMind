import os
import re
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.auth import get_current_user
from core.config import STORAGE_PATH, IMAGES_PATH
from core.database import get_pool
from core.security import sanitize_filename, validate_file
from services.classifier import classify_document
from services.embedder import index_document
from services.parser import parse_document

router = APIRouter()


async def _set_status(doc_id: str, **fields) -> None:
    pool = get_pool()
    sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(fields))
    vals = list(fields.values())
    await pool.execute(f"UPDATE documents SET {sets} WHERE doc_id = $1", doc_id, *vals)


async def _process_document(doc_id: str, file_path: str, filename: str) -> None:
    try:
        await _set_status(doc_id, stage="parsing", status="processing", progress=10)
        pages = await parse_document(file_path, doc_id)

        await _set_status(doc_id, page_count=len(pages), progress=50, stage="classifying")

        combined_text = " ".join(p["text"][:1000] for p in pages[:5])
        classification = await classify_document(combined_text, filename)

        await _set_status(
            doc_id,
            document_type=classification.get("document_type", "other"),
            sensitivity_level=classification.get("sensitivity_level", "internal"),
            summary=classification.get("summary", ""),
            progress=75,
            stage="indexing",
        )

        await index_document(doc_id, filename, pages, classification)

        await _set_status(doc_id, stage="complete", status="complete", progress=100)

    except Exception as exc:
        print(f"[{doc_id}] ERROR: {exc}")
        await _set_status(doc_id, status="error", stage="error", error_message=str(exc))

    finally:
        if Path(file_path).suffix.lower() == ".pdf" and os.path.exists(file_path):
            pdf_dir = os.path.join(STORAGE_PATH, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)
            dest = os.path.join(pdf_dir, f"{doc_id}.pdf")
            os.replace(file_path, dest)
        elif os.path.exists(file_path):
            os.remove(file_path)


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    pool = get_pool()
    job_ids = []

    for file in files:
        if not await validate_file(file):
            raise HTTPException(400, f"Invalid file: {file.filename}")

        doc_id = str(uuid.uuid4())
        safe_name = sanitize_filename(file.filename or "upload")

        temp_dir = os.path.join(STORAGE_PATH, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{doc_id}_{safe_name}")

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        await pool.execute(
            """
            INSERT INTO documents (doc_id, user_id, doc_name, status, stage, progress)
            VALUES ($1, $2, $3, 'queued', 'queued', 0)
            """,
            doc_id,
            current_user["id"],
            safe_name,
        )

        background_tasks.add_task(_process_document, doc_id, temp_path, safe_name)
        job_ids.append({"doc_id": doc_id, "filename": safe_name})

    return {"jobs": job_ids}


@router.get("/status/{doc_id}")
async def get_status(doc_id: str, current_user: dict = Depends(get_current_user)):
    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM documents WHERE doc_id = $1 AND user_id = $2",
        doc_id,
        current_user["id"],
    )
    if not row:
        raise HTTPException(404, "Job not found")

    classification = None
    if row["status"] == "complete":
        classification = {
            "document_type": row["document_type"] or "other",
            "primary_topic": row["summary"] or "",
            "sensitivity_level": row["sensitivity_level"] or "internal",
            "summary": row["summary"] or "",
        }

    return {
        "status": row["status"],
        "filename": row["doc_name"],
        "stage": row["stage"],
        "progress": row["progress"],
        "error": row["error_message"],
        "classification": classification,
        "page_count": row["page_count"],
    }


@router.get("/pdf/{doc_id}")
async def get_pdf(doc_id: str, current_user: dict = Depends(get_current_user)):
    if not re.match(r"^[0-9a-f-]+$", doc_id):
        raise HTTPException(400, "Invalid doc_id")

    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT doc_id FROM documents WHERE doc_id = $1 AND user_id = $2",
        doc_id,
        current_user["id"],
    )
    if not row:
        raise HTTPException(404, "Document not found")

    pdf_path = os.path.join(STORAGE_PATH, "pdfs", f"{doc_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")

    return FileResponse(pdf_path, media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


@router.get("/page-image/{image_filename}")
async def get_page_image(
    image_filename: str,
    current_user: dict = Depends(get_current_user),
):
    if not all(c.isalnum() or c in "-_." for c in image_filename):
        raise HTTPException(400, "Invalid filename")

    img_path = os.path.join(IMAGES_PATH, image_filename)
    abs_img = os.path.realpath(img_path)
    abs_dir = os.path.realpath(IMAGES_PATH)
    if not abs_img.startswith(abs_dir):
        raise HTTPException(400, "Invalid path")
    if not os.path.exists(abs_img):
        raise HTTPException(404, "Image not found")

    return FileResponse(abs_img, media_type="image/jpeg")
