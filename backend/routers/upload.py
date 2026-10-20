import asyncio
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from core.auth import get_current_user
from core.config import STORAGE_PATH, RATE_LIMIT_UPLOAD
from core.database import get_pool
from core.rate_limit import limiter
from core.security import sanitize_filename, validate_file
from services.classifier import classify_document
from services.embedder import index_document
from services import storage
from services.parser import parse_document

router = APIRouter()


async def _set_status(doc_id: str, **fields) -> None:
    pool = get_pool()
    sets = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(fields))
    vals = list(fields.values())
    await pool.execute(f"UPDATE documents SET {sets} WHERE doc_id = $1", doc_id, *vals)


async def _process_document(
    doc_id: str,
    user_id: str,
    file_path: str,
    filename: str,
) -> None:
    image_dir = os.path.join(STORAGE_PATH, "temp", f"{doc_id}_pages")
    try:
        await _set_status(doc_id, stage="parsing", status="processing", progress=10)
        if Path(file_path).suffix.lower() == ".pdf":
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            await asyncio.to_thread(
                storage.put, storage.pdf_key(user_id, doc_id), pdf_bytes, "application/pdf"
            )

        pages = await parse_document(file_path, doc_id, image_dir)
        await _store_page_images(user_id, doc_id, pages)

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

        await index_document(
            doc_id,
            user_id,
            filename,
            pages,
            classification,
        )

        await _set_status(doc_id, stage="complete", status="complete", progress=100)

    except Exception as exc:
        print(f"[{doc_id}] ERROR: {exc}")
        await _set_status(doc_id, status="error", stage="error", error_message=str(exc))

    finally:
        # Everything worth keeping is in storage by now; temp files never are.
        if os.path.exists(file_path):
            os.remove(file_path)
        shutil.rmtree(image_dir, ignore_errors=True)


async def _store_page_images(user_id: str, doc_id: str, pages: list[dict]) -> None:
    for page in pages:
        if not page["image_filename"]:
            continue
        with open(page["image_path"], "rb") as f:
            image_bytes = f.read()
        await asyncio.to_thread(
            storage.put,
            storage.page_image_key(user_id, doc_id, page["image_filename"]),
            image_bytes,
            "image/jpeg",
        )


async def _stream_object(key: str, headers: dict[str, str]) -> StreamingResponse | None:
    obj = await asyncio.to_thread(storage.get, key)
    if obj is None:
        return None
    if obj.size is not None:
        headers = {**headers, "Content-Length": str(obj.size)}
    return StreamingResponse(obj.body, media_type=obj.content_type, headers=headers)


@router.post("/upload")
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    pool = get_pool()
    job_ids = []

    for file in files:
        if not await validate_file(file):
            raise HTTPException(400, "Only PDF files are supported")

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

        background_tasks.add_task(
            _process_document,
            doc_id,
            current_user["id"],
            temp_path,
            safe_name,
        )
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

    # Ownership is proven above, so the caller's id is the owner's id in the key.
    response = await _stream_object(
        storage.pdf_key(current_user["id"], doc_id),
        {"Content-Disposition": "inline", "Cache-Control": "private, no-store"},
    )
    if response is None:
        raise HTTPException(404, "PDF not found")
    return response


@router.get("/page-image/{image_filename}")
async def get_page_image(
    image_filename: str,
    current_user: dict = Depends(get_current_user),
):
    if not all(c.isalnum() or c in "-_." for c in image_filename):
        raise HTTPException(400, "Invalid filename")

    match = re.fullmatch(r"([0-9a-f-]+)_p\d+\.jpg", image_filename)
    if not match:
        raise HTTPException(400, "Invalid filename")

    pool = get_pool()
    owned_document = await pool.fetchval(
        "SELECT 1 FROM documents WHERE doc_id = $1 AND user_id = $2",
        match.group(1),
        current_user["id"],
    )
    if not owned_document:
        raise HTTPException(403, "Document access denied")

    response = await _stream_object(
        storage.page_image_key(current_user["id"], match.group(1), image_filename),
        # Shown repeatedly in chat citations; browser-only cache, briefly.
        {"Cache-Control": "private, max-age=300"},
    )
    if response is None:
        raise HTTPException(404, "Image not found")
    return response
