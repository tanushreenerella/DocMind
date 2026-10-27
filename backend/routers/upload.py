"""Uploads: presign → browser PUTs the PDF → complete → background processing.

With R2 the browser PUTs straight into the bucket through a presigned URL, so
file bytes never pass through the Next.js proxy or this API. With local disk
(no R2 configured) the "presigned" URL is PUT /api/uploads/{doc_id}/file, so
the frontend flow is identical either way.
"""

import asyncio
import os
import re
import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.auth import get_current_user
from core.config import (
    MAX_PAGES,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    RATE_LIMIT_UPLOAD,
    STORAGE_PATH,
    UPLOAD_URL_EXPIRES_SECONDS,
)
from core.database import get_pool
from core.rate_limit import limiter
from core.token_budget import charged_to, over_budget
from core.security import sanitize_filename, validate_magic_bytes
from services.classifier import budget_skipped_classification, classify_document
from services.embedder import index_document
from services import storage
from services.parser import count_pdf_pages, parse_document

router = APIRouter()

# Statuses a finished job can end in; "partial" = indexed, but some pages' OCR failed.
TERMINAL_STATUSES = ("complete", "partial", "error")
# Before /complete: a document row whose original PDF may not be stored yet.
UPLOADING = "uploading"
PDF_CONTENT_TYPE = "application/pdf"


class PresignRequest(BaseModel):
    filename: str
    size: int
    content_type: str


class NothingReadableError(Exception):
    """OCR failed on every page, so there is nothing to index."""


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
    """Parse, classify and index ``file_path``: a temp copy of the stored original."""
    image_dir = os.path.join(STORAGE_PATH, "temp", f"{doc_id}_pages")
    try:
        await _set_status(doc_id, stage="parsing", status="processing", progress=10)
        pages = await parse_document(file_path, doc_id, image_dir)
        failed_pages = [p["page_number"] for p in pages if p.get("ocr_failed")]
        if pages and len(failed_pages) == len(pages):
            raise NothingReadableError(f"OCR failed on all {len(pages)} pages")
        await _store_page_images(user_id, doc_id, pages)

        await _set_status(doc_id, page_count=len(pages), progress=50, stage="classifying")

        if await over_budget(user_id):
            # Out of AI budget: still index the document, just without a Groq classification.
            print(f"[{doc_id}] daily token budget used up; skipping classification")
            classification = budget_skipped_classification()
        else:
            combined_text = " ".join(p["text"][:1000] for p in pages[:5])
            with charged_to(user_id):
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

        if failed_pages:
            print(f"[{doc_id}] indexed without OCR text for pages {failed_pages}")
        hidden_text_pages = [p["page_number"] for p in pages if p.get("hidden_text_removed")]
        if hidden_text_pages:
            print(f"[{doc_id}] suspicious: hidden text removed from pages {hidden_text_pages}")
        await _set_status(
            doc_id,
            stage="complete",
            status="partial" if failed_pages else "complete",
            failed_pages=failed_pages,
            hidden_text_pages=hidden_text_pages,
            hidden_text_removed=sum(p.get("hidden_text_removed", 0) for p in pages),
            suspicious=bool(hidden_text_pages),
            progress=100,
        )

    except Exception as exc:
        # Full detail stays server-side (log + error_message); /status never returns it.
        print(f"[{doc_id}] processing failed:\n{traceback.format_exc()}")
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


def _too_large(size: int) -> HTTPException:
    return HTTPException(
        413, f"File is too large ({size / (1024 * 1024):.1f} MB); the limit is {MAX_UPLOAD_MB} MB"
    )


async def _pending_upload(doc_id: str, user_id: str) -> dict:
    """The caller's document row, which must still be waiting for its file."""
    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(404, "Upload not found")
    row = await get_pool().fetchrow(
        "SELECT doc_name, status FROM documents WHERE doc_id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    if not row:
        raise HTTPException(404, "Upload not found")
    if row["status"] != UPLOADING:
        raise HTTPException(409, "This upload has already been completed")
    return row


async def _discard_upload(doc_id: str, user_id: str) -> None:
    """Drop a rejected upload: its stored file and its (never listed) row."""
    await asyncio.to_thread(storage.delete_prefix, storage.document_prefix(user_id, doc_id))
    await get_pool().execute(
        "DELETE FROM documents WHERE doc_id = $1 AND user_id = $2", doc_id, user_id
    )


@router.post("/uploads/presign")
@limiter.limit(RATE_LIMIT_UPLOAD)
async def presign_upload(
    request: Request,
    body: PresignRequest,
    current_user: dict = Depends(get_current_user),
):
    safe_name = sanitize_filename(body.filename)
    if Path(safe_name).suffix.lower() != ".pdf" or body.content_type != PDF_CONTENT_TYPE:
        raise HTTPException(400, "Only PDF files are supported")
    if body.size <= 0:
        raise HTTPException(400, "File is empty")
    if body.size > MAX_UPLOAD_BYTES:
        raise _too_large(body.size)

    user_id = current_user["id"]
    doc_id = str(uuid.uuid4())
    if storage.direct_uploads():
        upload_url = await asyncio.to_thread(
            storage.presign_put,
            storage.pdf_key(user_id, doc_id),
            PDF_CONTENT_TYPE,
            UPLOAD_URL_EXPIRES_SECONDS,
        )
    else:
        upload_url = f"/api/uploads/{doc_id}/file"

    await get_pool().execute(
        """
        INSERT INTO documents (doc_id, user_id, doc_name, status, stage, progress)
        VALUES ($1, $2, $3, 'uploading', 'uploading', 0)
        """,
        doc_id,
        user_id,
        safe_name,
    )
    return {
        "doc_id": doc_id,
        "filename": safe_name,
        "upload_url": upload_url,
        "method": "PUT",
        "headers": {"Content-Type": PDF_CONTENT_TYPE},
        "expires_in": UPLOAD_URL_EXPIRES_SECONDS,
    }


@router.put("/uploads/{doc_id}/file", status_code=204)
async def put_upload_file(
    doc_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Local-disk stand-in for the presigned PUT; R2 deployments never hand it out."""
    if storage.direct_uploads():
        raise HTTPException(404, "Not found")
    await _pending_upload(doc_id, current_user["id"])
    if request.headers.get("content-type") != PDF_CONTENT_TYPE:
        raise HTTPException(400, "Only PDF files are supported")

    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            raise _too_large(len(data))
    await asyncio.to_thread(
        storage.put, storage.pdf_key(current_user["id"], doc_id), bytes(data), PDF_CONTENT_TYPE
    )
    return Response(status_code=204)


@router.post("/uploads/{doc_id}/complete")
async def complete_upload(
    doc_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """Check the stored file (size, PDF header, page count), then start processing.

    Rejected files are deleted with their row. Unexpected failures (storage
    unreachable...) leave both in place so the call can be retried.
    """
    user_id = current_user["id"]
    row = await _pending_upload(doc_id, user_id)
    key = storage.pdf_key(user_id, doc_id)

    size = await asyncio.to_thread(storage.size, key)
    if size is None:
        raise HTTPException(400, "The file was not uploaded; please try again")
    if size > MAX_UPLOAD_BYTES:
        await _discard_upload(doc_id, user_id)
        raise _too_large(size)

    temp_dir = os.path.join(STORAGE_PATH, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{doc_id}_{row['doc_name']}")
    try:
        await asyncio.to_thread(storage.download, key, temp_path)
        with open(temp_path, "rb") as f:
            if not validate_magic_bytes(f.read(8), ".pdf"):
                raise HTTPException(400, "This file is not a PDF")
        try:
            page_count = await asyncio.to_thread(count_pdf_pages, temp_path)
        except Exception:
            raise HTTPException(
                400, "This PDF could not be read; it may be damaged or password-protected"
            )
        if page_count > MAX_PAGES:
            raise HTTPException(
                422,
                f"This PDF has {page_count} pages; the limit is {MAX_PAGES}. "
                "Split it into smaller files and upload them separately.",
            )

        # Conditional, so a repeated or concurrent /complete starts one job only.
        claimed = await get_pool().execute(
            """
            UPDATE documents SET status = 'queued', stage = 'queued', page_count = $3
            WHERE doc_id = $1 AND user_id = $2 AND status = 'uploading'
            """,
            doc_id,
            user_id,
            page_count,
        )
        if claimed != "UPDATE 1":
            raise HTTPException(409, "This upload has already been completed")
    except HTTPException as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if exc.status_code != 409:
            await _discard_upload(doc_id, user_id)
        raise
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    # The job owns temp_path from here and removes it when done.
    background_tasks.add_task(_process_document, doc_id, user_id, temp_path, row["doc_name"])
    return {"doc_id": doc_id, "filename": row["doc_name"], "page_count": page_count}


@router.get("/status/{doc_id}")
# Polled every few seconds per uploading document; a cheap, owner-scoped read
# that must not eat the general per-minute budget the rest of the app shares.
@limiter.exempt
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
    if row["status"] in ("complete", "partial"):
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
        "classification": classification,
        "page_count": row["page_count"],
        "failed_pages": list(row.get("failed_pages") or []),
        # Hidden (white, invisible or tiny) text was found and left out of the index.
        "suspicious": bool(row.get("suspicious")),
        "hidden_text_pages": list(row.get("hidden_text_pages") or []),
        "hidden_text_removed": row.get("hidden_text_removed") or 0,
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
