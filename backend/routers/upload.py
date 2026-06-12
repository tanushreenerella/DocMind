import os
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.config import settings
from core.security import sanitize_filename, validate_file
from services.classifier import classify_document
from services.embedder import index_document
from services.parser import parse_document

router = APIRouter()

# In-memory job store — swap for Redis in production
job_status: dict[str, dict] = {}


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    job_ids = []

    for file in files:
        if not await validate_file(file):
            raise HTTPException(400, f"Invalid file: {file.filename}")

        doc_id = str(uuid.uuid4())
        safe_name = sanitize_filename(file.filename or "upload")

        temp_dir = os.path.join(settings.STORAGE_PATH, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{doc_id}_{safe_name}")

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        job_status[doc_id] = {
            "status": "queued",
            "filename": safe_name,
            "stage": "queued",
            "progress": 0,
            "error": None,
            "classification": None,
            "page_count": 0,
        }

        background_tasks.add_task(_process_document, doc_id, temp_path, safe_name)
        job_ids.append({"doc_id": doc_id, "filename": safe_name})

    return {"jobs": job_ids}


async def _process_document(doc_id: str, file_path: str, filename: str) -> None:
    try:
        job_status[doc_id]["stage"] = "parsing"
        job_status[doc_id]["status"] = "processing"
        job_status[doc_id]["progress"] = 10

        pages = await parse_document(file_path, doc_id)
        job_status[doc_id]["page_count"] = len(pages)
        job_status[doc_id]["progress"] = 50

        job_status[doc_id]["stage"] = "classifying"
        combined_text = " ".join(p["text"] for p in pages)
        classification = await classify_document(combined_text, filename)
        job_status[doc_id]["classification"] = classification
        job_status[doc_id]["progress"] = 75

        job_status[doc_id]["stage"] = "indexing"
        await index_document(doc_id, filename, pages, classification)

        job_status[doc_id]["stage"] = "complete"
        job_status[doc_id]["status"] = "complete"
        job_status[doc_id]["progress"] = 100

    except Exception as exc:
        job_status[doc_id]["status"] = "error"
        job_status[doc_id]["stage"] = "error"
        # Return generic message; log full error server-side only
        job_status[doc_id]["error"] = "Processing failed. Please try again."
        print(f"[ERROR] doc_id={doc_id}: {exc}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.get("/status/{doc_id}")
async def get_status(doc_id: str) -> dict:
    if doc_id not in job_status:
        raise HTTPException(404, "Job not found")
    return job_status[doc_id]


@router.get("/page-image/{image_filename}")
async def get_page_image(image_filename: str):
    """Serve page images through a validated endpoint — never from a public folder."""
    # Allow only safe chars: alphanumeric, dash, underscore, dot
    if not all(c.isalnum() or c in "-_." for c in image_filename):
        raise HTTPException(400, "Invalid filename")

    img_path = os.path.join(settings.IMAGES_PATH, image_filename)

    # Resolve to absolute path and confirm it's inside IMAGES_PATH
    abs_img = os.path.realpath(img_path)
    abs_dir = os.path.realpath(settings.IMAGES_PATH)
    if not abs_img.startswith(abs_dir):
        raise HTTPException(400, "Invalid path")

    if not os.path.exists(abs_img):
        raise HTTPException(404, "Image not found")

    return FileResponse(abs_img, media_type="image/jpeg")
