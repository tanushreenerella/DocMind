"""Document parser with pdfplumber extraction and Gemini Vision OCR fallback."""
import asyncio
from io import BytesIO
import os
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types

from core.config import GEMINI_API_KEY, GEMINI_OCR_MODEL, IMAGES_PATH

OCR_TEXT_THRESHOLD = 20
PAGE_IMAGE_DPI = 175
_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


async def _ocr_page_with_gemini(pil_image) -> str:
    """Transcribe one rendered PDF page with Gemini Vision."""
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="JPEG", quality=85)
    response = await asyncio.to_thread(
        _get_gemini_client().models.generate_content,
        model=GEMINI_OCR_MODEL,
        contents=[
            "Transcribe all readable text on this document page. Return only the transcription.",
            types.Part.from_bytes(data=image_bytes.getvalue(), mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(temperature=0),
    )
    return (response.text or "").strip()


async def parse_document(file_path: str, doc_id: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await _parse_pdf(file_path, doc_id)
    return []


async def _parse_pdf(file_path: str, doc_id: str) -> list[dict]:
    os.makedirs(IMAGES_PATH, exist_ok=True)

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, plumber_page in enumerate(pdf.pages, start=1):
            text = plumber_page.extract_text() or ""

            pil_image = None

            # OCR only pages where pdfplumber found almost no text.
            if len(text.strip()) < OCR_TEXT_THRESHOLD:
                try:
                    pil_image = plumber_page.to_image(resolution=PAGE_IMAGE_DPI).original
                    ocr_text = await _ocr_page_with_gemini(pil_image)
                    if ocr_text.strip():
                        print(f"[parser] page {page_num}: Gemini OCR got {len(ocr_text)} chars")
                        text = ocr_text
                    else:
                        print(f"[parser] page {page_num}: Gemini OCR returned empty")
                except Exception as e:
                    print(f"[parser] page {page_num}: Gemini OCR error — {e}")

            # Table extraction
            raw_tables = plumber_page.extract_tables() or []
            structured_tables = []
            for table in raw_tables:
                if table and table[0]:
                    headers = table[0]
                    rows = table[1:]
                    structured_tables.append({"headers": headers, "rows": rows})
                    text = (text or "") + "\n" + _table_to_text(headers, rows)

            # Save page thumbnail
            img_filename = ""
            if pil_image is None:
                try:
                    pil_image = plumber_page.to_image(resolution=PAGE_IMAGE_DPI).original
                except Exception as e:
                    print(f"[parser] page {page_num}: thumbnail render error — {e}")
            if pil_image is not None:
                img_filename = f"{doc_id}_p{page_num}.jpg"
                img_path = os.path.join(IMAGES_PATH, img_filename)
                try:
                    pil_image.save(img_path, "JPEG", quality=75)
                except Exception as e:
                    print(f"[parser] could not save image for page {page_num}: {e}")
                    img_filename = ""

            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "image_path": os.path.join(IMAGES_PATH, img_filename) if img_filename else "",
                "image_filename": img_filename,
                "has_tables": bool(raw_tables),
                "tables": structured_tables,
            })

    extracted = sum(1 for p in pages if p["text"])
    print(f"[parser] done — {len(pages)} pages, {extracted} with text")
    return pages


def _table_to_text(headers: list, rows: list) -> str:
    lines = [" | ".join(str(h) for h in headers if h)]
    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c))
    return "\n".join(lines)
