"""
Document parser: handles PDFs (text, scanned, mixed) and images.
Returns per-page extracted text + rendered page image path.
"""
import os
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

from core.config import IMAGES_PATH


async def parse_document(file_path: str, doc_id: str) -> list[dict]:
    """
    Returns list of page dicts:
    {
      "page_number": int,
      "text": str,
      "image_path": str,
      "image_filename": str,
      "has_tables": bool,
      "tables": list[dict]
    }
    """
    ext = Path(file_path).suffix.lower()
    os.makedirs(IMAGES_PATH, exist_ok=True)

    if ext == ".pdf":
        return await _parse_pdf(file_path, doc_id)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff"}:
        return await _parse_image(file_path, doc_id)
    return []


async def _parse_pdf(file_path: str, doc_id: str) -> list[dict]:
    pages = []
    # Render all pages as images for display + OCR fallback
    page_images = convert_from_path(file_path, dpi=150)

    with pdfplumber.open(file_path) as pdf:
        for i, (plumber_page, pil_image) in enumerate(zip(pdf.pages, page_images)):
            page_num = i + 1

            img_filename = f"{doc_id}_page_{page_num}.jpg"
            img_path = os.path.join(IMAGES_PATH, img_filename)
            pil_image.save(img_path, "JPEG", quality=85)

            text = plumber_page.extract_text() or ""

            raw_tables = plumber_page.extract_tables() or []
            has_tables = bool(raw_tables)
            structured_tables = []
            for table in raw_tables:
                if table and table[0]:
                    headers = table[0]
                    rows = table[1:]
                    structured_tables.append({"headers": headers, "rows": rows})
                    text = (text or "") + "\n" + _table_to_text(headers, rows)

            # Scanned page — fall back to OCR
            if len(text.strip()) < 50:
                text = pytesseract.image_to_string(pil_image)

            pages.append(
                {
                    "page_number": page_num,
                    "text": text.strip(),
                    "image_path": img_path,
                    "image_filename": img_filename,
                    "has_tables": has_tables,
                    "tables": structured_tables,
                }
            )

    return pages


def _table_to_text(headers: list, rows: list) -> str:
    lines = [" | ".join(str(h) for h in headers if h)]
    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c))
    return "\n".join(lines)


async def _parse_image(file_path: str, doc_id: str) -> list[dict]:
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)

    img_filename = f"{doc_id}_page_1.jpg"
    img_path = os.path.join(IMAGES_PATH, img_filename)
    img.save(img_path, "JPEG", quality=85)

    return [
        {
            "page_number": 1,
            "text": text.strip(),
            "image_path": img_path,
            "image_filename": img_filename,
            "has_tables": False,
            "tables": [],
        }
    ]
