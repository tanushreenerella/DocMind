"""
Document parser: PDF text extraction via pdfplumber + pdf2image page rendering.
pytesseract OCR runs as fallback for scanned pages with minimal text.
"""
from pathlib import Path
import pytesseract
import pdfplumber
import os
from pdf2image import convert_from_path
IMAGES_PATH = os.getenv("IMAGES_PATH", "./storage/page_images")
async def parse_document(file_path: str, doc_id: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await _parse_pdf(file_path, doc_id)
    return []

async def _parse_pdf(file_path: str, doc_id: str) -> list[dict]:
    os.makedirs(IMAGES_PATH, exist_ok=True)
    page_images = convert_from_path(file_path, dpi=100, fmt="jpeg")
    
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, plumber_page in enumerate(pdf.pages):
            page_num = i + 1
            text = plumber_page.extract_text() or ""
            
            # OCR only if pdfplumber got less than 20 chars (scanned page)
            if len(text.strip()) < 20 and i < len(page_images):
                try:
                    text = pytesseract.image_to_string(
                        page_images[i].resize(
                            (page_images[i].width // 2, page_images[i].height // 2)
                        )
                    )
                except Exception:
                    text = ""

            # Table extraction (runs for ALL pages regardless of OCR)
            raw_tables = plumber_page.extract_tables() or []
            has_tables = bool(raw_tables)
            structured_tables = []
            for table in raw_tables:
                if table and table[0]:
                    headers = table[0]
                    rows = table[1:]
                    structured_tables.append({"headers": headers, "rows": rows})
                    text = (text or "") + "\n" + _table_to_text(headers, rows)

            # Save page image
            img_filename = f"{doc_id}_p{page_num}.jpg"
            img_path = os.path.join(IMAGES_PATH, img_filename)
            if i < len(page_images):
                page_images[i].save(img_path, "JPEG", quality=75)

            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "image_path": img_path,
                "image_filename": img_filename,
                "has_tables": has_tables,
                "tables": structured_tables,
            })
    return pages

def _table_to_text(headers: list, rows: list) -> str:
    lines = [" | ".join(str(h) for h in headers if h)]
    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c))
    return "\n".join(lines)
