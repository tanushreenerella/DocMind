"""
Document parser: PDF text extraction via pdfplumber.
pytesseract OCR runs as fallback for scanned/handwritten pages with minimal text.
Requires: Tesseract-OCR + Poppler installed on system (set paths in .env).
"""
import os
from pathlib import Path

import pdfplumber

IMAGES_PATH = os.getenv("IMAGES_PATH", "./storage/page_images")
POPPLER_PATH = os.getenv("POPPLER_PATH", "")       # e.g. C:\poppler\bin
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")     # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_TEXT_THRESHOLD = 20
PAGE_IMAGE_DPI = 100


def _try_import_ocr():
    """Returns (convert_from_path, pytesseract) or (None, None) if unavailable."""
    try:
        from pdf2image import convert_from_path
        import pytesseract as _tess

        if TESSERACT_CMD:
            _tess.pytesseract.tesseract_cmd = TESSERACT_CMD

        _tess.get_tesseract_version()   # raises if binary not found
        print("[parser] OCR ready (Tesseract + pdf2image available)")
        return convert_from_path, _tess
    except Exception as e:
        print(f"[parser] OCR unavailable: {e}")
        print("[parser] Tip: set TESSERACT_CMD and POPPLER_PATH in .env")
        return None, None


_convert_from_path, _pytesseract = _try_import_ocr()


async def parse_document(file_path: str, doc_id: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await _parse_pdf(file_path, doc_id)
    return []


async def _parse_pdf(file_path: str, doc_id: str) -> list[dict]:
    os.makedirs(IMAGES_PATH, exist_ok=True)

    # Convert all pages to images (needed for OCR + thumbnails)
    page_images: list = []
    if _convert_from_path is not None:
        try:
            with pdfplumber.open(file_path) as _tmp:
                total_pages = len(_tmp.pages)
            dpi = PAGE_IMAGE_DPI if total_pages <= 30 else 72

            kwargs: dict = {"dpi": dpi, "fmt": "jpeg"}
            if POPPLER_PATH:
                kwargs["poppler_path"] = POPPLER_PATH

            print(f"[parser] converting {total_pages} pages at {dpi} DPI")
            page_images = _convert_from_path(file_path, **kwargs)
            print(f"[parser] got {len(page_images)} page images")
        except Exception as e:
            print(f"[parser] image conversion failed: {e}")
            print("[parser] Tip: check POPPLER_PATH is set correctly in .env")

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, plumber_page in enumerate(pdf.pages):
            page_num = i + 1
            text = plumber_page.extract_text() or ""

            pil_image = page_images[i] if i < len(page_images) else None

            # OCR for pages where pdfplumber found almost no text
            if len(text.strip()) < OCR_TEXT_THRESHOLD and pil_image is not None and _pytesseract is not None:
                try:
                    small = pil_image.resize(
                        (pil_image.width // 2, pil_image.height // 2)
                    )
                    ocr_text = _pytesseract.image_to_string(small)
                    if ocr_text.strip():
                        print(f"[parser] page {page_num}: OCR got {len(ocr_text)} chars")
                        text = ocr_text
                    else:
                        print(f"[parser] page {page_num}: OCR returned empty — image may be blank")
                except Exception as e:
                    print(f"[parser] page {page_num}: OCR error — {e}")

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
