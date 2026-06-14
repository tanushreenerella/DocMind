"""
Document parser: PDF text extraction via pdfplumber.
Image rendering and OCR removed to stay within free-tier RAM limits.
"""
from pathlib import Path

import pdfplumber

async def parse_document(file_path: str, _doc_id: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await _parse_pdf(file_path)
    return []


async def _parse_pdf(file_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, plumber_page in enumerate(pdf.pages):
            page_num = i + 1
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

            pages.append(
                {
                    "page_number": page_num,
                    "text": text.strip(),
                    "image_path": "",
                    "image_filename": "",
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
