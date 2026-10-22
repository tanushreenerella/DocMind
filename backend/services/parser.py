"""Document parser with pdfplumber extraction and Gemini Vision OCR fallback."""
import asyncio
from collections import deque
from io import BytesIO
import os
from pathlib import Path
import re
import time

import pdfplumber
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from core.config import GEMINI_API_KEY, GEMINI_OCR_MODEL, OCR_MAX_PER_MINUTE

OCR_TEXT_THRESHOLD = 20
PAGE_IMAGE_DPI = 175
OCR_MAX_ATTEMPTS = 4
_OCR_RETRYABLE_CODES = {429, 503}
_OCR_BACKOFF_BASE_SECONDS = 2.0
_OCR_MAX_RETRY_DELAY_SECONDS = 60.0
_gemini_client: genai.Client | None = None
# Indirection so tests can skip real waiting.
_sleep = asyncio.sleep


class SlidingWindowLimiter:
    """Allow at most ``max_calls`` per ``period`` seconds; callers wait their turn in order."""

    def __init__(self, max_calls: int, period: float = 60.0, clock=time.monotonic, sleep=None) -> None:
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        # asyncio.Lock binds to one event loop; tests run several in sequence.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def acquire(self) -> None:
        sleep = self._sleep or _sleep
        async with self._get_lock():
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                await sleep(self.period - (now - self._calls[0]))


# Per process: with several workers the effective cap is workers x this.
_ocr_limiter = SlidingWindowLimiter(OCR_MAX_PER_MINUTE)


def _server_retry_delay(exc: genai_errors.APIError) -> float | None:
    """Seconds from a google.rpc.RetryInfo ``retryDelay`` (e.g. "17s"), if the error has one."""
    details = exc.details
    payload = details.get("error", details) if isinstance(details, dict) else None
    entries = payload.get("details") if isinstance(payload, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and "retryDelay" in entry:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(entry["retryDelay"]).strip())
            if match:
                return float(match.group(1))
    return None


def _ocr_retry_delay(exc: genai_errors.APIError, attempt: int) -> float:
    delay = _server_retry_delay(exc)
    if delay is None:
        delay = _OCR_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    return min(delay, _OCR_MAX_RETRY_DELAY_SECONDS)


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


async def _ocr_page_with_gemini(pil_image) -> str:
    """Transcribe one rendered PDF page with Gemini Vision.

    Throttled to OCR_MAX_PER_MINUTE; 429/503 are retried up to OCR_MAX_ATTEMPTS,
    waiting the server's retryDelay when given, else exponential backoff.
    """
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="JPEG", quality=85)
    contents = [
        "Transcribe all readable text on this document page. Return only the transcription.",
        types.Part.from_bytes(data=image_bytes.getvalue(), mime_type="image/jpeg"),
    ]
    for attempt in range(1, OCR_MAX_ATTEMPTS + 1):
        await _ocr_limiter.acquire()
        try:
            response = await asyncio.to_thread(
                _get_gemini_client().models.generate_content,
                model=GEMINI_OCR_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0),
            )
            return (response.text or "").strip()
        except genai_errors.APIError as exc:
            if exc.code not in _OCR_RETRYABLE_CODES or attempt == OCR_MAX_ATTEMPTS:
                raise
            delay = _ocr_retry_delay(exc, attempt)
            print(f"[parser] Gemini OCR {exc.code} (attempt {attempt}/{OCR_MAX_ATTEMPTS}); retrying in {delay:.1f}s")
            await _sleep(delay)
    raise AssertionError("unreachable")


def count_pdf_pages(file_path: str) -> int:
    """Page count from the PDF's page tree, without extracting any content.

    Raises if the file is not a PDF pdfplumber can open (corrupt, encrypted...).
    """
    with pdfplumber.open(file_path) as pdf:
        return len(pdf.pages)


async def parse_document(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
    """Extract page text; render page images into ``image_dir`` for the caller to store."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await _parse_pdf(file_path, doc_id, image_dir)
    return []


async def _parse_pdf(file_path: str, doc_id: str, image_dir: str) -> list[dict]:
    os.makedirs(image_dir, exist_ok=True)

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, plumber_page in enumerate(pdf.pages, start=1):
            text = plumber_page.extract_text() or ""

            pil_image = None
            ocr_failed = False

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
                    ocr_failed = True
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
                img_path = os.path.join(image_dir, img_filename)
                try:
                    pil_image.save(img_path, "JPEG", quality=75)
                except Exception as e:
                    print(f"[parser] could not save image for page {page_num}: {e}")
                    img_filename = ""

            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "image_path": os.path.join(image_dir, img_filename) if img_filename else "",
                "image_filename": img_filename,
                "has_tables": bool(raw_tables),
                "tables": structured_tables,
                # Needed OCR and every attempt failed: this page's text is missing.
                "ocr_failed": ocr_failed,
            })

    extracted = sum(1 for p in pages if p["text"])
    print(f"[parser] done — {len(pages)} pages, {extracted} with text")
    return pages


def _table_to_text(headers: list, rows: list) -> str:
    lines = [" | ".join(str(h) for h in headers if h)]
    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c))
    return "\n".join(lines)
