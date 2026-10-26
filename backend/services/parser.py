"""Document parser with pdfplumber extraction and Gemini Vision OCR fallback."""
import asyncio
from collections import Counter, deque
from io import BytesIO
import math
import os
from pathlib import Path
import re
import time

import pdfplumber
from pdfminer.layout import LTChar, LTCurve, LTImage
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfplumber.page import PDFPageAggregatorWithMarkedContent
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


# ── text cleanup ────────────────────────────────────────────────────────────

# Invisible or layout-only characters PDFs often carry.
_INVISIBLE = str.maketrans({"­": "", "​": "", "﻿": "", " ": " "})
# "inter-\nnational" -> "international": a letter, hyphen, line break, lowercase letter.
_WORD_BROKEN_AT_HYPHEN = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-z])")
# "IL-\n6", "COVID-\n19": a real hyphen split across lines; keep it, drop the break.
_TERM_BROKEN_AT_HYPHEN = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# Header/footer candidates: at most this many non-blank lines at the top and bottom of a page.
_EDGE_LINES = 3


def clean_page_text(text: str) -> str:
    """Rejoin words hyphenated across lines and normalise whitespace.

    Spaces and tabs collapse to one space and lines are trimmed; single line
    breaks stay (they separate table rows and list items) and runs of blank
    lines become one blank line, which still marks a paragraph.
    """
    text = text.translate(_INVISIBLE).replace("\r\n", "\n").replace("\r", "\n")
    text = _WORD_BROKEN_AT_HYPHEN.sub("", text)
    text = _TERM_BROKEN_AT_HYPHEN.sub("-", text)
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def _line_key(line: str) -> str:
    # Digits masked, so "Page 3 of 10" and "Page 4 of 10" are the same footer.
    return re.sub(r"\d+", "#", line.casefold()).strip()


def _edge_line_indexes(lines: list[str]) -> set[int]:
    """Header/footer candidates: up to _EDGE_LINES lines at each end, but at most a
    quarter of the page, so a short page's body is never treated as its edges."""
    non_blank = [i for i, line in enumerate(lines) if line.strip()]
    edge = max(1, min(_EDGE_LINES, len(non_blank) // 4))
    return set(non_blank[:edge] + non_blank[-edge:])


def strip_repeated_lines(page_texts: list[str]) -> list[str]:
    """Remove running headers and footers: lines at the top or bottom of a page
    that (ignoring digits) recur at the top or bottom of most pages with text.

    Needs at least three pages with text; with fewer, repetition says nothing.
    """
    with_text = [text for text in page_texts if text.strip()]
    if len(with_text) < 3:
        return page_texts

    counts: Counter[str] = Counter()
    for text in with_text:
        lines = text.split("\n")
        counts.update({_line_key(lines[i]) for i in _edge_line_indexes(lines)} - {""})
    repeated = {key for key, pages in counts.items() if pages > len(with_text) / 2}
    if not repeated:
        return page_texts

    cleaned = []
    for text in page_texts:
        lines = text.split("\n")
        edges = _edge_line_indexes(lines)
        kept = [line for i, line in enumerate(lines) if not (i in edges and _line_key(line) in repeated)]
        cleaned.append(_EXTRA_BLANK_LINES.sub("\n\n", "\n".join(kept)).strip())
    return cleaned


# ── hidden text ─────────────────────────────────────────────────────────────

# Text rendering modes that paint nothing: 3 (invisible) and 7 (clip only).
_INVISIBLE_RENDER_MODES = {3, 7}
# Glyphs shorter than this (in points, after scaling) can't be read.
_MIN_VISIBLE_FONT_SIZE = 2.0
# Every colour component at least this light (gray/RGB) counts as white.
_NEAR_WHITE = 0.95
# Colour spaces whose components are not light levels (a Separation tint of 1 is full ink).
_NON_LIGHTNESS_SPACES = {"Separation", "DeviceN", "Indexed", "Pattern"}


def _is_near_white(color, colorspace_name: str | None = None) -> bool:
    if colorspace_name in _NON_LIGHTNESS_SPACES:
        return False
    if isinstance(color, (int, float)):
        color = (color,)
    if not isinstance(color, (tuple, list)) or not all(isinstance(c, (int, float)) for c in color):
        return False
    if len(color) in (1, 3):  # gray, RGB
        return min(color) >= _NEAR_WHITE
    if len(color) == 4:  # CMYK: no ink
        return max(color) <= 1 - _NEAR_WHITE
    return False


class _VisibleTextAggregator(PDFPageAggregatorWithMarkedContent):
    """pdfplumber's layout device, minus characters a reader can't see.

    Drops glyphs drawn in an invisible rendering mode, smaller than
    _MIN_VISIBLE_FONT_SIZE, or filled near-white where nothing non-white was
    painted under them first (so white text on a dark box or image stays).
    Dropped glyphs still advance the pen, so the visible text keeps its layout.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_chars = 0
        self._render_mode = 0
        # Bounding boxes of images and non-white fills painted so far, in draw order.
        self._backgrounds: list[tuple[float, float, float, float]] = []

    def render_string(self, textstate, seq, ncs, graphicstate) -> None:
        self._render_mode = textstate.render
        super().render_string(textstate, seq, ncs, graphicstate)

    def paint_path(self, gstate, stroke, fill, evenodd, path) -> None:
        start = len(self.cur_item._objs)
        super().paint_path(gstate, stroke, fill, evenodd, path)
        for obj in self.cur_item._objs[start:]:
            if isinstance(obj, LTCurve) and obj.fill and not _is_near_white(obj.non_stroking_color):
                self._backgrounds.append(obj.bbox)

    def render_image(self, name, stream) -> None:
        super().render_image(name, stream)
        if self.cur_item._objs and isinstance(self.cur_item._objs[-1], LTImage):
            self._backgrounds.append(self.cur_item._objs[-1].bbox)

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate) -> float:
        adv = super().render_char(matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate)
        char = self.cur_item._objs[-1]
        if isinstance(char, LTChar) and self._is_hidden(char, matrix, fontsize, ncs, graphicstate):
            self.cur_item._objs.pop()
            if not char.get_text().isspace():
                self.hidden_chars += 1
        return adv

    def _is_hidden(self, char: LTChar, matrix, fontsize, ncs, graphicstate) -> bool:
        if self._render_mode in _INVISIBLE_RENDER_MODES:
            return True
        # Glyph height on the page: the font size scaled by the text's y axis.
        if abs(fontsize) * math.hypot(matrix[2], matrix[3]) < _MIN_VISIBLE_FONT_SIZE:
            return True
        if _is_near_white(graphicstate.ncolor, getattr(ncs, "name", None)):
            x = (char.x0 + char.x1) / 2
            y = (char.y0 + char.y1) / 2
            return not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self._backgrounds)
        return False


def _strip_hidden_text(plumber_page) -> int:
    """Lay ``plumber_page`` out without hidden characters; return how many were dropped.

    Must run before anything reads the page's chars, text or tables. pdfplumber
    (pinned 0.11.0) builds all of those from ``Page._layout`` and reuses it once
    set, so filling it here filters every later extraction. Page images are
    rendered separately and still show the page as it is.
    """
    pdf = plumber_page.pdf
    device = _VisibleTextAggregator(pdf.rsrcmgr, pageno=plumber_page.page_number, laparams=pdf.laparams)
    PDFPageInterpreter(pdf.rsrcmgr, device).process_page(plumber_page.page_obj)
    plumber_page._layout = device.get_result()
    return device.hidden_chars


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
            # Text a reader can't see is only there for the model: injection.
            hidden_removed = _strip_hidden_text(plumber_page)
            if hidden_removed:
                print(f"[parser] page {page_num}: removed {hidden_removed} hidden characters")
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
            table_texts = []
            for table in raw_tables:
                if table and table[0]:
                    headers = table[0]
                    rows = table[1:]
                    structured_tables.append({"headers": headers, "rows": rows})
                    table_texts.append(_table_to_text(headers, rows))

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
                "text": clean_page_text(text),  # body only until headers/footers go, below
                "image_path": os.path.join(image_dir, img_filename) if img_filename else "",
                "image_filename": img_filename,
                "has_tables": bool(raw_tables),
                "tables": structured_tables,
                # Needed OCR and every attempt failed: this page's text is missing.
                "ocr_failed": ocr_failed,
                # Non-space characters dropped as invisible to a reader.
                "hidden_text_removed": hidden_removed,
                "_table_texts": table_texts,
            })

    # Running headers/footers only show up across pages. Tables are added after,
    # so a table row repeated on every page is never mistaken for one.
    bodies = strip_repeated_lines([p["text"] for p in pages])
    for page, body in zip(pages, bodies):
        page["text"] = "\n\n".join(part for part in (body, *page.pop("_table_texts")) if part).strip()

    extracted = sum(1 for p in pages if p["text"])
    print(f"[parser] done — {len(pages)} pages, {extracted} with text")
    return pages


def _table_to_text(headers: list, rows: list) -> str:
    lines = [" | ".join(str(h) for h in headers if h)]
    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c))
    return "\n".join(lines)
