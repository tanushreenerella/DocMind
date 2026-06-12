import re
import os
from pathlib import Path
from fastapi import UploadFile
from backend.core.config import settings

# Magic bytes for supported file types
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
}

ALLOWED_MIME_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".tiff": {"image/tiff"},
}


def sanitize_filename(filename: str) -> str:
    """Strip path traversal and sanitize to safe characters only."""
    # Get just the basename — no directory components
    filename = os.path.basename(filename)
    # Remove null bytes
    filename = filename.replace("\x00", "")
    # Keep only alphanumeric, dash, underscore, dot
    filename = re.sub(r"[^\w\-.]", "_", filename)
    # Collapse multiple underscores/dots
    filename = re.sub(r"\.{2,}", ".", filename)
    # Enforce max length keeping extension
    if len(filename) > 100:
        stem, ext = os.path.splitext(filename)
        filename = stem[: 100 - len(ext)] + ext
    return filename or "unnamed_file"


def validate_magic_bytes(content: bytes, extension: str) -> bool:
    """Check actual file header bytes match the declared extension."""
    ext = extension.lower()
    expected = MAGIC_BYTES.get(ext)
    if not expected:
        return False
    return any(content.startswith(magic) for magic in expected)


async def validate_file(file: UploadFile) -> bool:
    """
    Validate extension, content-type, file size, and magic bytes.
    Rewinds the file stream before returning.
    """
    if not file.filename:
        return False

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False

    # Content-type check (browsers send this; don't rely solely on it)
    if file.content_type:
        allowed_types = ALLOWED_MIME_TYPES.get(ext, set())
        if allowed_types and file.content_type not in allowed_types:
            # Tolerate application/octet-stream fallback
            if file.content_type != "application/octet-stream":
                return False

    # Read header bytes for magic check + measure total size
    content = await file.read()

    # Size check
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        await file.seek(0)
        return False

    # Magic bytes check
    if not validate_magic_bytes(content, ext):
        await file.seek(0)
        return False

    # Rewind so downstream code can read the full content
    await file.seek(0)
    return True
