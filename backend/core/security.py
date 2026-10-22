import re
import os

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
