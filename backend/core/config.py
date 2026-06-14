import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
STORAGE_PATH = os.environ.get("STORAGE_PATH", "./storage")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./storage/chroma_db")
IMAGES_PATH = os.environ.get("IMAGES_PATH", "./storage/page_images")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))

_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if _origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",")]
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "https://doc-mind-three.vercel.app"]