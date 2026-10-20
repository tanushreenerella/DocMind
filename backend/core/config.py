import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_OCR_MODEL = os.environ.get("GEMINI_OCR_MODEL", "gemini-3.6-flash")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
STORAGE_PATH = os.environ.get("STORAGE_PATH", "./storage")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./storage/chroma_db")
# PDFs and page images go to Cloudflare R2 when all four R2_* vars are set
# (see services/storage.py), else to local disk under LOCAL_OBJECTS_PATH.
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "")
LOCAL_OBJECTS_PATH = os.environ.get("LOCAL_OBJECTS_PATH", "./storage/objects")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
# Per-route limits use the `limits` syntax, e.g. "5/minute", "10/hour".
RATE_LIMIT_AUTH = os.environ.get("RATE_LIMIT_AUTH", "5/minute")
# Login attempts per (lowercased) email, whatever IP or user they come from.
RATE_LIMIT_LOGIN_EMAIL = os.environ.get("RATE_LIMIT_LOGIN_EMAIL", "10/15 minutes")
RATE_LIMIT_CHAT = os.environ.get("RATE_LIMIT_CHAT", "20/minute")
RATE_LIMIT_AGENTIC_CHAT = os.environ.get("RATE_LIMIT_AGENTIC_CHAT", "10/minute")
RATE_LIMIT_UPLOAD = os.environ.get("RATE_LIMIT_UPLOAD", "10/hour")
RATE_LIMIT_VOICE = os.environ.get("RATE_LIMIT_VOICE", "10/minute")
# e.g. redis://host:6379/0; counters are per-process with memory://.
RATE_LIMIT_STORAGE_URI = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")
HYBRID_SEARCH_ENABLED = (
    os.environ.get("HYBRID_SEARCH_ENABLED", "false").strip().lower()
    == "true"
)
RERANK_ENABLED = (
    os.environ.get("RERANK_ENABLED", "false").strip().lower()
    == "true"
)
# Must be a FlashRank model name (see flashrank.Config.model_file_map).
RERANK_MODEL = os.environ.get("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
# How many fused candidates the cross-encoder scores per query.
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "10"))

# Chroma Cloud
_missing_chroma_vars = [
    name
    for name in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
    if not os.environ.get(name)
]
if _missing_chroma_vars:
    raise RuntimeError(
        "Missing required Chroma Cloud environment variable(s): "
        + ", ".join(_missing_chroma_vars)
    )
CHROMA_API_KEY = os.environ["CHROMA_API_KEY"]
CHROMA_TENANT = os.environ["CHROMA_TENANT"]
CHROMA_DATABASE = os.environ["CHROMA_DATABASE"]

_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if _origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",")]
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "https://doc-mind-three.vercel.app"]

# PostgreSQL / Auth
DATABASE_URL = os.environ.get("DATABASE_URL", "")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable must be set")
JWT_ALGORITHM = "HS256"
# Shared with the Next.js proxy, which sends it as x-proxy-secret on every
# /api request; anything else reaching /api directly is rejected.
PROXY_SECRET = os.environ.get("PROXY_SECRET")
if not PROXY_SECRET:
    raise RuntimeError("PROXY_SECRET environment variable must be set")
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
# Secure by default; set COOKIE_SECURE=false only for plain-http local dev.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").strip().lower() != "false"
