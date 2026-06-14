import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.config import IMAGES_PATH, CHROMA_PATH, STORAGE_PATH
from routers import chat, documents, upload

# Ensure storage directories exist at startup
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(os.path.join(STORAGE_PATH, "temp"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_PATH, "pdfs"), exist_ok=True)

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(
    title="DocMind — Document Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    print(f"[ERROR] {request.method} {request.url.path}: {exc}")
    response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
    # CORSMiddleware doesn't wrap exception handler responses in all FastAPI versions;
    # add the header manually so browsers can read the 500 body.
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
