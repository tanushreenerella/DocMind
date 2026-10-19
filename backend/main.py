import os
from contextlib import asynccontextmanager

# Compatibility shim: adapt older positional `posthog.capture(user, event, props)`
# calls (used by Chromadb's telemetry) to the newer `posthog.capture(event, distinct_id=..., properties=...)` API.
try:
    import posthog

    _posthog_capture = getattr(posthog, "capture", None)

    def _capture_compat(*args, **kwargs):
        if _posthog_capture is None:
            return None
        # Older code called: posthog.capture(user_id, event_name, properties)
        if len(args) == 3 and isinstance(args[2], dict):
            user_id, event_name, properties = args
            return _posthog_capture(event_name, distinct_id=user_id, properties=properties, **kwargs)
        return _posthog_capture(*args, **kwargs)

    posthog.capture = _capture_compat
except Exception:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core.config import (
    ALLOWED_ORIGINS,
    CHROMA_PATH,
    IMAGES_PATH,
    STORAGE_PATH,
)
from core.auth import require_allowed_origin
from core.database import connect, disconnect
from core.rate_limit import limiter, rate_limit_exceeded_handler
from routers import chat, documents, upload
from routers import auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directories exist
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(os.path.join(STORAGE_PATH, "temp"), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_PATH, "pdfs"), exist_ok=True)
    # Open DB pool
    await connect()
    yield
    # Close DB pool on shutdown
    await disconnect()


app = FastAPI(
    title="DocMind — Document Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Applies the default limit to routes without their own @limiter.limit.
app.add_middleware(SlowAPIMiddleware)
# Added last so it runs first: rejected origins never spend rate-limit budget.
app.middleware("http")(require_allowed_origin)

app.include_router(auth.router)
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")


@app.get("/health")
@limiter.exempt
async def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    print(f"[ERROR] {request.method} {request.url.path}: {exc}")
    origin = request.headers.get("origin", "")
    response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
@app.get("/api/debug-ip")
async def debug_ip(request: Request):
    return {
        "xff": request.headers.get("x-forwarded-for"),
        "real_ip": request.headers.get("x-real-ip"),
        "client": request.client.host if request.client else None,
    }