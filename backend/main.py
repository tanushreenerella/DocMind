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
    STORAGE_PATH,
)
from core.auth import require_allowed_origin, require_proxy_secret
from core.database import connect, disconnect, ensure_schema
from core.rate_limit import CLIENT_IP_HEADER, limiter, rate_limit_exceeded_handler
from routers import chat, documents, tokens, upload
from routers import auth
from services import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directories exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(os.path.join(STORAGE_PATH, "temp"), exist_ok=True)
    # Pick R2 or local disk now, so a half-configured R2 fails at boot.
    storage.get_storage()
    # Open DB pool
    await connect()
    await ensure_schema()
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
# Runs before the rate limiter: rejected origins never spend rate-limit budget.
app.middleware("http")(require_allowed_origin)
# Outermost: anything not sent by the Next.js proxy is rejected before any other work.
app.middleware("http")(require_proxy_secret)

app.include_router(auth.router)
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(tokens.router, prefix="/api")


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
        "x_client_ip": request.headers.get(CLIENT_IP_HEADER),
        "xff": request.headers.get("x-forwarded-for"),
        "real_ip": request.headers.get("x-real-ip"),
        "client": request.client.host if request.client else None,
    }
