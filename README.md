# DocMind — Document Intelligence + Agentic RAG

A production-quality web application that ingests messy real-world documents (scanned PDFs, handwritten pages, image-heavy reports, tables), extracts content accurately, classifies each document, and powers a chatbot that answers questions with grounded citations showing the exact source page.

---

## Overview

**What it does**

1. **Upload** — drag-and-drop PDFs or images; background processing pipeline handles parsing, OCR, classification, and vector indexing
2. **Parse** — pdfplumber extracts text and tables; pytesseract OCR handles scanned/handwritten pages automatically
3. **Classify** — Groq LLaMA-3 70B classifies each document (type, topic, sensitivity level, key entities, summary)
4. **Index** — Gemini text-embedding-004 embeds 500-char chunks into ChromaDB with persistent disk storage
5. **Chat** — ask questions in natural language; the RAG pipeline retrieves relevant chunks, synthesizes a grounded answer with inline `[Doc, p.N]` citation markers, and shows clickable page thumbnails

---

## Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│   Next.js Frontend      │        │         FastAPI Backend              │
│  (Vercel)               │        │  (Render)                            │
│                         │        │                                      │
│  /           ChatWindow ├──POST──▶  /api/chat   → RAG service           │
│  /upload     UploadZone ├──POST──▶  /api/uploads/presign, …/complete    │
│                         │        │      → Parser + Embedder             │
│              CitationCard│        │  /api/status/{id}                   │
│              PageViewer  ├──GET───▶  /api/page-image/{file}             │
│              VoiceInput  ├──POST──▶  /api/transcribe → Groq Whisper     │
└─────────────────────────┘        └────────────┬─────────────────────────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          │                     │                     │
                     ChromaDB              Groq API              storage/
                  (persistent,         (LLaMA-3 70B +          page_images/
                   HNSW cosine)         Whisper v3)            (served via
                                                               /api/page-image)
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Tesseract OCR installed on your system
  - **Windows**: [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Ubuntu**: `sudo apt install tesseract-ocr`
- Poppler (for pdf2image)
  - **Windows**: [Poppler Windows](https://github.com/oschwartz10612/poppler-windows/releases)
  - **macOS**: `brew install poppler`
  - **Ubuntu**: `sudo apt install poppler-utils`

### Backend Setup

```bash
cd bfai-assessment/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY

# Run the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Backend Tests

```bash
cd backend
python -m unittest discover -s tests -t . -v
```

Tests are fully offline: `backend/tests/__init__.py` swaps in a fake Chroma
client and dummy `CHROMA_*` credentials before `services.embedder` is imported,
so no real Chroma Cloud credentials are needed (and none can be used) in local
or CI runs.

### Frontend Setup

```bash
cd bfai-assessment/frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# Edit .env.local — set NEXT_PUBLIC_API_URL=http://localhost:8000

# Run dev server
npm run dev
```

Open http://localhost:3000

### Generate Sample Documents

```bash
cd bfai-assessment
pip install reportlab  # if not already installed
python scripts/create_sample_docs.py
# Creates 5 PDFs in sample_docs/
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Groq API key for LLaMA-3 & Whisper |
| `ALLOWED_ORIGINS` | | `["http://localhost:3000"]` | CORS whitelist (JSON array) |
| `MAX_UPLOAD_MB` | | `50` | Max PDF size per file (checked at presign and again on the stored object) |
| `MAX_PAGES` | | `150` | PDFs with more pages are rejected at `/complete` |
| `STORAGE_PATH` | | `./storage` | Base storage directory |
| `CHROMA_API_KEY` | ✅ | — | Chroma Cloud API key (startup fails if missing) |
| `CHROMA_TENANT` | ✅ | — | Chroma Cloud tenant ID |
| `CHROMA_DATABASE` | ✅ | — | Chroma Cloud database. Use a separate one for local dev vs production |
| `IMAGES_PATH` | | `./storage/page_images` | Rendered page images path |
| `RATE_LIMIT_PER_MINUTE` | | `30` | API rate limit per IP |
| `DAILY_TOKEN_BUDGET` | | `50000` | Groq tokens per user per UTC day; chat returns 429 once used up |
| `TOKEN_BUDGET_REDIS_URL` | | `RATE_LIMIT_STORAGE_URI` if Redis | Where daily token counts live; without Redis they go to Postgres |

### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | ✅ | `http://localhost:8000` | Backend API base URL |

---

## MCP server (Claude Desktop and other MCP clients)

DocMind exposes your documents as MCP tools at **`https://<your-frontend>/api/mcp`**: Streamable HTTP, stateless, plain JSON responses. Use the frontend URL (e.g. `https://your-app.vercel.app/api/mcp`, or `http://localhost:3000/api/mcp` locally), not the backend's. The Next.js proxy adds the secret the backend requires, so calling the backend directly gets `403`.

Every request needs a personal access token, which you create under **Settings → Personal access tokens**. Tools only ever see that token's user's documents.

| Tool | Arguments | Returns |
|------|-----------|---------|
| `list_documents` | none | your indexed documents: `doc_id`, name, type, sensitivity, `suspicious` |
| `search_documents` | `query`, `doc_id?`, `k` (1–10, default 5) | matching passages: `doc_name`, `page`, trimmed `excerpt` |
| `ask_documents` | `question`, `doc_id?`, `mode` (`basic` \| `agentic`) | `answer` and `citations` (`doc_name`, `page`, `excerpt`) |

The tools go through the same checks as the chat UI: document ownership, the daily token budget, and outage handling. A refusal comes back as an MCP tool error with a code in brackets, for example `Error executing tool ask_documents: [daily_budget_exceeded] Daily AI usage limit reached, resets at midnight UTC`. The codes are `forbidden`, `daily_budget_exceeded`, `rate_limited`, `invalid_arguments`, `ai_unavailable`, `search_unavailable` and `timeout`.

Limits are per user, whichever token or IP a request comes from: `RATE_LIMIT_PER_MINUTE` requests to `/api/mcp`, plus `ask_documents` capped at `RATE_LIMIT_CHAT` (basic) or `RATE_LIMIT_AGENTIC_CHAT` (agentic). A revoked token is refused on its next request.

### Claude Desktop (via `mcp-remote`)

Claude Desktop launches local MCP servers, so [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) bridges it to the HTTP endpoint. It needs Node.js. In `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "docmind": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://your-app.vercel.app/api/mcp",
        "--header",
        "Authorization:${DOCMIND_AUTH}"
      ],
      "env": {
        "DOCMIND_AUTH": "Bearer dm_your_token_here"
      }
    }
  }
}
```

The header value comes from `env` and there's no space after `Authorization:`. Some clients split arguments on spaces, which would break `"Authorization: Bearer …"`. Restart Claude Desktop, and the three tools appear under the tools menu. For a plain-`http` URL other than `localhost`, add `"--allow-http"` to `args`.

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector
```

Open the URL it prints. Set **Transport Type** to *Streamable HTTP* and **URL** to `https://your-app.vercel.app/api/mcp`. Under **Authentication**, set the `Authorization` header to `Bearer dm_…`. Leave **Connection Type** as *Via Proxy*: the backend doesn't allow CORS from the Inspector's page, so *Direct* fails. Click **Connect**, then **Tools → List Tools**.

Or without any client:

```bash
curl -s https://your-app.vercel.app/api/mcp \
  -H "Authorization: Bearer dm_your_token_here" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_documents","arguments":{}}}'
```

---

## Security Decisions

### What's implemented

| Layer | Control | Implementation |
|-------|---------|----------------|
| Upload | Extension whitelist | `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff` only |
| Upload | Magic bytes validation | Checks actual file header bytes (`%PDF`, `\x89PNG`, `\xff\xd8\xff`, etc.) — not just extension |
| Upload | File size limit | 20 MB max, checked before any processing |
| Upload | Filename sanitization | `os.path.basename` + regex strips path traversal (`../`), limits to alphanumeric + `-_.,` |
| Upload | Content-type check | MIME type validated against per-extension whitelist |
| Storage | Private file storage | Uploaded files and page images stored in `./storage/` — never in a public/static directory |
| Storage | Temp file cleanup | Temp files deleted in a `finally` block after processing completes or fails |
| API | Path traversal guard | `os.path.realpath` comparison ensures served images stay within `IMAGES_PATH` |
| API | Rate limiting | slowapi 30 req/min per IP; exceeds returns HTTP 429 |
| API | Input length limits | Chat questions capped at 2000 chars; role values validated |
| API | CORS whitelist | Only Vercel domain + localhost; credentials allowed |
| API | No stack traces | Global exception handler returns generic `"Internal server error"` to clients |
| Config | Secrets via env | All secrets in `.env`; `.env.example` committed with placeholders |
| Auth | JWT authentication | bcrypt password hashing, JWT tokens, per-user document isolation, chat history stored per user in Postgres (Neon) |
| Auth | Personal access tokens | `dm_` + 32 random bytes, created in Settings and sent as `Authorization: Bearer dm_…`. Only the SHA-256 is stored; the token is shown once. Revoked tokens stop working immediately. Tokens can't create or revoke tokens (session only), and token requests skip the cookie CSRF Origin check, since they never use the cookie |

### What was considered but excluded from scope
- **Virus scanning (ClamAV)** — Would integrate `python-clamd` before any file processing; excluded due to ClamAV daemon dependency on the deployment environment.
- **Document encryption at rest** — Page images and ChromaDB data are stored unencrypted; would add AES-256 envelope encryption with key rotation in production.

### What would be added with more time

- Proper auth layer (Auth0 or Supabase) with per-user document namespacing
- File encryption at rest with KMS-managed keys
- Audit logging (who uploaded/queried what, when)
- Redis for rate limiting and job status (replacing in-memory dict)
- Document-level ACLs (restrict which users can query which docs)
- ClamAV virus scanning in the upload pipeline
- Async job queue (Celery + Redis) replacing FastAPI BackgroundTasks

---

## Deployment

### Backend — Render

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repo, root directory: `bfai-assessment/backend`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the Render dashboard (`GROQ_API_KEY`, `ALLOWED_ORIGINS`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` — the last three pointing at the **production** Chroma database)
6. Set `ALLOWED_ORIGINS` to include your Vercel URL: `["https://your-app.vercel.app"]`

> **Note**: Render free tier has ephemeral storage — ChromaDB and page images won't persist across deploys. Use a paid plan with a persistent disk, or swap ChromaDB for a hosted vector DB (Pinecone, Qdrant Cloud).

### Frontend — Vercel

1. Import the repo on [Vercel](https://vercel.com)
2. Set root directory to `bfai-assessment/frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`
4. Deploy — Vercel auto-detects Next.js

---

## Sample Documents

Five reproducible PDFs are generated by `scripts/create_sample_docs.py`:

| File | Type | Tests |
|------|------|-------|
| `invoice_acme_2024.pdf` | Invoice | Table extraction, line items, totals |
| `research_paper_ai.pdf` | Academic paper | Multi-page text, results table |
| `medical_report.pdf` | Medical record | Sensitivity classification (confidential), lab values table |
| `financial_summary.pdf` | Financial report | YoY metrics table, segment data |
| `technical_manual.pdf` | Technical manual | API endpoint table, multi-section |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, uvicorn |
| Frontend | Next.js 15, TypeScript, Tailwind CSS v4 |
| LLM | Groq API — LLaMA-3.3 70B Versatile |
| Voice | Groq Whisper Large v3 Turbo |
| OCR | pytesseract + pdf2image (Poppler) |
| PDF parsing | pdfplumber |
| Embeddings | Gemini text-embedding-004 (free tier) |
| Vector DB | ChromaDB (persistent, HNSW cosine) |
| Rate limiting | slowapi |
| Icons | lucide-react |
