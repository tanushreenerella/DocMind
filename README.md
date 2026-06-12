# DocMind — Document Intelligence + Agentic RAG

A production-quality web application that ingests messy real-world documents (scanned PDFs, handwritten pages, image-heavy reports, tables), extracts content accurately, classifies each document, and powers a chatbot that answers questions with grounded citations showing the exact source page.

---

## Overview

**What it does**

1. **Upload** — drag-and-drop PDFs or images; background processing pipeline handles parsing, OCR, classification, and vector indexing
2. **Parse** — pdfplumber extracts text and tables; pytesseract OCR handles scanned/handwritten pages automatically
3. **Classify** — Groq LLaMA-3 70B classifies each document (type, topic, sensitivity level, key entities, summary)
4. **Index** — sentence-transformers (all-MiniLM-L6-v2) embeds 500-char chunks into ChromaDB with persistent disk storage
5. **Chat** — ask questions in natural language; the RAG pipeline retrieves relevant chunks, synthesizes a grounded answer with inline `[Doc, p.N]` citation markers, and shows clickable page thumbnails

---

## Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│   Next.js Frontend      │        │         FastAPI Backend              │
│  (Vercel)               │        │  (Render)                            │
│                         │        │                                      │
│  /           ChatWindow ├──POST──▶  /api/chat   → RAG service           │
│  /upload     UploadZone ├──POST──▶  /api/upload → Parser + Embedder     │
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
| `MAX_FILE_SIZE_MB` | | `20` | Max upload size per file |
| `STORAGE_PATH` | | `./storage` | Base storage directory |
| `CHROMA_PATH` | | `./storage/chroma_db` | ChromaDB persistence path |
| `IMAGES_PATH` | | `./storage/page_images` | Rendered page images path |
| `RATE_LIMIT_PER_MINUTE` | | `30` | API rate limit per IP |

### Frontend (`frontend/.env.local`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | ✅ | `http://localhost:8000` | Backend API base URL |

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

### What was considered but excluded from scope

- **Authentication / JWT** — No user accounts in scope for this assessment. Would add OAuth2 + JWT for multi-user production deployment.
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
5. Add environment variables in the Render dashboard (`GROQ_API_KEY`, `ALLOWED_ORIGINS`)
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
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, free) |
| Vector DB | ChromaDB (persistent, HNSW cosine) |
| Rate limiting | slowapi |
| Icons | lucide-react |
