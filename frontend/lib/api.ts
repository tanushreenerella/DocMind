const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Citation {
  doc_name: string;
  page_number: number;
  image_filename: string;
  excerpt: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  has_answer: boolean;
}

export interface UploadJob {
  doc_id: string;
  filename: string;
}

export interface UploadResponse {
  jobs: UploadJob[];
}

export interface Classification {
  document_type: string;
  primary_topic: string;
  sensitivity_level: "public" | "internal" | "confidential" | "restricted";
  sensitivity_reason: string;
  summary: string;
  key_entities: string[];
  content_characteristics: {
    has_tables: boolean;
    has_handwriting: boolean;
    has_images: boolean;
    is_scanned: boolean;
    language: string;
    estimated_formality: string;
  };
}

export interface JobStatus {
  status: "queued" | "processing" | "complete" | "error";
  filename: string;
  stage: string;
  progress: number;
  error: string | null;
  classification: Classification | null;
  page_count: number;
}

export interface IndexedDocument {
  doc_id: string;
  doc_name: string;
  document_type: string;
  sensitivity_level: string;
}

export interface DocumentsResponse {
  documents: IndexedDocument[];
  count: number;
}

// ─── API helpers ──────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((body as { detail?: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

// Silently pings /health until the server responds OK.
// Call on page mount so Render's free-tier instance wakes up before
// the user tries to upload or chat. Resolves when ready, gives up after ~60 s.
export async function warmupBackend(): Promise<boolean> {
  for (let i = 0; i < 30; i++) {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) return true;
    } catch {
      // still waking up
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
  return false;
}

// Retries on network-level failures (TypeError: Failed to fetch / net::ERR_FAILED)
// which happen when Render's free tier is cold-starting (up to 50 s delay).
// Delays: 2 s, 4 s, 8 s, 10 s, 10 s → ~34 s total before giving up.
async function fetchWithRetry(
  input: RequestInfo,
  init?: RequestInit,
  maxAttempts = 6
): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (attempt > 0) {
      const delay = Math.min(2 ** attempt * 1000, 10000); // cap at 10 s
      await new Promise((r) => setTimeout(r, delay));
    }
    try {
      return await fetch(input, init);
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError;
}

export async function sendMessage(
  question: string,
  history: Message[],
  docId?: string
): Promise<ChatResponse> {
  const res = await fetchWithRetry(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_history: history,
      ...(docId ? { doc_id: docId } : {}),
    }),
  });
  return handleResponse<ChatResponse>(res);
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetchWithRetry(`${API_BASE}/api/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse<UploadResponse>(res);
}

export async function getStatus(docId: string): Promise<JobStatus> {
  const res = await fetchWithRetry(`${API_BASE}/api/status/${encodeURIComponent(docId)}`);
  return handleResponse<JobStatus>(res);
}

export async function transcribeAudio(
  base64Audio: string
): Promise<{ transcript: string }> {
  const res = await fetchWithRetry(`${API_BASE}/api/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_base64: base64Audio }),
  });
  return handleResponse<{ transcript: string }>(res);
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await fetchWithRetry(`${API_BASE}/api/documents`);
  return handleResponse<DocumentsResponse>(res);
}

export function getPageImageUrl(imageFilename: string): string {
  return `${API_BASE}/api/page-image/${encodeURIComponent(imageFilename)}`;
}
