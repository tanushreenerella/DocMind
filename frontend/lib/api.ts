import { getToken, saveAuth, type AuthUser } from "./auth";

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
  doc_id: string;
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

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  email: string;
  full_name: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((body as { detail?: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function fetchWithRetry(
  input: RequestInfo,
  init?: RequestInit,
  maxAttempts = 6
): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (attempt > 0) {
      const delay = Math.min(2 ** attempt * 1000, 10000);
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

// ─── Auth API ─────────────────────────────────────────────────────────────────

export async function signup(
  email: string,
  password: string,
  fullName: string
): Promise<AuthUser> {
  const res = await fetchWithRetry(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName }),
  });
  const data = await handleResponse<AuthResponse>(res);
  const user: AuthUser = { user_id: data.user_id, email: data.email, full_name: data.full_name };
  saveAuth(data.access_token, user);
  return user;
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const res = await fetchWithRetry(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await handleResponse<AuthResponse>(res);
  const user: AuthUser = { user_id: data.user_id, email: data.email, full_name: data.full_name };
  saveAuth(data.access_token, user);
  return user;
}

// ─── Protected API ────────────────────────────────────────────────────────────

export async function sendMessage(
  question: string,
  history: Message[],
  docId?: string
): Promise<ChatResponse> {
  const res = await fetchWithRetry(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
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
    headers: { ...authHeaders() },
    body: form,
  });
  return handleResponse<UploadResponse>(res);
}

export async function getStatus(docId: string): Promise<JobStatus> {
  const res = await fetchWithRetry(
    `${API_BASE}/api/status/${encodeURIComponent(docId)}`,
    { headers: { ...authHeaders() } }
  );
  return handleResponse<JobStatus>(res);
}

export async function transcribeAudio(
  base64Audio: string
): Promise<{ transcript: string }> {
  const res = await fetchWithRetry(`${API_BASE}/api/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ audio_base64: base64Audio }),
  });
  return handleResponse<{ transcript: string }>(res);
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await fetchWithRetry(`${API_BASE}/api/documents`, {
    headers: { ...authHeaders() },
  });
  return handleResponse<DocumentsResponse>(res);
}

export function getPageImageUrl(imageFilename: string): string {
  return `${API_BASE}/api/page-image/${encodeURIComponent(imageFilename)}`;
}

export function getPdfUrl(docId: string): string {
  return `${API_BASE}/api/pdf/${encodeURIComponent(docId)}`;
}
