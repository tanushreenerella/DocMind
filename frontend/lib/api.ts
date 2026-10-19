import { clearUser, saveUser, type AuthUser } from "./auth";

// Same-origin paths: next.config rewrites /api/* to the backend, so the
// HttpOnly auth cookie is first-party and sent automatically.
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

export interface EvidenceTimeline {
  briefing: string;
  citations: Citation[];
  has_timeline: boolean;
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
  user_id: string;
  email: string;
  full_name: string;
}

interface MeResponse {
  id: string;
  email: string;
  full_name: string | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export const RATE_LIMIT_MESSAGE = "Too many requests, try again shortly";

/** The rate-limit message for 429s; `fallback` for every other failure. */
export function userFacingError(err: unknown, fallback: string): string {
  return err instanceof ApiError && err.status === 429 ? err.message : fallback;
}

function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  clearUser();
  if (window.location.pathname !== "/login") window.location.assign("/login");
}

interface HandleResponseOptions {
  /** Off for login/signup (401 = bad credentials) and /me (AuthGuard redirects itself). */
  redirectOnUnauthorized?: boolean;
}

async function handleResponse<T>(
  res: Response,
  { redirectOnUnauthorized = true }: HandleResponseOptions = {}
): Promise<T> {
  if (!res.ok) {
    if (res.status === 429) throw new ApiError(RATE_LIMIT_MESSAGE, 429);
    if (res.status === 401 && redirectOnUnauthorized) redirectToLogin();
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError((body as { detail?: string }).detail ?? res.statusText, res.status);
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
      const res = await fetch("/health");
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
  const res = await fetchWithRetry("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName }),
  });
  const data = await handleResponse<AuthResponse>(res, { redirectOnUnauthorized: false });
  const user: AuthUser = { user_id: data.user_id, email: data.email, full_name: data.full_name };
  saveUser(user);
  return user;
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const formData = new URLSearchParams();
  formData.append("username", email);
  formData.append("password", password);

  const res = await fetchWithRetry("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: formData.toString(),
  });

  const data = await handleResponse<AuthResponse>(res, { redirectOnUnauthorized: false });

  const user: AuthUser = {
    user_id: data.user_id,
    email: data.email,
    full_name: data.full_name,
  };

  saveUser(user);
  return user;
}

/** Resolves the cookie session; throws ApiError(401) when not logged in. */
export async function getCurrentUser(): Promise<AuthUser> {
  const res = await fetchWithRetry("/api/auth/me");
  const data = await handleResponse<MeResponse>(res, { redirectOnUnauthorized: false });
  return { user_id: data.id, email: data.email, full_name: data.full_name ?? "" };
}

// ─── Protected API ────────────────────────────────────────────────────────────

export async function sendMessage(
  question: string,
  history: Message[],
  docId?: string
): Promise<ChatResponse> {
  const res = await fetchWithRetry("/api/chat", {
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

export async function sendAgenticMessage(
  question: string,
  history: Message[],
  docId?: string
): Promise<ChatResponse> {
  const res = await fetchWithRetry("/api/chat/agentic", {
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
  const res = await fetchWithRetry("/api/upload", {
    method: "POST",
    body: form,
  });
  return handleResponse<UploadResponse>(res);
}

export async function getStatus(docId: string): Promise<JobStatus> {
  const res = await fetchWithRetry(`/api/status/${encodeURIComponent(docId)}`);
  return handleResponse<JobStatus>(res);
}

export async function transcribeAudio(
  base64Audio: string
): Promise<{ transcript: string }> {
  const res = await fetchWithRetry("/api/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_base64: base64Audio }),
  });
  return handleResponse<{ transcript: string }>(res);
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await fetchWithRetry("/api/documents");
  return handleResponse<DocumentsResponse>(res);
}

export async function generateEvidenceTimeline(docId: string): Promise<EvidenceTimeline> {
  const res = await fetchWithRetry(`/api/documents/${encodeURIComponent(docId)}/evidence-timeline`, {
    method: "POST",
  });
  return handleResponse<EvidenceTimeline>(res);
}

export function getPageImageUrl(imageFilename: string): string {
  return `/api/page-image/${encodeURIComponent(imageFilename)}`;
}

export function getPdfUrl(docId: string): string {
  return `/api/pdf/${encodeURIComponent(docId)}`;
}
