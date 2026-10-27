import { clearUser, saveUser, type AuthUser } from "./auth";

// Same-origin paths: middleware.ts proxies /api/* to the backend, so the
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

/**
 * Set when the backend couldn't answer because a service failed (Groq busy or
 * down, document search down, or the answer ran past its deadline). The answer
 * text then explains this and asks the user to try again shortly.
 */
export type ServiceErrorCode = "ai_unavailable" | "search_unavailable" | "timeout" | "service_unavailable";

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  has_answer: boolean;
  error?: ServiceErrorCode | null;
}

export interface EvidenceTimeline {
  briefing: string;
  citations: Citation[];
  has_timeline: boolean;
  error?: ServiceErrorCode | null;
}

/**
 * The conversation to send with the next question: without failed turns (an
 * error notice and the question it answered), which the backend doesn't keep either.
 */
export function conversationHistory(messages: (Message & { isError?: boolean })[]): Message[] {
  const history: Message[] = [];
  for (const m of messages) {
    if (m.isError) {
      if (history.at(-1)?.role === "user") history.pop();
      continue;
    }
    history.push({ role: m.role, content: m.content });
  }
  return history;
}

export interface UploadJob {
  doc_id: string;
  filename: string;
}

/** From POST /api/uploads/presign: where and how to PUT the file. */
export interface PresignedUpload {
  doc_id: string;
  filename: string;
  /** A presigned R2 URL, or /api/uploads/{doc_id}/file when the backend stores on local disk. */
  upload_url: string;
  method: "PUT";
  headers: Record<string, string>;
  expires_in: number;
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
  /** "partial": indexed, but OCR failed on `failed_pages`. */
  status: "queued" | "processing" | "complete" | "partial" | "error";
  filename: string;
  stage: string;
  progress: number;
  classification: Classification | null;
  page_count: number;
  failed_pages: number[];
  /** Hidden text (white, invisible or tiny) was found and left out of the index. */
  suspicious: boolean;
  hidden_text_pages: number[];
  /** Characters removed, across all pages. */
  hidden_text_removed: number;
}

export interface IndexedDocument {
  doc_id: string;
  doc_name: string;
  document_type: string;
  sensitivity_level: string;
  suspicious: boolean;
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

/**
 * The server's message for 429s and any status in `passThrough` (whose detail is
 * written for users); `fallback` for every other failure.
 */
export function userFacingError(
  err: unknown,
  fallback: string,
  passThrough: readonly number[] = []
): string {
  if (err instanceof ApiError && (err.status === 429 || passThrough.includes(err.status))) {
    return err.message;
  }
  return fallback;
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
    if (res.status === 429) {
      // Which limit was hit (per-minute rate, daily AI budget) is in the detail.
      const body = await res.json().catch(() => ({}));
      const detail = (body as { detail?: unknown }).detail;
      throw new ApiError(typeof detail === "string" && detail ? detail : RATE_LIMIT_MESSAGE, 429);
    }
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

/** Mirrors the backend's MAX_UPLOAD_MB default; the server enforces the real limit. */
export const MAX_UPLOAD_MB = 50;
export const FILE_TOO_LARGE_MESSAGE = `File is too large — the limit is ${MAX_UPLOAD_MB} MB.`;
export const NETWORK_ERROR_MESSAGE = "Network error — check your connection and try again.";

/** Sending the file bytes failed (the PUT, not an API call). */
export class UploadError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
  }
}

async function putFile(ticket: PresignedUpload, file: File): Promise<void> {
  let res: Response;
  try {
    // One attempt: retrying a large body on a flaky connection rarely helps.
    res = await fetch(ticket.upload_url, { method: ticket.method, headers: ticket.headers, body: file });
  } catch {
    throw new UploadError(NETWORK_ERROR_MESSAGE);
  }
  if (res.ok) return;
  if (res.status === 413) throw new UploadError(FILE_TOO_LARGE_MESSAGE, 413);
  if (res.status === 403) {
    throw new UploadError("The upload link expired or was rejected — please try again.", 403);
  }
  if (res.status === 401) redirectToLogin(); // session ended (local-disk route only)
  // The local-disk route answers JSON like the rest of the API; R2 answers XML.
  const body = await res.json().catch(() => null);
  const detail = (body as { detail?: string } | null)?.detail;
  throw new UploadError(detail ?? `Upload failed (HTTP ${res.status}) — please try again.`, res.status);
}

/**
 * presign → PUT the file (straight to R2, or through the API on local disk) →
 * complete, which checks it and starts processing. Throws ApiError/UploadError.
 */
export async function uploadFile(file: File): Promise<UploadJob> {
  const presign = await fetchWithRetry("/api/uploads/presign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // file.type is empty on some systems; the extension was checked already.
    body: JSON.stringify({ filename: file.name, size: file.size, content_type: "application/pdf" }),
  });
  const ticket = await handleResponse<PresignedUpload>(presign);

  await putFile(ticket, file);

  const res = await fetchWithRetry(`/api/uploads/${encodeURIComponent(ticket.doc_id)}/complete`, {
    method: "POST",
  });
  const done = await handleResponse<UploadJob & { page_count: number }>(res);
  return { doc_id: done.doc_id, filename: done.filename };
}

/** What to tell the user when uploadFile fails. */
export function uploadErrorMessage(err: unknown): string {
  if ((err instanceof ApiError || err instanceof UploadError) && err.status === 413) {
    return FILE_TOO_LARGE_MESSAGE;
  }
  if (err instanceof UploadError) return err.message;
  // fetch rejects with a TypeError when the request never got a response.
  if (err instanceof TypeError) return NETWORK_ERROR_MESSAGE;
  // 400/409/422 details are written for users (not a PDF, too many pages...).
  return userFacingError(err, "Upload failed — please try again.", [400, 409, 422]);
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

/** Deletes a document and everything derived from it. 409 while it is still processing. */
export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetchWithRetry(`/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  // 204 has no body, so only the error path goes through handleResponse.
  if (!res.ok) await handleResponse<never>(res);
}

export async function generateEvidenceTimeline(docId: string): Promise<EvidenceTimeline> {
  const res = await fetchWithRetry(`/api/documents/${encodeURIComponent(docId)}/evidence-timeline`, {
    method: "POST",
  });
  return handleResponse<EvidenceTimeline>(res);
}

// ─── Personal access tokens ───────────────────────────────────────────────────

/** A personal access token as listed: never the secret itself. */
export interface AccessToken {
  id: string;
  name: string;
  /** "dm_" plus the first few characters, to tell tokens apart. */
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

/** POST /api/tokens: the only response that ever contains the full token. */
export interface CreatedAccessToken extends AccessToken {
  token: string;
}

export async function listAccessTokens(): Promise<AccessToken[]> {
  const res = await fetchWithRetry("/api/tokens");
  return (await handleResponse<{ tokens: AccessToken[] }>(res)).tokens;
}

export async function createAccessToken(name: string): Promise<CreatedAccessToken> {
  // No retry: resending after a dropped response would mint a second token the user never sees.
  const res = await fetch("/api/tokens", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return handleResponse<CreatedAccessToken>(res);
}

export async function revokeAccessToken(tokenId: string): Promise<void> {
  const res = await fetchWithRetry(`/api/tokens/${encodeURIComponent(tokenId)}`, { method: "DELETE" });
  if (!res.ok) await handleResponse<never>(res);
}

export function getPageImageUrl(imageFilename: string): string {
  return `/api/page-image/${encodeURIComponent(imageFilename)}`;
}

export function getPdfUrl(docId: string): string {
  return `/api/pdf/${encodeURIComponent(docId)}`;
}
