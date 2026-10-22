import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  FILE_TOO_LARGE_MESSAGE,
  NETWORK_ERROR_MESSAGE,
  UploadError,
  uploadErrorMessage,
  uploadFile,
} from "@/lib/api";

const R2_URL = "https://acct.r2.cloudflarestorage.com/docmind/users/u/d/original.pdf?X-Amz-Signature=abc";

const ticket = (upload_url = R2_URL) => ({
  doc_id: "doc-1",
  filename: "report.pdf",
  upload_url,
  method: "PUT",
  headers: { "Content-Type": "application/pdf" },
  expires_in: 600,
});

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });

type Handler = (init?: RequestInit) => Response | Promise<Response>;

/** Routes fetch by URL: presign, the PUT target, and complete. */
function mockFetch({ presign, put, complete }: { presign?: Handler; put?: Handler; complete?: Handler }) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/uploads/presign") return (presign ?? (() => json(200, ticket())))(init);
    if (url === "/api/uploads/doc-1/complete") {
      return (complete ?? (() => json(200, { doc_id: "doc-1", filename: "report.pdf", page_count: 3 })))(init);
    }
    return (put ?? (() => new Response(null, { status: 200 })))(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const pdf = () => new File(["%PDF-1.4"], "report.pdf", { type: "" });

describe("uploadFile", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("presigns, PUTs the file to the presigned URL, then completes", async () => {
    const fetchMock = mockFetch({});
    const file = pdf();

    await expect(uploadFile(file)).resolves.toEqual({ doc_id: "doc-1", filename: "report.pdf" });

    const [presign, put, complete] = fetchMock.mock.calls;
    expect(presign[0]).toBe("/api/uploads/presign");
    // content_type is always PDF, even when the browser leaves file.type empty.
    expect(JSON.parse(presign[1]!.body as string)).toEqual({
      filename: "report.pdf", size: file.size, content_type: "application/pdf",
    });
    expect(put[0]).toBe(R2_URL);
    expect(put[1]).toMatchObject({ method: "PUT", headers: { "Content-Type": "application/pdf" }, body: file });
    expect(complete[0]).toBe("/api/uploads/doc-1/complete");
    expect(complete[1]).toMatchObject({ method: "POST" });
  });

  it("PUTs to the API route when the backend stores on local disk", async () => {
    const fetchMock = mockFetch({ presign: () => json(200, ticket("/api/uploads/doc-1/file")) });

    await uploadFile(pdf());

    expect(fetchMock.mock.calls[1][0]).toBe("/api/uploads/doc-1/file");
  });

  it("reports a network failure during the PUT and never completes", async () => {
    const fetchMock = mockFetch({ put: () => Promise.reject(new TypeError("Failed to fetch")) });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(err).toBeInstanceOf(UploadError);
    expect(uploadErrorMessage(err)).toBe(NETWORK_ERROR_MESSAGE);
    expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain("/api/uploads/doc-1/complete");
  });

  it("reports a 413 from the upload target as too large", async () => {
    mockFetch({ put: () => new Response("<Error/>", { status: 413 }) });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(uploadErrorMessage(err)).toBe(FILE_TOO_LARGE_MESSAGE);
  });

  it("reports an expired or rejected presigned URL", async () => {
    mockFetch({ put: () => new Response("<Error><Code>AccessDenied</Code></Error>", { status: 403 }) });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(uploadErrorMessage(err)).toMatch(/expired or was rejected/);
  });

  it("shows the local route's JSON detail", async () => {
    mockFetch({
      presign: () => json(200, ticket("/api/uploads/doc-1/file")),
      put: () => json(409, { detail: "This upload has already been completed" }),
    });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(uploadErrorMessage(err)).toBe("This upload has already been completed");
  });

  it("shows presign's 413 as too large without uploading anything", async () => {
    const fetchMock = mockFetch({
      presign: () => json(413, { detail: "File is too large (61.0 MB); the limit is 50 MB" }),
    });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(err).toBeInstanceOf(ApiError);
    expect(uploadErrorMessage(err)).toBe(FILE_TOO_LARGE_MESSAGE);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("passes the page-limit rejection from complete through", async () => {
    const detail = "This PDF has 200 pages; the limit is 150. Split it into smaller files and upload them separately.";
    mockFetch({ complete: () => json(422, { detail }) });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(uploadErrorMessage(err)).toBe(detail);
  });

  it("passes 'not a PDF' from complete through", async () => {
    mockFetch({ complete: () => json(400, { detail: "This file is not a PDF" }) });

    const err = await uploadFile(pdf()).catch((e: unknown) => e);

    expect(uploadErrorMessage(err)).toBe("This file is not a PDF");
  });
});

describe("uploadErrorMessage", () => {
  it("hides server errors and maps network failures", () => {
    expect(uploadErrorMessage(new ApiError("Internal server error", 500))).toBe("Upload failed — please try again.");
    expect(uploadErrorMessage(new TypeError("Failed to fetch"))).toBe(NETWORK_ERROR_MESSAGE);
    expect(uploadErrorMessage(new ApiError("Too many requests, try again shortly", 429))).toBe(
      "Too many requests, try again shortly"
    );
  });

  it("treats any 413 as too large, whatever the body said", () => {
    // e.g. a proxy's HTML error page: handleResponse falls back to statusText.
    expect(uploadErrorMessage(new ApiError("Payload Too Large", 413))).toBe(FILE_TOO_LARGE_MESSAGE);
  });
});
