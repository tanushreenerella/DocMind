import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  FILE_TOO_LARGE_MESSAGE,
  NETWORK_ERROR_MESSAGE,
  UploadError,
  type JobStatus,
} from "@/lib/api";

const api = vi.hoisted(() => ({
  uploadFile: vi.fn(),
  getStatus: vi.fn(),
  deleteDocument: vi.fn(),
  warmupBackend: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  ...api,
}));
vi.mock("@/components/AuthGuard", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

import UploadPage from "@/app/upload/page";

function status(s: JobStatus["status"], extra: Partial<JobStatus> = {}): JobStatus {
  return {
    status: s,
    filename: "x.pdf",
    stage: s === "partial" ? "complete" : s,
    progress: s === "error" ? 0 : 100,
    classification: s === "complete" || s === "partial"
      ? { document_type: "report", sensitivity_level: "internal", primary_topic: "", summary: "" } as JobStatus["classification"]
      : null,
    page_count: 3,
    failed_pages: [],
    suspicious: false,
    hidden_text_pages: [],
    hidden_text_removed: 0,
    ...extra,
  };
}

const pdf = (name: string) => new File(["%PDF-1.4"], name, { type: "application/pdf" });

/**
 * Selects `names` and clicks Upload. Each file becomes doc-{n} (its position),
 * unless `failures` gives an error for its name.
 */
async function uploadDocs(names: string[], failures: Record<string, unknown> = {}) {
  const { container } = render(<UploadPage />);
  const button = await screen.findByRole("button", { name: /Upload/ });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(true)); // no files yet
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const files = names.map(pdf);
  fireEvent.change(input, { target: { files } });
  api.uploadFile.mockImplementation(async (file: File) => {
    if (file.name in failures) throw failures[file.name];
    return { doc_id: `doc-${names.indexOf(file.name) + 1}`, filename: file.name };
  });
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: /Upload \d+ File/ }));
  });
  return files;
}

function statusesById(map: Record<string, JobStatus | Error>) {
  api.getStatus.mockImplementation(async (docId: string) => {
    const s = map[docId];
    if (s instanceof Error) throw s;
    return s;
  });
}

describe("upload page", () => {
  beforeEach(() => {
    api.warmupBackend.mockResolvedValue(true);
    api.deleteDocument.mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("shows a failed document as 'Failed — try again' without the server's error", async () => {
    // Even if a server ever sent a raw error, the UI must not print it.
    statusesById({ "doc-1": { ...status("error"), error: "ConnectError at 10.0.0.5" } as JobStatus });
    await uploadDocs(["broken.pdf"]);

    const row = (await screen.findByText(/Failed —/)).closest("div") as HTMLElement;
    expect(row.textContent).toContain("Failed — try again");
    expect(screen.getByRole("button", { name: "try again" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("10.0.0.5");
    expect(document.body.textContent).not.toContain("ConnectError");
  });

  it("never shows ✅ when nothing was indexed", async () => {
    statusesById({ "doc-1": status("error"), "doc-2": status("error") });
    await uploadDocs(["a.pdf", "b.pdf"]);

    expect(await screen.findByText(/No documents were indexed/)).toBeTruthy();
    expect(document.body.textContent).not.toContain("✅");
    expect(screen.queryByText("Go to Chat")).toBeNull();
  });

  it("shows ✅ only when every document indexed cleanly", async () => {
    statusesById({ "doc-1": status("complete"), "doc-2": status("complete") });
    await uploadDocs(["a.pdf", "b.pdf"]);

    expect((await screen.findByText(/documents indexed/)).textContent).toContain("✅ 2 documents indexed");
    expect(screen.getByText("Go to Chat")).toBeTruthy();
  });

  it("counts failures in the summary without a ✅", async () => {
    statusesById({ "doc-1": status("complete"), "doc-2": status("error") });
    await uploadDocs(["a.pdf", "b.pdf"]);

    const summary = await screen.findByText(/documents indexed/);
    expect(summary.textContent).toContain("⚠️ 1 of 2 documents indexed");
    expect(summary.textContent).not.toContain("✅");
  });

  it("shows partial documents with the pages that could not be read", async () => {
    statusesById({ "doc-1": status("partial", { failed_pages: [2, 5] }) });
    await uploadDocs(["scan.pdf"]);

    expect(await screen.findByText("Indexed, but couldn't read pages 2, 5")).toBeTruthy();
    expect(screen.getByText(/missing text from some pages/)).toBeTruthy();
    expect(document.body.textContent).not.toContain("✅");
  });

  it("warns about documents whose hidden text was removed", async () => {
    statusesById({
      "doc-1": status("complete", { suspicious: true, hidden_text_pages: [1], hidden_text_removed: 146 }),
      "doc-2": status("complete"),
    });
    await uploadDocs(["invoice.pdf", "clean.pdf"]);

    const badge = await screen.findByText("Hidden text removed");
    expect(screen.getAllByText("Hidden text removed")).toHaveLength(1);
    expect(badge.closest("span")?.getAttribute("title")).toContain("146 characters removed from page 1.");
    expect(screen.getByText("1 document contained hidden text, which was not indexed.")).toBeTruthy();
    expect(document.body.textContent).not.toContain("✅");
  });

  it("try again re-uploads the same file and cleans up the failed attempt", async () => {
    statusesById({ "doc-1": status("error"), "doc-9": status("complete") });
    const [file] = await uploadDocs(["broken.pdf"]);
    await screen.findByRole("button", { name: "try again" });

    api.uploadFile.mockResolvedValueOnce({ doc_id: "doc-9", filename: "broken.pdf" });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "try again" }));
    });

    expect(api.uploadFile).toHaveBeenLastCalledWith(file);
    expect(api.deleteDocument).toHaveBeenCalledWith("doc-1");
    expect(await screen.findByText(/✅ 1 document indexed/)).toBeTruthy();
    expect(screen.queryByText(/Failed —/)).toBeNull();
  });

  it("keeps the failed row when the retry upload itself fails", async () => {
    statusesById({ "doc-1": status("error") });
    await uploadDocs(["broken.pdf"]);
    await screen.findByRole("button", { name: "try again" });

    api.uploadFile.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "try again" }));
    });

    expect(screen.getByRole("button", { name: "try again" })).toBeTruthy();
    expect(screen.getByText(NETWORK_ERROR_MESSAGE)).toBeTruthy();
    expect(api.deleteDocument).not.toHaveBeenCalled();
  });

  it("shows why each file failed to upload and keeps only those selected", async () => {
    statusesById({ "doc-1": status("complete") });
    const pageLimit = "This PDF has 200 pages; the limit is 150. Split it into smaller files and upload them separately.";
    await uploadDocs(["ok.pdf", "huge.pdf", "long.pdf", "offline.pdf"], {
      "huge.pdf": new ApiError("File is too large (61.0 MB); the limit is 50 MB", 413),
      "long.pdf": new ApiError(pageLimit, 422),
      "offline.pdf": new UploadError(NETWORK_ERROR_MESSAGE),
    });

    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain(`huge.pdf: ${FILE_TOO_LARGE_MESSAGE}`);
    expect(alert.textContent).toContain(`long.pdf: ${pageLimit}`);
    expect(alert.textContent).toContain(`offline.pdf: ${NETWORK_ERROR_MESSAGE}`);
    expect(alert.textContent).not.toContain("ok.pdf");

    // The good file is processing; the three failures stay queued for another try.
    expect(await screen.findByText(/1 document indexed/)).toBeTruthy();
    expect(screen.getByText("3 files selected")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Upload 3 Files" })).toBeTruthy();
  });

  it("hides internal server errors behind a generic message", async () => {
    await uploadDocs(["a.pdf"], { "a.pdf": new ApiError("ConnectError at 10.0.0.5", 500) });

    expect(screen.getByRole("alert").textContent).toBe("a.pdf: Upload failed — please try again.");
  });

  it("polls each document once per interval, however often the page re-renders", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let slowCalls = 0;
    api.getStatus.mockImplementation(async (docId: string) => {
      if (docId === "doc-1") return status("complete");
      slowCalls++;
      return slowCalls < 3 ? status("processing", { stage: "parsing", progress: 10 }) : status("complete");
    });
    await uploadDocs(["fast.pdf", "slow.pdf"]);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2500 * 6);
    });

    const calls = (id: string) => api.getStatus.mock.calls.filter(([d]) => d === id).length;
    expect(calls("doc-1")).toBe(1);
    expect(calls("doc-2")).toBe(3);
  });
});
