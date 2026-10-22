import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { IndexedDocument } from "@/lib/api";

const api = vi.hoisted(() => ({
  listDocuments: vi.fn(),
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
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
// pdf.js can't run in jsdom.
vi.mock("@/components/PdfPage", () => ({ default: () => null }));
vi.mock("@/components/PageViewer", () => ({ default: () => null }));

import HomePage from "@/app/page";
import { ApiError } from "@/lib/api";
import { saveUser } from "@/lib/auth";

const REPORT: IndexedDocument = {
  doc_id: "doc-report",
  doc_name: "report.pdf",
  document_type: "report",
  sensitivity_level: "internal",
};
const INVOICE: IndexedDocument = {
  doc_id: "doc-invoice",
  doc_name: "invoice.pdf",
  document_type: "invoice",
  sensitivity_level: "public",
};
const STILL_PROCESSING = "Document is still processing; try again when it finishes";

async function renderWithDocs(docs: IndexedDocument[]) {
  api.listDocuments.mockResolvedValue({ documents: docs, count: docs.length });
  render(<HomePage />);
  await screen.findByRole("button", { name: `Delete ${docs[0].doc_name}` });
}

function openDeleteDialog(doc: IndexedDocument) {
  fireEvent.click(screen.getByRole("button", { name: `Delete ${doc.doc_name}` }));
  return screen.getByRole("alertdialog");
}

describe("document list fetching", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    saveUser({ user_id: "u1", email: "u1@example.com", full_name: "" });
    api.warmupBackend.mockResolvedValue(true);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("fetches once on load, not on focus, visibility changes or over time", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderWithDocs([REPORT]);
      expect(api.listDocuments).toHaveBeenCalledTimes(1);

      await act(async () => {
        window.dispatchEvent(new Event("focus"));
        document.dispatchEvent(new Event("visibilitychange"));
        await vi.advanceTimersByTimeAsync(5 * 60_000);
      });

      expect(api.listDocuments).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refetches after a delete", async () => {
    api.deleteDocument.mockResolvedValue(undefined);
    await renderWithDocs([REPORT, INVOICE]);
    expect(api.listDocuments).toHaveBeenCalledTimes(1);

    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    await waitFor(() => expect(api.listDocuments).toHaveBeenCalledTimes(2));
  });
});

describe("document delete", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    saveUser({ user_id: "u1", email: "u1@example.com", full_name: "" });
    api.warmupBackend.mockResolvedValue(true);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("asks for confirmation and does nothing on cancel", async () => {
    await renderWithDocs([REPORT, INVOICE]);

    const dialog = openDeleteDialog(REPORT);
    expect(dialog.textContent).toContain("report.pdf");
    // Cancel has focus, so a stray Enter can't delete.
    expect(document.activeElement?.textContent).toBe("Cancel");

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(api.deleteDocument).not.toHaveBeenCalled();
  });

  it("closes on Escape without deleting", async () => {
    await renderWithDocs([REPORT]);
    openDeleteDialog(REPORT);

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(api.deleteDocument).not.toHaveBeenCalled();
  });

  it("deletes after confirming, then refreshes the list", async () => {
    await renderWithDocs([REPORT, INVOICE]);
    localStorage.setItem("chat_u1_doc-report", JSON.stringify([{ id: "m", role: "user", content: "hi", timestamp: 1 }]));
    api.deleteDocument.mockResolvedValue(undefined);
    const listCallsBefore = api.listDocuments.mock.calls.length;
    api.listDocuments.mockResolvedValue({ documents: [INVOICE], count: 1 });

    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    expect(api.deleteDocument).toHaveBeenCalledWith("doc-report");
    expect(api.listDocuments.mock.calls.length).toBeGreaterThan(listCallsBefore);
    await waitFor(() => expect(screen.queryByText("report.pdf")).toBeNull());
    expect(screen.getByText("invoice.pdf")).toBeTruthy();
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(localStorage.getItem("chat_u1_doc-report")).toBeNull();
  });

  it("returns to All Documents when the open document is deleted", async () => {
    await renderWithDocs([REPORT, INVOICE]);
    fireEvent.click(screen.getByText("report.pdf"));
    expect(screen.getAllByText("report.pdf").length).toBeGreaterThan(1); // header shows it too

    api.deleteDocument.mockResolvedValue(undefined);
    api.listDocuments.mockResolvedValue({ documents: [INVOICE], count: 1 });
    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    await waitFor(() => expect(screen.queryByText("report.pdf")).toBeNull());
    expect(screen.getByPlaceholderText(/Ask across all documents/)).toBeTruthy();
  });

  it("shows the still-processing message on 409 and keeps the document", async () => {
    await renderWithDocs([REPORT]);
    api.deleteDocument.mockRejectedValue(new ApiError(STILL_PROCESSING, 409));

    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    expect((await screen.findByRole("alert")).textContent).toBe(STILL_PROCESSING);
    expect(screen.getByRole("alertdialog")).toBeTruthy();
    expect(screen.getAllByText("report.pdf").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Delete" }).hasAttribute("disabled")).toBe(false);
  });

  it("shows a generic message for other failures", async () => {
    await renderWithDocs([REPORT]);
    api.deleteDocument.mockRejectedValue(new ApiError("Internal server error", 500));

    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    expect((await screen.findByRole("alert")).textContent).toBe(
      "Could not delete the document. Please try again."
    );
  });

  it("treats 404 (already deleted elsewhere) as done", async () => {
    await renderWithDocs([REPORT, INVOICE]);
    api.deleteDocument.mockRejectedValue(new ApiError("Document not found", 404));
    api.listDocuments.mockResolvedValue({ documents: [INVOICE], count: 1 });

    openDeleteDialog(REPORT);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    });

    await waitFor(() => expect(screen.queryByText("report.pdf")).toBeNull());
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});
