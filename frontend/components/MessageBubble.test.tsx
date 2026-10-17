import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import MessageBubble from "@/components/MessageBubble";

// pdf.js can't run in jsdom; the viewer is replaced with a probe that exposes its props.
vi.mock("@/components/PdfPage", () => ({ default: () => null }));
vi.mock("@/components/PageViewer", () => ({
  default: ({ docName, pageNumber }: { docName: string; pageNumber: number }) => (
    <div data-testid="page-viewer">
      {docName} page {pageNumber}
    </div>
  ),
}));

afterEach(cleanup);

const maliciousContent = [
  "Intro <script>window.__xss = true</script> text",
  '<img src="x" onerror="window.__xss = true">',
  "[click me](javascript:alert(1))",
  "![tracker](https://evil.example/pixel.png)",
  "Revenue grew 12% [report.pdf, p.3].",
].join("\n\n");

describe("MessageBubble", () => {
  it("renders no active content from model output and keeps citations clickable", () => {
    const { container } = render(
      <MessageBubble
        message={{
          role: "assistant",
          content: maliciousContent,
          citations: [
            {
              doc_name: "report.pdf",
              page_number: 3,
              image_filename: "",
              doc_id: "doc-1",
              excerpt: "Revenue grew 12%...",
            },
          ],
        }}
      />
    );

    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("[onerror]")).toBeNull();
    for (const link of container.querySelectorAll("a")) {
      expect(link.getAttribute("href") ?? "").not.toMatch(/^\s*javascript:/i);
    }
    expect(screen.getByText("click me").closest("a")).toBeNull();

    const citation = screen.getByRole("button", { name: "[report.pdf, p.3]" });
    fireEvent.click(citation);
    expect(screen.getByTestId("page-viewer").textContent).toBe("report.pdf page 3");
  });
});
