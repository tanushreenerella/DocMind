"use client";

import { useEffect, useRef, useState } from "react";
import { FileText } from "lucide-react";

interface PdfPageProps {
  pdfUrl: string;
  pageNumber: number;
  scale?: number;
  className?: string;
}

export default function PdfPage({
  pdfUrl,
  pageNumber,
  scale = 1.5,
  className = "",
}: PdfPageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Each outcome is tagged with the inputs it was rendered for, so a new page
  // starts out not-ready without resetting state inside the effect.
  const renderKey = `${pdfUrl}#${pageNumber}@${scale}`;
  const [outcome, setOutcome] = useState<{ key: string; status: "ready" | "error" } | null>(null);
  const ready = outcome?.key === renderKey && outcome.status === "ready";
  const error = outcome?.key === renderKey && outcome.status === "error";

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const pdfjs = await import("pdfjs-dist");
        // Use unpkg CDN worker — version matches the installed package automatically
        pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

        const pdf = await pdfjs.getDocument({
          url: pdfUrl,
          withCredentials: true,
        }).promise;
        const page = await pdf.getPage(pageNumber);
        const viewport = page.getViewport({ scale });

        if (cancelled || !canvasRef.current) return;

        const canvas = canvasRef.current;
        canvas.width = viewport.width;
        canvas.height = viewport.height;

        await page.render({ canvasContext: canvas.getContext("2d")!, viewport,canvas }).promise;

        if (!cancelled) setOutcome({ key: renderKey, status: "ready" });
      } catch {
        if (!cancelled) setOutcome({ key: renderKey, status: "error" });
      }
    }

    render();
    return () => { cancelled = true; };
  }, [pdfUrl, pageNumber, scale, renderKey]);

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center bg-gray-100 gap-2 ${className}`}>
        <FileText className="w-8 h-8 text-gray-400" />
        <p className="text-xs text-gray-400">Preview unavailable</p>
      </div>
    );
  }

  return (
    <div className={`relative bg-gray-100 ${className}`}>
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      <canvas
        ref={canvasRef}
        className="w-full h-auto"
        style={{ visibility: ready ? "visible" : "hidden" }}
      />
    </div>
  );
}
