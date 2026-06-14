"use client";

import { useEffect, useCallback } from "react";
import { X, ChevronLeft, ChevronRight, FileText } from "lucide-react";
import PdfPage from "@/components/PdfPage";

interface PageViewerProps {
  pdfUrl: string;
  docName: string;
  pageNumber: number;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
}

export default function PageViewer({
  pdfUrl,
  docName,
  pageNumber,
  onClose,
  onPrev,
  onNext,
  hasPrev = false,
  hasNext = false,
}: PageViewerProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && hasPrev && onPrev) onPrev();
      if (e.key === "ArrowRight" && hasNext && onNext) onNext();
    },
    [onClose, onPrev, onNext, hasPrev, hasNext]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-2xl max-w-3xl w-full overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-gray-50">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-indigo-500" />
            <span className="text-sm font-semibold text-gray-800 truncate max-w-xs">
              {docName}
            </span>
            <span className="text-xs text-indigo-600 font-medium bg-indigo-50 px-2 py-0.5 rounded-full">
              Page {pageNumber}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-200 transition-colors text-gray-500 hover:text-gray-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Rendered page image */}
        <div className="overflow-y-auto max-h-[75vh]">
          <PdfPage pdfUrl={pdfUrl} pageNumber={pageNumber} scale={1.5} className="w-full" />
        </div>

        {/* Navigation */}
        {(hasPrev || hasNext) && (
          <div className="flex items-center justify-between px-5 py-3 border-t border-gray-100 bg-gray-50">
            <button
              onClick={onPrev}
              disabled={!hasPrev}
              className="flex items-center gap-1.5 text-sm text-gray-600 hover:text-indigo-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>
            <button
              onClick={onNext}
              disabled={!hasNext}
              className="flex items-center gap-1.5 text-sm text-gray-600 hover:text-indigo-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
