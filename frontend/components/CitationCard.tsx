"use client";

import PdfPage from "@/components/PdfPage";
import { getPdfUrl } from "@/lib/api";

interface CitationCardProps {
  docName: string;
  pageNumber: number;
  docId: string;
  excerpt: string;
  onExpand: (pdfUrl: string, docName: string, pageNumber: number) => void;
}

export default function CitationCard({
  docName,
  pageNumber,
  docId,
  excerpt,
  onExpand,
}: CitationCardProps) {
  const pdfUrl = getPdfUrl(docId);

  return (
    <div className="flex gap-3 p-3 bg-gray-50 border border-gray-200 rounded-xl hover:border-indigo-200 hover:bg-indigo-50/30 transition-all duration-200">
      {/* Thumbnail — renders the actual PDF page */}
      <button
        onClick={() => onExpand(pdfUrl, docName, pageNumber)}
        className="shrink-0 w-16 h-20 rounded-lg overflow-hidden border border-gray-200 hover:scale-105 hover:shadow-md transition-all duration-200 cursor-pointer"
        title="View page"
      >
        <PdfPage pdfUrl={pdfUrl} pageNumber={pageNumber} scale={0.15} className="w-full h-full" />
      </button>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-800 truncate">{docName}</p>
        <p className="text-xs text-indigo-600 font-medium mt-0.5">Page {pageNumber}</p>
        <p className="text-xs text-gray-500 mt-1.5 line-clamp-3 leading-relaxed">
          {excerpt}
        </p>
      </div>
    </div>
  );
}
