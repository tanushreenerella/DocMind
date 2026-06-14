"use client";

import { useState } from "react";
import { Bot, User } from "lucide-react";
import CitationCard from "@/components/CitationCard";
import PageViewer from "@/components/PageViewer";
import { getPdfUrl } from "@/lib/api";
import type { Citation, Message } from "@/lib/api";

interface AIMessage extends Message {
  citations?: Citation[];
}

interface MessageBubbleProps {
  message: AIMessage;
}

interface ViewerState {
  pdfUrl: string;
  docName: string;
  pageNumber: number;
  index: number;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const isUser = message.role === "user";

  const handleExpand = (pdfUrl: string, docName: string, pageNumber: number) => {
    const idx = message.citations?.findIndex(
      (c) => c.doc_name === docName && c.page_number === pageNumber
    ) ?? 0;
    setViewer({ pdfUrl, docName, pageNumber, index: idx });
  };

  const handlePrev = () => {
    if (!viewer || !message.citations) return;
    if (viewer.index > 0) {
      const c = message.citations[viewer.index - 1];
      setViewer({
        pdfUrl: getPdfUrl(c.doc_id),
        docName: c.doc_name,
        pageNumber: c.page_number,
        index: viewer.index - 1,
      });
    }
  };

  const handleNext = () => {
    if (!viewer || !message.citations) return;
    if (viewer.index < message.citations.length - 1) {
      const c = message.citations[viewer.index + 1];
      setViewer({
        pdfUrl: getPdfUrl(c.doc_id),
        docName: c.doc_name,
        pageNumber: c.page_number,
        index: viewer.index + 1,
      });
    }
  };

  return (
    <>
      <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        {/* Avatar */}
        <div
          className={`
            shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium
            ${isUser ? "bg-indigo-500" : "bg-violet-500"}
          `}
        >
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Bubble */}
        <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-2`}>
          <div
            className={`
              px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
              ${isUser
                ? "bg-indigo-500 text-white rounded-tr-sm"
                : "bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm"
              }
            `}
          >
            {isUser ? (
              message.content
            ) : (
              <span
                dangerouslySetInnerHTML={{
                  __html: message.content.replace(
                    /\[([^\]]+),\s*p\.(\d+)\]/g,
                    '<span class="inline-flex items-center gap-0.5 text-indigo-600 font-medium text-xs bg-indigo-50 px-1.5 py-0.5 rounded mx-0.5">[$1, p.$2]</span>'
                  ),
                }}
              />
            )}
          </div>

          {/* Citations */}
          {!isUser && message.citations && message.citations.length > 0 && (
            <div className="w-full space-y-2">
              <p className="text-xs text-gray-400 font-medium uppercase tracking-wide pl-1">
                Sources
              </p>
              <div className="grid gap-2">
                {message.citations.map((c, i) => (
                  <CitationCard
                    key={`${c.doc_name}-${c.page_number}-${i}`}
                    docName={c.doc_name}
                    pageNumber={c.page_number}
                    docId={c.doc_id}
                    excerpt={c.excerpt}
                    onExpand={handleExpand}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {viewer && (
        <PageViewer
          pdfUrl={viewer.pdfUrl}
          docName={viewer.docName}
          pageNumber={viewer.pageNumber}
          onClose={() => setViewer(null)}
          onPrev={handlePrev}
          onNext={handleNext}
          hasPrev={viewer.index > 0}
          hasNext={(message.citations?.length ?? 0) - 1 > viewer.index}
        />
      )}
    </>
  );
}
