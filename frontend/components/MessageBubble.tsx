"use client";

import { useState, type ReactNode } from "react";
import { Bot, User } from "lucide-react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationCard from "@/components/CitationCard";
import PageViewer from "@/components/PageViewer";
import { getPdfUrl } from "@/lib/api";
import type { Citation, Message } from "@/lib/api";

const CITE_SCHEME = "cite:";
const CITATION_PATTERN = /\[([^\]]+),\s*p\.(\d+)\]/g;
const CITATION_PILL_CLASS =
  "inline-flex items-center gap-0.5 text-indigo-600 font-medium text-xs bg-indigo-50 px-1.5 py-0.5 rounded mx-0.5";

// Rewrite "[doc, p.N]" markers into cite: links so the `a` override can render
// them as citation pills. The link text is ignored; the label comes from the href.
function citationsToLinks(content: string): string {
  return content.replace(CITATION_PATTERN, (_match, docName: string, page: string) => {
    const target = encodeURIComponent(docName).replace(/\(/g, "%28").replace(/\)/g, "%29");
    return `[cite](${CITE_SCHEME}${target}/${page})`;
  });
}

function parseCiteHref(href: string): { docName: string; pageNumber: number } | null {
  const body = href.slice(CITE_SCHEME.length);
  const slash = body.lastIndexOf("/");
  if (slash < 0) return null;
  const pageNumber = Number(body.slice(slash + 1));
  if (!Number.isInteger(pageNumber)) return null;
  try {
    return { docName: decodeURIComponent(body.slice(0, slash)), pageNumber };
  } catch {
    return null;
  }
}

function isHttpUrl(href: string): boolean {
  try {
    const { protocol } = new URL(href);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

// Keep cite: links for the `a` override; every other URL gets react-markdown's
// default sanitization (javascript:, data:, etc. become empty).
function urlTransform(url: string): string {
  return url.startsWith(CITE_SCHEME) ? url : defaultUrlTransform(url);
}

interface MarkdownLinkProps {
  href?: string;
  children?: ReactNode;
  citations?: Citation[];
  onOpenCitation: (citation: Citation) => void;
}

function MarkdownLink({ href, children, citations, onOpenCitation }: MarkdownLinkProps) {
  if (href?.startsWith(CITE_SCHEME)) {
    const parsed = parseCiteHref(href);
    if (!parsed) return <>{children}</>;
    const label = `[${parsed.docName}, p.${parsed.pageNumber}]`;
    const citation = citations?.find(
      (c) => c.doc_name === parsed.docName && c.page_number === parsed.pageNumber
    );
    if (!citation) return <span className={CITATION_PILL_CLASS}>{label}</span>;
    return (
      <button
        type="button"
        onClick={() => onOpenCitation(citation)}
        className={`${CITATION_PILL_CLASS} cursor-pointer`}
        title="View page"
      >
        {label}
      </button>
    );
  }

  if (href && isHttpUrl(href)) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-600 underline">
        {children}
      </a>
    );
  }

  return <>{children}</>;
}

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

  const markdownComponents: Components = {
    a: ({ href, children }) => (
      <MarkdownLink
        href={href}
        citations={message.citations}
        onOpenCitation={(c) => handleExpand(getPdfUrl(c.doc_id), c.doc_name, c.page_number)}
      >
        {children}
      </MarkdownLink>
    ),
    p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
    ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 last:mb-0 space-y-1">{children}</ol>,
    ul: ({ children }) => <ul className="list-disc pl-5 mb-3 last:mb-0 space-y-1">{children}</ul>,
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
              px-4 py-3 rounded-2xl text-sm leading-relaxed
              ${isUser
                ? "bg-indigo-500 text-white rounded-tr-sm whitespace-pre-wrap"
                : "bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm"
              }
            `}
          >
            {isUser ? (
              message.content
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents}
                urlTransform={urlTransform}
                disallowedElements={["img"]}
              >
                {citationsToLinks(message.content)}
              </ReactMarkdown>
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
