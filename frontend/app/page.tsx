"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { Upload, Brain, Menu, X, Bot, User, Loader2, Send, FileText } from "lucide-react";
import MessageBubble from "@/components/MessageBubble";
import VoiceInput from "@/components/VoiceInput";
import { listDocuments, sendMessage, getPageImageUrl, warmupBackend } from "@/lib/api";
import type { IndexedDocument, Citation } from "@/lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SENSITIVITY_COLORS: Record<string, string> = {
  public: "bg-green-100 text-green-700",
  internal: "bg-yellow-100 text-yellow-700",
  confidential: "bg-orange-100 text-orange-700",
  restricted: "bg-red-100 text-red-700",
};

const TYPE_ICONS: Record<string, string> = {
  invoice: "🧾",
  report: "📊",
  contract: "📜",
  academic_paper: "🎓",
  handwritten_note: "✍️",
  medical_record: "🏥",
  legal: "⚖️",
  financial: "💰",
  technical_manual: "🔧",
  other: "📄",
};

function typeLabel(t: string) {
  return t.replace(/_/g, " ");
}

// ─── localStorage helpers ─────────────────────────────────────────────────────

function loadHistory(docId: string): StoredMessage[] {
  try {
    const raw = localStorage.getItem(`chat_${docId}`);
    return raw ? (JSON.parse(raw) as StoredMessage[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(docId: string, messages: StoredMessage[]) {
  try {
    localStorage.setItem(`chat_${docId}`, JSON.stringify(messages));
  } catch {
    /* storage full — ignore */
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [docs, setDocs] = useState<IndexedDocument[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<IndexedDocument | null>(null);
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [serverReady, setServerReady] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Wake up Render free-tier instance on mount
  useEffect(() => {
    warmupBackend().then((ok) => setServerReady(ok));
  }, []);

  // Fetch document list only after server is confirmed awake
  useEffect(() => {
    if (!serverReady) return;
    listDocuments()
      .then((r) => setDocs(r.documents))
      .catch(() => {});
  }, [serverReady]);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Persist messages whenever they change
  useEffect(() => {
    if (selectedDoc) saveHistory(selectedDoc.doc_id, messages);
  }, [messages, selectedDoc]);

  const selectDoc = (doc: IndexedDocument) => {
    if (selectedDoc?.doc_id === doc.doc_id) {
      setSidebarOpen(false);
      return;
    }
    setMessages(loadHistory(doc.doc_id));
    setSelectedDoc(doc);
    setInput("");
    setSidebarOpen(false);
  };

  const adjustTextarea = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  };

  const handleSend = useCallback(
    async (text?: string) => {
      const q = (text ?? input).trim();
      if (!q || !selectedDoc || loading) return;

      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";

      const userMsg: StoredMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: q,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const history = messages.map((m) => ({ role: m.role, content: m.content }));
        const result = await sendMessage(q, history, selectedDoc.doc_id);
        const aiMsg: StoredMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          citations: result.citations,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, aiMsg]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Something went wrong. Please try again.",
            timestamp: Date.now(),
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, selectedDoc, loading, messages]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ─── Sidebar ──────────────────────────────────────────────────────────────

  const Sidebar = (
    <aside
      className={`
        fixed lg:relative inset-y-0 left-0 z-30 w-72 flex flex-col
        bg-white border-r border-gray-200
        transition-transform duration-300 ease-in-out
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}
    >
      {/* Logo */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-linear-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <span className="text-lg font-bold text-gray-900">DocMind</span>
        </div>
        <button
          onClick={() => setSidebarOpen(false)}
          className="lg:hidden p-1 rounded-lg hover:bg-gray-100 text-gray-400"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Upload link */}
      <div className="px-3 py-3">
        <Link
          href="/upload"
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-gray-600 hover:bg-gray-100 font-medium text-sm transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload Documents
        </Link>
      </div>

      <div className="mx-4 border-t border-gray-100" />

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-3 py-3">
        <div className="flex items-center justify-between px-2 mb-2">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Knowledge Base
          </span>
          <span className="text-xs bg-indigo-100 text-indigo-600 font-semibold px-2 py-0.5 rounded-full">
            {docs.length}
          </span>
        </div>

        {docs.length === 0 ? (
          <div className="text-center py-8 px-4">
            <FileText className="w-8 h-8 text-gray-300 mx-auto mb-2" />
            <p className="text-xs text-gray-400">No documents yet.</p>
            <Link href="/upload" className="text-xs text-indigo-500 hover:text-indigo-600 mt-1 inline-block">
              Upload some →
            </Link>
          </div>
        ) : (
          <div className="space-y-0.5">
            {docs.map((doc) => {
              const isActive = selectedDoc?.doc_id === doc.doc_id;
              return (
                <button
                  key={doc.doc_id}
                  onClick={() => selectDoc(doc)}
                  className={`
                    w-full text-left px-3 py-2.5 rounded-xl transition-all duration-150
                    ${isActive
                      ? "bg-indigo-50 border-l-2 border-indigo-500 pl-2.5"
                      : "hover:bg-gray-50 border-l-2 border-transparent"
                    }
                  `}
                >
                  <div className="flex items-start gap-2">
                    <span className="text-base mt-0.5 shrink-0">
                      {TYPE_ICONS[doc.document_type] ?? "📄"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p
                        className={`text-sm font-medium truncate ${
                          isActive ? "text-indigo-700" : "text-gray-700"
                        }`}
                      >
                        {doc.doc_name}
                      </p>
                      <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                        <span className="text-xs text-gray-400 capitalize">
                          {typeLabel(doc.document_type)}
                        </span>
                        <span
                          className={`text-xs px-1.5 py-0.5 rounded-full font-medium capitalize ${
                            SENSITIVITY_COLORS[doc.sensitivity_level] ?? "bg-gray-100 text-gray-600"
                          }`}
                        >
                          {doc.sensitivity_level}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );

  // ─── Main content ─────────────────────────────────────────────────────────

  const noDocSelected = !selectedDoc;
  const docSelectedNoMessages = selectedDoc && messages.length === 0 && !loading;

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {Sidebar}

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden bg-white">
        {/* Header */}
        <header className="flex items-center justify-between px-5 py-3 bg-white border-b border-gray-100 shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 text-gray-500"
            >
              <Menu className="w-5 h-5" />
            </button>

            {selectedDoc ? (
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xl shrink-0">{TYPE_ICONS[selectedDoc.document_type] ?? "📄"}</span>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h1 className="text-sm font-semibold text-gray-900 truncate max-w-50">
                      {selectedDoc.doc_name}
                    </h1>
                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full capitalize shrink-0">
                      {typeLabel(selectedDoc.document_type)}
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-medium capitalize shrink-0 ${
                        SENSITIVITY_COLORS[selectedDoc.sensitivity_level] ?? "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {selectedDoc.sensitivity_level}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {messages.length > 0
                      ? `${messages.length} message${messages.length !== 1 ? "s" : ""} · Only searching this document`
                      : "Only searching this document"}
                  </p>
                </div>
              </div>
            ) : (
              <div>
                <h1 className="text-base font-semibold text-gray-900">DocMind</h1>
                <p className="text-xs text-gray-400">Select a document to start chatting</p>
              </div>
            )}
          </div>

          <Link
            href="/upload"
            className="flex items-center gap-1.5 text-sm font-medium text-white bg-indigo-500 hover:bg-indigo-600 px-4 py-2 rounded-xl transition-colors shrink-0"
          >
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">Upload</span>
          </Link>
        </header>

        {/* Cold-start banner — disappears once /health responds OK */}
        {!serverReady && (
          <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 flex items-center justify-center gap-2 text-sm text-amber-700 shrink-0">
            <Loader2 className="w-4 h-4 animate-spin shrink-0" />
            <span>Server starting up (free tier) — please wait ~30 s before chatting.</span>
          </div>
        )}

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {/* State 1: no document selected */}
          {noDocSelected && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-linear-to-br from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-700">Select a document to start chatting</h2>
                <p className="text-sm text-gray-400 mt-1 max-w-xs">
                  Choose a document from the sidebar to ask questions about it.
                </p>
              </div>
              {docs.length === 0 && (
                <Link
                  href="/upload"
                  className="text-sm font-medium text-indigo-600 hover:text-indigo-700 underline underline-offset-2"
                >
                  Upload documents first →
                </Link>
              )}
            </div>
          )}

          {/* State 2: document selected, no messages */}
          {docSelectedNoMessages && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-3">
              <span className="text-5xl">{TYPE_ICONS[selectedDoc.document_type] ?? "📄"}</span>
              <div>
                <h2 className="text-base font-semibold text-gray-700">
                  Ask anything about
                </h2>
                <h2 className="text-base font-semibold text-indigo-600 truncate max-w-xs">
                  {selectedDoc.doc_name}
                </h2>
                <p className="text-sm text-gray-400 mt-2 max-w-xs">
                  Questions in this chat only search this document&apos;s content.
                </p>
              </div>
            </div>
          )}

          {/* State 3: messages */}
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={{ role: msg.role, content: msg.content, citations: msg.citations }}
            />
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-violet-500 flex items-center justify-center shrink-0">
                <Loader2 className="w-4 h-4 text-white animate-spin" />
              </div>
              <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1 items-center h-5">
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-gray-100 bg-white px-4 py-3 shrink-0">
          <div className="flex items-end gap-2 max-w-4xl mx-auto">
            <div
              className={`
                flex-1 flex items-end gap-2 bg-gray-50 border rounded-2xl px-3 py-2
                transition-all duration-200
                ${selectedDoc
                  ? "border-gray-200 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100"
                  : "border-gray-200 opacity-50 cursor-not-allowed"
                }
              `}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => { setInput(e.target.value); adjustTextarea(); }}
                onKeyDown={handleKeyDown}
                placeholder={
                  selectedDoc
                    ? `Ask about ${selectedDoc.doc_name}… (Enter to send)`
                    : "Select a document first"
                }
                disabled={!selectedDoc || loading}
                rows={1}
                className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-800 placeholder-gray-400 py-1 max-h-30 disabled:cursor-not-allowed"
              />
              <VoiceInput onSend={handleSend} disabled={!selectedDoc || loading} />
            </div>
            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || !selectedDoc || loading}
              className="p-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-xl transition-all duration-200 disabled:cursor-not-allowed shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-center text-xs text-gray-300 mt-2">
            Shift+Enter for new line · Mic auto-sends after 1.5 s silence
          </p>
        </div>
      </main>
    </div>
  );
}
