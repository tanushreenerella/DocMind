"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2 } from "lucide-react";
import MessageBubble from "@/components/MessageBubble";
import VoiceInput from "@/components/VoiceInput";
import { sendMessage } from "@/lib/api";
import type { Message, Citation } from "@/lib/api";

interface AIMessage extends Message {
  citations?: Citation[];
}

interface ChatWindowProps {
  docCount: number;
}

export default function ChatWindow({ docCount }: ChatWindowProps) {
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const adjustTextarea = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  };

  const handleSend = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg: AIMessage = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const history: Message[] = messages.map((m) => ({ role: m.role, content: m.content }));
      const result = await sendMessage(q, history);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: result.answer, citations: result.citations },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong. Please try again.", citations: [] },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // VoiceInput now auto-sends; append to input as fallback in standalone ChatWindow usage
  const handleVoiceSend = useCallback(
    (text: string) => {
      setInput((prev) => (prev ? prev + " " + text : text));
      setTimeout(adjustTextarea, 0);
    },
    []
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 bg-white">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4 py-16">
            <div className="w-16 h-16 rounded-2xl bg-linear-to-br from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg">
              <span className="text-2xl">💬</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-700">Ask your documents</h3>
              <p className="text-sm text-gray-400 mt-1 max-w-xs">
                {docCount === 0
                  ? "Upload documents first, then ask me anything about them."
                  : `${docCount} document${docCount !== 1 ? "s" : ""} indexed. Ask me anything.`}
              </p>
            </div>
            {docCount === 0 && (
              <a
                href="/upload"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-700 underline underline-offset-2"
              >
                Upload documents →
              </a>
            )}
          </div>
        ) : (
          messages.map((msg, i) => <MessageBubble key={i} message={msg} />)
        )}

        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-violet-500 flex items-center justify-center">
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
      <div className="border-t border-gray-100 bg-white px-4 py-3">
        <div className="flex items-end gap-2 max-w-4xl mx-auto">
          <div className="flex-1 flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-2 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => { setInput(e.target.value); adjustTextarea(); }}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your documents… (Enter to send)"
              rows={1}
              className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-800 placeholder-gray-400 py-1 max-h-30"
            />
            <VoiceInput onSend={handleVoiceSend} disabled={loading} />
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="p-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded-xl transition-all duration-200 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-center text-xs text-gray-300 mt-2">
          Shift+Enter for new line · Click mic for voice
        </p>
      </div>
    </div>
  );
}
