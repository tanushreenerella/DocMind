"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, Brain, CheckCircle2, AlertCircle, Loader2, MessageSquare } from "lucide-react";
import UploadZone from "@/components/UploadZone";
import { uploadFiles, getStatus } from "@/lib/api";
import type { UploadJob, JobStatus, Classification } from "@/lib/api";

interface TrackedJob extends UploadJob {
  status: JobStatus | null;
  polling: boolean;
}

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  parsing: "Parsing document…",
  classifying: "Classifying…",
  indexing: "Indexing embeddings…",
  complete: "Complete",
  error: "Failed",
};

const SENSITIVITY_COLORS: Record<string, string> = {
  public: "bg-green-100 text-green-700 border-green-200",
  internal: "bg-blue-100 text-blue-700 border-blue-200",
  confidential: "bg-orange-100 text-orange-700 border-orange-200",
  restricted: "bg-red-100 text-red-700 border-red-200",
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

function ClassificationBadge({ c }: { c: Classification }) {
  return (
    <div className="flex flex-wrap items-center gap-2 mt-2">
      <span className="flex items-center gap-1 text-xs font-medium text-gray-700">
        <span>{TYPE_ICONS[c.document_type] ?? "📄"}</span>
        <span className="capitalize">{c.document_type.replace("_", " ")}</span>
      </span>
      <span
        className={`text-xs px-2 py-0.5 rounded-full font-medium border capitalize ${
          SENSITIVITY_COLORS[c.sensitivity_level] ?? "bg-gray-100 text-gray-600"
        }`}
      >
        {c.sensitivity_level}
      </span>
      {c.primary_topic && (
        <span className="text-xs text-gray-500 italic truncate max-w-xs">{c.primary_topic}</span>
      )}
    </div>
  );
}

function JobRow({ job }: { job: TrackedJob }) {
  const s = job.status;
  const isDone = s?.status === "complete";
  const isError = s?.status === "error";
  const progress = s?.progress ?? 0;

  return (
    <div className={`p-4 rounded-2xl border transition-all ${isDone ? "border-green-200 bg-green-50" : isError ? "border-red-200 bg-red-50" : "border-gray-200 bg-white"}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-800 truncate">{job.filename}</p>
          {s && (
            <p className={`text-xs mt-0.5 ${isError ? "text-red-600" : isDone ? "text-green-600" : "text-gray-400"}`}>
              {isError ? (s.error ?? "Processing failed") : STAGE_LABELS[s.stage] ?? s.stage}
            </p>
          )}
          {isDone && s?.classification && <ClassificationBadge c={s.classification} />}
        </div>

        <div className="shrink-0">
          {!s && <Loader2 className="w-5 h-5 text-gray-300 animate-spin" />}
          {s && isDone && <CheckCircle2 className="w-5 h-5 text-green-500" />}
          {s && isError && <AlertCircle className="w-5 h-5 text-red-500" />}
          {s && !isDone && !isError && <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />}
        </div>
      </div>

      {/* Progress bar */}
      {s && !isError && (
        <div className="mt-3 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${isDone ? "bg-green-400" : "bg-indigo-500"}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Stage pills */}
      {s && !isDone && !isError && (
        <div className="flex gap-1.5 mt-2.5">
          {["parsing", "classifying", "indexing", "complete"].map((stage) => {
            const stages = ["parsing", "classifying", "indexing", "complete"];
            const currentIdx = stages.indexOf(s.stage);
            const stageIdx = stages.indexOf(stage);
            const active = stageIdx === currentIdx;
            const done = stageIdx < currentIdx;
            return (
              <span
                key={stage}
                className={`text-[10px] px-2 py-0.5 rounded-full font-medium capitalize transition-colors ${
                  active ? "bg-indigo-100 text-indigo-700" : done ? "bg-green-100 text-green-600" : "bg-gray-100 text-gray-400"
                }`}
              >
                {stage}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<TrackedJob[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Poll active jobs — use a ref so the interval never goes stale
  const jobsRef = useRef<TrackedJob[]>([]);
  jobsRef.current = jobs;

  useEffect(() => {
    const tick = async () => {
      const active = jobsRef.current.filter(
        (j) => !j.status || (j.status.status !== "complete" && j.status.status !== "error")
      );
      if (!active.length) return;

      await Promise.all(
        active.map(async (j) => {
          try {
            const status = await getStatus(j.doc_id);
            setJobs((prev) =>
              prev.map((jj) => (jj.doc_id === j.doc_id ? { ...jj, status, polling: false } : jj))
            );
          } catch {
            /* ignore transient poll errors */
          }
        })
      );
    };

    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  // Run once on mount — tick reads live state via ref, no deps needed
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setUploadError(null);

    try {
      const response = await uploadFiles(files);
      const newJobs: TrackedJob[] = response.jobs.map((j) => ({
        ...j,
        status: null,
        polling: false,
      }));
      setJobs((prev) => [...prev, ...newJobs]);
      setFiles([]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const allDone = jobs.length > 0 && jobs.every((j) => j.status?.status === "complete" || j.status?.status === "error");
  const completedCount = jobs.filter((j) => j.status?.status === "complete").length;
  const totalPages = jobs.reduce((s, j) => s + (j.status?.page_count ?? 0), 0);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-5 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Chat
          </Link>
          <div className="flex items-center gap-2.5 ml-2">
            <div className="w-7 h-7 rounded-lg bg-linear-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <Brain className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="text-base font-bold text-gray-900">Upload Documents</span>
          </div>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-5 py-8 space-y-6">
        {/* Drop zone */}
        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Select files to upload</h2>
          <UploadZone files={files} onChange={setFiles} />

          {uploadError && (
            <div className="flex items-center gap-2 mt-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {uploadError}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            className="mt-5 w-full py-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white font-semibold rounded-xl transition-all duration-200 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {uploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Uploading…
              </>
            ) : (
              `Upload ${files.length > 0 ? files.length + " " : ""}File${files.length !== 1 ? "s" : ""}`
            )}
          </button>
        </div>

        {/* Job list */}
        {jobs.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-gray-700">Processing queue</h2>
            {jobs.map((job) => (
              <JobRow key={job.doc_id} job={job} />
            ))}
          </div>
        )}

        {/* Summary + Go to Chat */}
        {allDone && (
          <div className="bg-white rounded-2xl border border-green-200 p-5 shadow-sm text-center space-y-3">
            <p className="text-sm font-semibold text-gray-800">
              ✅ {completedCount} document{completedCount !== 1 ? "s" : ""} indexed
              {totalPages > 0 && ` · ${totalPages} page${totalPages !== 1 ? "s" : ""} processed`}
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition-colors"
            >
              <MessageSquare className="w-4 h-4" />
              Go to Chat
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
