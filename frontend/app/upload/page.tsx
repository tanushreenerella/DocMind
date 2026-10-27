"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Brain, CheckCircle2, AlertCircle, AlertTriangle, Loader2, MessageSquare } from "lucide-react";
import UploadZone from "@/components/UploadZone";
import SuspiciousBadge from "@/components/SuspiciousBadge";
import AuthGuard from "@/components/AuthGuard";
import { uploadFile, uploadErrorMessage, getStatus, deleteDocument, warmupBackend } from "@/lib/api";
import type { UploadJob, JobStatus, Classification } from "@/lib/api";
import { createStatusPoller, isTerminal } from "@/lib/statusPoller";

interface TrackedJob extends UploadJob {
  status: JobStatus | null;
  /** Kept so a failed document can be uploaded again. */
  file: File;
  retrying: boolean;
  /** Why the last "try again" upload itself failed. */
  retryError: string | null;
}

function track(job: UploadJob, file: File): TrackedJob {
  return { ...job, status: null, file, retrying: false, retryError: null };
}

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  parsing: "Parsing document…",
  classifying: "Classifying…",
  indexing: "Indexing embeddings…",
  complete: "Complete",
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

/** A status for documents the server lost track of (404 while polling). */
function failedStatus(job: TrackedJob): JobStatus {
  return {
    status: "error",
    filename: job.filename,
    stage: "error",
    progress: 0,
    classification: null,
    page_count: 0,
    failed_pages: [],
    suspicious: false,
    hidden_text_pages: [],
    hidden_text_removed: 0,
  };
}

function pageList(pages: number[]): string {
  return `page${pages.length !== 1 ? "s" : ""} ${pages.join(", ")}`;
}

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

function JobRow({ job, onRetry }: { job: TrackedJob; onRetry: (job: TrackedJob) => void }) {
  const s = job.status;
  const isDone = s?.status === "complete";
  const isPartial = s?.status === "partial";
  const isError = s?.status === "error";
  const isFinished = isDone || isPartial || isError;
  const progress = s?.progress ?? 0;

  const tone = isDone
    ? "border-green-200 bg-green-50"
    : isPartial
      ? "border-amber-200 bg-amber-50"
      : isError
        ? "border-red-200 bg-red-50"
        : "border-gray-200 bg-white";

  return (
    <div className={`p-4 rounded-2xl border transition-all ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-800 truncate">{job.filename}</p>
          {isError ? (
            // The server logs the real reason; users only need to know what to do.
            <p className="text-xs mt-0.5 text-red-600">
              {job.retrying ? (
                "Retrying…"
              ) : (
                <>
                  Failed —{" "}
                  <button
                    type="button"
                    onClick={() => onRetry(job)}
                    className="font-semibold underline underline-offset-2 hover:text-red-700"
                  >
                    try again
                  </button>
                </>
              )}
              {job.retryError && !job.retrying && <span className="block">{job.retryError}</span>}
            </p>
          ) : isPartial ? (
            <p className="text-xs mt-0.5 text-amber-700">
              Indexed, but couldn&apos;t read {pageList(s.failed_pages)}
            </p>
          ) : (
            s && (
              <p className={`text-xs mt-0.5 ${isDone ? "text-green-600" : "text-gray-400"}`}>
                {STAGE_LABELS[s.stage] ?? s.stage}
              </p>
            )
          )}
          {(isDone || isPartial) && s?.classification && <ClassificationBadge c={s.classification} />}
          {(isDone || isPartial) && s?.suspicious && (
            <div className="mt-2">
              <SuspiciousBadge
                detail={`${s.hidden_text_removed} characters removed from ${pageList(s.hidden_text_pages)}.`}
              />
            </div>
          )}
        </div>

        <div className="shrink-0">
          {(!s || job.retrying) && <Loader2 className="w-5 h-5 text-gray-300 animate-spin" />}
          {isDone && <CheckCircle2 className="w-5 h-5 text-green-500" aria-label="Indexed" />}
          {isPartial && <AlertTriangle className="w-5 h-5 text-amber-500" aria-label="Partly indexed" />}
          {isError && !job.retrying && <AlertCircle className="w-5 h-5 text-red-500" aria-label="Failed" />}
          {s && !isFinished && <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />}
        </div>
      </div>

      {/* Progress bar */}
      {s && !isError && (
        <div className="mt-3 h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isDone ? "bg-green-400" : isPartial ? "bg-amber-400" : "bg-indigo-500"
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Stage pills */}
      {s && !isFinished && (
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

function Summary({ jobs }: { jobs: TrackedJob[] }) {
  const indexed = jobs.filter((j) => j.status?.status === "complete" || j.status?.status === "partial");
  const failedCount = jobs.filter((j) => j.status?.status === "error").length;
  const partialCount = indexed.filter((j) => j.status?.status === "partial").length;
  const suspiciousCount = indexed.filter((j) => j.status?.suspicious).length;
  const totalPages = indexed.reduce((sum, j) => sum + (j.status?.page_count ?? 0), 0);

  if (indexed.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-red-200 p-5 shadow-sm text-center">
        <p className="text-sm font-semibold text-red-700">
          No documents were indexed. Use “try again” on the files above.
        </p>
      </div>
    );
  }

  const allClean = failedCount === 0 && partialCount === 0 && suspiciousCount === 0;
  return (
    <div
      className={`bg-white rounded-2xl border p-5 shadow-sm text-center space-y-3 ${
        allClean ? "border-green-200" : "border-amber-200"
      }`}
    >
      <p className="text-sm font-semibold text-gray-800">
        {allClean ? "✅" : "⚠️"}{" "}
        {failedCount > 0 ? `${indexed.length} of ${jobs.length}` : indexed.length} document
        {(failedCount > 0 ? jobs.length : indexed.length) !== 1 ? "s" : ""} indexed
        {totalPages > 0 && ` · ${totalPages} page${totalPages !== 1 ? "s" : ""} processed`}
      </p>
      {partialCount > 0 && (
        <p className="text-xs text-amber-700">
          {partialCount} document{partialCount !== 1 ? "s are" : " is"} missing text from some pages.
        </p>
      )}
      {suspiciousCount > 0 && (
        <p className="text-xs text-red-700">
          {suspiciousCount} document{suspiciousCount !== 1 ? "s" : ""} contained hidden text, which was not indexed.
        </p>
      )}
      <Link
        href="/"
        className="inline-flex items-center gap-2 bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition-colors"
      >
        <MessageSquare className="w-4 h-4" />
        Go to Chat
      </Link>
    </div>
  );
}

export default function UploadPage() {
  return (
    <AuthGuard>
      <UploadContent />
    </AuthGuard>
  );
}

function UploadContent() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<TrackedJob[]>([]);
  const [uploading, setUploading] = useState(false);
  /** One "file: reason" line per file that could not be uploaded. */
  const [uploadErrors, setUploadErrors] = useState<string[]>([]);
  const [serverReady, setServerReady] = useState(false);

  // One poller for the page; it keeps at most one loop per document.
  const [poller] = useState(() =>
    createStatusPoller({
      fetchStatus: getStatus,
      onStatus: (docId, status) =>
        setJobs((prev) => prev.map((j) => (j.doc_id === docId ? { ...j, status } : j))),
      onGone: (docId) =>
        setJobs((prev) => prev.map((j) => (j.doc_id === docId ? { ...j, status: failedStatus(j) } : j))),
    })
  );

  // Ping /health on mount so Render's free-tier instance wakes up
  // before the user tries to upload (cold start can take ~50 s).
  useEffect(() => {
    warmupBackend().then((ok) => setServerReady(ok));
  }, []);

  // Start a poller for every unfinished job. start() ignores documents that
  // are already polled, so re-renders never add duplicates.
  useEffect(() => {
    for (const job of jobs) {
      if (!isTerminal(job.status)) poller.start(job.doc_id);
    }
  }, [jobs, poller]);

  useEffect(() => () => poller.stopAll(), [poller]);

  const handleUpload = async () => {
    if (!files.length) return;
    const sent = files;
    setUploading(true);
    setUploadErrors([]);

    // Each file uploads on its own; one bad file doesn't stop the others.
    const results = await Promise.allSettled(sent.map((f) => uploadFile(f)));
    const started: TrackedJob[] = [];
    const failed: File[] = [];
    const errors: string[] = [];
    results.forEach((result, i) => {
      if (result.status === "fulfilled") {
        started.push(track(result.value, sent[i]));
      } else {
        failed.push(sent[i]);
        errors.push(`${sent[i].name}: ${uploadErrorMessage(result.reason)}`);
      }
    });

    setJobs((prev) => [...prev, ...started]);
    // Failed files stay selected, so they can be retried or removed.
    setFiles(failed);
    setUploadErrors(errors);
    setUploading(false);
  };

  const handleRetry = async (job: TrackedJob) => {
    setJobs((prev) =>
      prev.map((j) => (j.doc_id === job.doc_id ? { ...j, retrying: true, retryError: null } : j))
    );
    try {
      const replacement = track(await uploadFile(job.file), job.file);
      setJobs((prev) => prev.map((j) => (j.doc_id === job.doc_id ? replacement : j)));
      // The failed attempt is never listed anywhere; drop its leftovers. Best effort.
      deleteDocument(job.doc_id).catch(() => {});
    } catch (err) {
      const retryError = uploadErrorMessage(err);
      setJobs((prev) =>
        prev.map((j) => (j.doc_id === job.doc_id ? { ...j, retrying: false, retryError } : j))
      );
    }
  };

  const allDone = jobs.length > 0 && jobs.every((j) => isTerminal(j.status) && !j.retrying);

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

      {/* Cold-start banner — disappears once /health responds OK */}
      {!serverReady && (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 flex items-center justify-center gap-2 text-sm text-amber-700">
          <Loader2 className="w-4 h-4 animate-spin shrink-0" />
          <span>Server starting up (free tier) — please wait ~30 s before uploading.</span>
        </div>
      )}

      <main className="max-w-2xl mx-auto px-5 py-8 space-y-6">
        {/* Drop zone */}
        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Select files to upload</h2>
          <UploadZone files={files} onChange={setFiles} />

          {uploadErrors.length > 0 && (
            <div
              role="alert"
              className="flex items-start gap-2 mt-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5"
            >
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <ul className="space-y-1">
                {uploadErrors.map((message) => (
                  <li key={message}>{message}</li>
                ))}
              </ul>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!serverReady || files.length === 0 || uploading}
            className="mt-5 w-full py-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white font-semibold rounded-xl transition-all duration-200 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {!serverReady ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Waiting for server…
              </>
            ) : uploading ? (
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
              <JobRow key={job.doc_id} job={job} onRetry={handleRetry} />
            ))}
          </div>
        )}

        {/* Summary + Go to Chat */}
        {allDone && <Summary jobs={jobs} />}
      </main>
    </div>
  );
}
