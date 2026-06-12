"use client";

import { useState, useCallback, useRef } from "react";
import { Upload, File, X, AlertCircle } from "lucide-react";

const ACCEPTED = ".pdf,.png,.jpg,.jpeg,.tiff";
const ACCEPTED_TYPES = new Set(["application/pdf", "image/png", "image/jpeg", "image/tiff"]);
const MAX_MB = 20;

interface UploadZoneProps {
  files: File[];
  onChange: (files: File[]) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileIcon(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "pdf") return "📄";
  if (["png", "jpg", "jpeg", "tiff"].includes(ext)) return "🖼️";
  return "📁";
}

export default function UploadZone({ files, onChange }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = (incoming: File[]): { valid: File[]; errors: string[] } => {
    const valid: File[] = [];
    const errors: string[] = [];
    for (const f of incoming) {
      const ext = "." + (f.name.split(".").pop() ?? "").toLowerCase();
      const acceptedExts = new Set([".pdf", ".png", ".jpg", ".jpeg", ".tiff"]);
      if (!acceptedExts.has(ext) && !ACCEPTED_TYPES.has(f.type)) {
        errors.push(`${f.name}: unsupported type`);
        continue;
      }
      if (f.size > MAX_MB * 1024 * 1024) {
        errors.push(`${f.name}: exceeds ${MAX_MB} MB limit`);
        continue;
      }
      valid.push(f);
    }
    return { valid, errors };
  };

  const addFiles = useCallback(
    (incoming: File[]) => {
      setError(null);
      const { valid, errors } = validate(incoming);
      if (errors.length) setError(errors.join(" · "));
      const merged = [...files, ...valid.filter((f) => !files.some((e) => e.name === f.name))];
      onChange(merged);
    },
    [files, onChange]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(Array.from(e.target.files));
    e.target.value = "";
  };

  const removeFile = (idx: number) => {
    const next = [...files];
    next.splice(idx, 1);
    onChange(next);
  };

  const totalSize = files.reduce((s, f) => s + f.size, 0);

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative flex flex-col items-center justify-center gap-3
          border-2 border-dashed rounded-2xl p-10 cursor-pointer
          transition-all duration-200 select-none
          ${dragging
            ? "border-indigo-400 bg-indigo-50 scale-[1.01]"
            : "border-gray-300 bg-gray-50 hover:border-indigo-300 hover:bg-indigo-50/40"
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={onInputChange}
        />
        <div className={`p-4 rounded-full transition-colors ${dragging ? "bg-indigo-100" : "bg-white border border-gray-200"}`}>
          <Upload className={`w-8 h-8 ${dragging ? "text-indigo-500" : "text-gray-400"}`} />
        </div>
        <div className="text-center">
          <p className="text-base font-semibold text-gray-700">
            Drop PDFs or images here, or <span className="text-indigo-600">click to browse</span>
          </p>
          <p className="text-sm text-gray-400 mt-1">
            PDF, PNG, JPG, JPEG, TIFF — max {MAX_MB} MB each
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* File queue */}
      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-500 px-1">
            <span>{files.length} file{files.length !== 1 ? "s" : ""} selected</span>
            <span>{formatSize(totalSize)} total</span>
          </div>
          {files.map((file, idx) => (
            <div
              key={`${file.name}-${idx}`}
              className="flex items-center gap-3 px-4 py-3 bg-white border border-gray-200 rounded-xl"
            >
              <span className="text-xl">{fileIcon(file.name)}</span>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">{file.name}</p>
                <p className="text-xs text-gray-400">{formatSize(file.size)}</p>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                className="p-1 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-red-500 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
