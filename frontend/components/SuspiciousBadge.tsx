import { ShieldAlert } from "lucide-react";

const EXPLANATION =
  "This PDF contained text a reader can't see (white, invisible or tiny), a common way to " +
  "hide instructions for AI. It was left out of the index; answers use only the visible text.";

/** Warning chip for a document whose hidden text was stripped at upload. */
export default function SuspiciousBadge({ detail }: { detail?: string }) {
  return (
    <span
      title={detail ? `${EXPLANATION} ${detail}` : EXPLANATION}
      className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded-full font-medium border bg-red-50 text-red-700 border-red-200 shrink-0"
    >
      <ShieldAlert className="w-3 h-3" aria-hidden="true" />
      Hidden text removed
    </span>
  );
}
