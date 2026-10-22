import { ApiError, type JobStatus } from "./api";

export const STATUS_POLL_INTERVAL_MS = 2500;

const TERMINAL: ReadonlySet<JobStatus["status"]> = new Set(["complete", "partial", "error"]);

export function isTerminal(status: JobStatus | null | undefined): boolean {
  return !!status && TERMINAL.has(status.status);
}

export interface StatusPoller {
  /** Starts polling `docId`; a no-op if it is already being polled. */
  start(docId: string): void;
  stop(docId: string): void;
  stopAll(): void;
  isPolling(docId: string): boolean;
}

interface Options {
  fetchStatus: (docId: string) => Promise<JobStatus>;
  onStatus: (docId: string, status: JobStatus) => void;
  /** The server no longer knows the document (404). Polling has stopped. */
  onGone?: (docId: string) => void;
  intervalMs?: number;
}

/**
 * At most one poller per document. Each polls immediately, then waits
 * `intervalMs` after every response (so requests never overlap), and stops by
 * itself once the status is terminal. Transient errors just wait for the next tick.
 */
export function createStatusPoller({
  fetchStatus,
  onStatus,
  onGone,
  intervalMs = STATUS_POLL_INTERVAL_MS,
}: Options): StatusPoller {
  // Present = active; the value is the pending timer, if any.
  const active = new Map<string, ReturnType<typeof setTimeout> | null>();

  const poll = async (docId: string) => {
    let status: JobStatus | null = null;
    try {
      status = await fetchStatus(docId);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404 && active.has(docId)) {
        active.delete(docId);
        onGone?.(docId);
        return;
      }
    }
    if (!active.has(docId)) return; // stopped while the request was in flight
    if (status) {
      onStatus(docId, status);
      if (isTerminal(status)) {
        active.delete(docId);
        return;
      }
    }
    active.set(docId, setTimeout(() => void poll(docId), intervalMs));
  };

  const stop = (docId: string) => {
    const timer = active.get(docId);
    if (timer) clearTimeout(timer);
    active.delete(docId);
  };

  return {
    start(docId) {
      if (active.has(docId)) return;
      active.set(docId, null);
      void poll(docId);
    },
    stop,
    stopAll() {
      for (const docId of [...active.keys()]) stop(docId);
    },
    isPolling: (docId) => active.has(docId),
  };
}
