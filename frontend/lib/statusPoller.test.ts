import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, type JobStatus } from "./api";
import { STATUS_POLL_INTERVAL_MS, createStatusPoller, isTerminal } from "./statusPoller";

function status(s: JobStatus["status"]): JobStatus {
  return {
    status: s,
    filename: "a.pdf",
    stage: s,
    progress: 0,
    classification: null,
    page_count: 0,
    failed_pages: [],
  };
}

/** Let the in-flight fetch promise settle without advancing timers. */
const flush = () => vi.advanceTimersByTimeAsync(0);

describe("createStatusPoller", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("polls every 2.5s", () => {
    expect(STATUS_POLL_INTERVAL_MS).toBe(2500);
  });

  it("polls immediately, then every interval, and stops on a terminal status", async () => {
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(status("processing"))
      .mockResolvedValueOnce(status("processing"))
      .mockResolvedValueOnce(status("complete"));
    const onStatus = vi.fn();
    const poller = createStatusPoller({ fetchStatus, onStatus });

    poller.start("d1");
    await flush();
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(2499);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchStatus).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(2500);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(poller.isPolling("d1")).toBe(false);

    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(onStatus.mock.calls.map(([, s]) => s.status)).toEqual(["processing", "processing", "complete"]);
  });

  it.each(["partial", "error"] as const)("treats %s as done", async (terminal) => {
    const fetchStatus = vi.fn().mockResolvedValue(status(terminal));
    const poller = createStatusPoller({ fetchStatus, onStatus: vi.fn() });

    poller.start("d1");
    await vi.advanceTimersByTimeAsync(10_000);

    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });

  it("never runs two loops for the same document", async () => {
    const fetchStatus = vi.fn().mockResolvedValue(status("processing"));
    const poller = createStatusPoller({ fetchStatus, onStatus: vi.fn() });

    poller.start("d1");
    poller.start("d1"); // e.g. a re-render
    await flush();
    poller.start("d1");
    await vi.advanceTimersByTimeAsync(2500 * 4);

    // 1 immediate + 4 ticks, not multiples of that.
    expect(fetchStatus).toHaveBeenCalledTimes(5);
  });

  it("does not overlap requests when the server is slow", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const fetchStatus = vi.fn(async () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 6000)); // slower than the interval
      inFlight--;
      return status("processing");
    });
    const poller = createStatusPoller({ fetchStatus, onStatus: vi.fn() });

    poller.start("d1");
    await vi.advanceTimersByTimeAsync(30_000);

    expect(maxInFlight).toBe(1);
  });

  it("polls documents independently", async () => {
    const fetchStatus = vi.fn(async (docId: string) =>
      status(docId === "fast" ? "complete" : "processing")
    );
    const poller = createStatusPoller({ fetchStatus, onStatus: vi.fn() });

    poller.start("fast");
    poller.start("slow");
    await vi.advanceTimersByTimeAsync(5000);

    expect(fetchStatus.mock.calls.filter(([id]) => id === "fast")).toHaveLength(1);
    expect(fetchStatus.mock.calls.filter(([id]) => id === "slow")).toHaveLength(3);
  });

  it("keeps polling through transient errors", async () => {
    const fetchStatus = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("network"))
      .mockRejectedValueOnce(new ApiError("busy", 503))
      .mockResolvedValueOnce(status("complete"));
    const onStatus = vi.fn();
    const poller = createStatusPoller({ fetchStatus, onStatus });

    poller.start("d1");
    await vi.advanceTimersByTimeAsync(5000);

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(onStatus).toHaveBeenCalledOnce();
    expect(poller.isPolling("d1")).toBe(false);
  });

  it("stops and reports a document the server no longer knows", async () => {
    const fetchStatus = vi.fn().mockRejectedValue(new ApiError("Job not found", 404));
    const onGone = vi.fn();
    const poller = createStatusPoller({ fetchStatus, onStatus: vi.fn(), onGone });

    poller.start("d1");
    await vi.advanceTimersByTimeAsync(10_000);

    expect(fetchStatus).toHaveBeenCalledOnce();
    expect(onGone).toHaveBeenCalledWith("d1");
  });

  it("stopAll cancels pending ticks and drops in-flight results", async () => {
    let resolve!: (s: JobStatus) => void;
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(status("processing"))
      .mockImplementationOnce(() => new Promise<JobStatus>((r) => (resolve = r)));
    const onStatus = vi.fn();
    const poller = createStatusPoller({ fetchStatus, onStatus });

    poller.start("a");
    await vi.advanceTimersByTimeAsync(2500); // second request now in flight
    poller.stopAll();
    resolve(status("processing"));
    await vi.advanceTimersByTimeAsync(10_000);

    expect(fetchStatus).toHaveBeenCalledTimes(2);
    expect(onStatus).toHaveBeenCalledTimes(1);
    expect(poller.isPolling("a")).toBe(false);
  });
});

describe("isTerminal", () => {
  it("is true only for finished statuses", () => {
    expect(isTerminal(null)).toBe(false);
    expect(isTerminal(status("queued"))).toBe(false);
    expect(isTerminal(status("processing"))).toBe(false);
    expect(isTerminal(status("complete"))).toBe(true);
    expect(isTerminal(status("partial"))).toBe(true);
    expect(isTerminal(status("error"))).toBe(true);
  });
});
