import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  RATE_LIMIT_MESSAGE,
  conversationHistory,
  deleteDocument,
  listDocuments,
  login,
  userFacingError,
} from "@/lib/api";
import { getUser, saveUser } from "@/lib/auth";

function respondWith(status: number, body: unknown) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status })));
}

describe("api error handling", () => {
  beforeEach(() => {
    saveUser({ user_id: "u1", email: "u1@example.com", full_name: "" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("turns a 429 into the rate-limit message", async () => {
    respondWith(429, { detail: "Too many requests, try again shortly" });

    const err = await listDocuments().catch((e: unknown) => e);

    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(429);
    expect(userFacingError(err, "fallback")).toBe(RATE_LIMIT_MESSAGE);
    expect(getUser()).not.toBeNull();
  });

  it("keeps the server's message for a daily-budget 429", async () => {
    const budget = "Daily AI usage limit reached, resets at midnight UTC";
    respondWith(429, { detail: budget });

    const err = await listDocuments().catch((e: unknown) => e);

    expect((err as ApiError).status).toBe(429);
    expect(userFacingError(err, "fallback")).toBe(budget);
  });

  it("uses the rate-limit message for a 429 without a usable detail", async () => {
    for (const body of [{}, { detail: "" }, { detail: [{ msg: "x" }] }]) {
      respondWith(429, body);
      const err = await listDocuments().catch((e: unknown) => e);
      expect(userFacingError(err, "fallback")).toBe(RATE_LIMIT_MESSAGE);
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not json", { status: 429 })));
    const err = await listDocuments().catch((e: unknown) => e);
    expect(userFacingError(err, "fallback")).toBe(RATE_LIMIT_MESSAGE);
  });

  it("logs out locally on a 401 from a protected endpoint", async () => {
    respondWith(401, { detail: "Invalid or expired token" });

    await expect(listDocuments()).rejects.toMatchObject({ status: 401 });
    expect(getUser()).toBeNull();
  });

  it("keeps the login error on the page for bad credentials", async () => {
    respondWith(401, { detail: "Invalid email or password" });

    await expect(login("u1@example.com", "wrong")).rejects.toThrow("Invalid email or password");
    expect(getUser()).not.toBeNull();
  });

  it("uses the fallback for non-rate-limit errors", () => {
    expect(userFacingError(new ApiError("boom", 500), "fallback")).toBe("fallback");
    expect(userFacingError(new Error("network"), "fallback")).toBe("fallback");
  });
});

describe("deleteDocument", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends DELETE for the encoded doc id and resolves on 204", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(deleteDocument("doc/1")).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledWith("/api/documents/doc%2F1", { method: "DELETE" });
  });

  it("surfaces the 409 still-processing detail", async () => {
    const detail = "Document is still processing; try again when it finishes";
    respondWith(409, { detail });

    const err = await deleteDocument("doc-1").catch((e: unknown) => e);

    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
    expect(userFacingError(err, "fallback", [409])).toBe(detail);
    // Without opting in, a 409's detail is not shown.
    expect(userFacingError(err, "fallback")).toBe("fallback");
  });

  it("rejects with 404 for a document that is not there", async () => {
    respondWith(404, { detail: "Document not found" });

    await expect(deleteDocument("doc-1")).rejects.toMatchObject({ status: 404 });
  });
});

describe("conversationHistory", () => {
  it("drops error notices and the questions they answered", () => {
    expect(
      conversationHistory([
        { role: "user", content: "q1" },
        { role: "assistant", content: "a1" },
        { role: "user", content: "q2" },
        { role: "assistant", content: "AI service busy", isError: true },
        { role: "user", content: "q3" },
        { role: "assistant", content: "a3" },
      ])
    ).toEqual([
      { role: "user", content: "q1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "q3" },
      { role: "assistant", content: "a3" },
    ]);
  });

  it("keeps everything when nothing failed, without extra fields", () => {
    const messages = [
      { role: "user" as const, content: "q1", isError: false },
      { role: "assistant" as const, content: "a1" },
    ];
    expect(conversationHistory(messages)).toEqual([
      { role: "user", content: "q1" },
      { role: "assistant", content: "a1" },
    ]);
  });
});
