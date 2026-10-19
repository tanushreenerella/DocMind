import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, RATE_LIMIT_MESSAGE, listDocuments, login, userFacingError } from "@/lib/api";
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
