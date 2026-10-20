// @vitest-environment node
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const ipAddress = vi.hoisted(() => vi.fn<(req: Request) => string | undefined>());
vi.mock("@vercel/functions", () => ({ ipAddress }));

import { config, middleware } from "./middleware";

// Headers the rewritten request will carry to the backend, as Next encodes them.
function forwardedHeader(response: Response, name: string): string | null {
  return response.headers.get(`x-middleware-request-${name}`);
}

function request(path: string, headers: Record<string, string> = {}) {
  return new NextRequest(`https://app.example${path}`, { headers });
}

describe("middleware", () => {
  beforeEach(() => {
    vi.stubEnv("PROXY_SECRET", "s3cret");
    vi.stubEnv("BACKEND_URL", "https://backend.example/");
    ipAddress.mockReset();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("only runs for /api", () => {
    expect(config.matcher).toBe("/api/:path*");
  });

  it("rewrites to BACKEND_URL keeping path and query", () => {
    const response = middleware(request("/api/documents?page=2"));

    expect(response.headers.get("x-middleware-rewrite")).toBe(
      "https://backend.example/api/documents?page=2",
    );
  });

  it("sets x-client-ip from ipAddress() and adds the proxy secret", () => {
    ipAddress.mockReturnValue("203.0.113.7");

    const response = middleware(request("/api/chat", { "x-real-ip": "198.51.100.9" }));

    expect(forwardedHeader(response, "x-client-ip")).toBe("203.0.113.7");
    expect(forwardedHeader(response, "x-proxy-secret")).toBe("s3cret");
  });

  it("falls back to x-real-ip", () => {
    ipAddress.mockReturnValue(undefined);

    const response = middleware(request("/api/chat", { "x-real-ip": "198.51.100.9" }));

    expect(forwardedHeader(response, "x-client-ip")).toBe("198.51.100.9");
  });

  it("overwrites client-supplied x-client-ip and x-proxy-secret", () => {
    ipAddress.mockReturnValue("203.0.113.7");

    const response = middleware(
      request("/api/chat", { "x-client-ip": "10.6.6.6", "x-proxy-secret": "guess" }),
    );

    expect(forwardedHeader(response, "x-client-ip")).toBe("203.0.113.7");
    expect(forwardedHeader(response, "x-proxy-secret")).toBe("s3cret");
  });

  it("drops a client-supplied x-client-ip when no IP is known", () => {
    ipAddress.mockReturnValue(undefined);

    const response = middleware(request("/api/chat", { "x-client-ip": "10.6.6.6" }));

    expect(forwardedHeader(response, "x-client-ip")).toBeNull();
    const overridden = response.headers.get("x-middleware-override-headers")?.split(",");
    expect(overridden).not.toContain("x-client-ip");
  });

  it("refuses to proxy without PROXY_SECRET", async () => {
    vi.stubEnv("PROXY_SECRET", "");

    const response = middleware(request("/api/chat"));

    expect(response.status).toBe(500);
    expect(response.headers.get("x-middleware-rewrite")).toBeNull();
    expect(await response.json()).toEqual({ detail: "Proxy misconfigured" });
  });
});
