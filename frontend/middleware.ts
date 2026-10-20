import { ipAddress } from "@vercel/functions";
import { NextResponse, type NextRequest } from "next/server";

// Proxies /api/* to the backend so the auth cookie stays first-party. The
// backend only accepts requests carrying PROXY_SECRET, and trusts x-client-ip
// for rate limiting because of it, so both headers are always overwritten
// here: anything the browser sent under those names is discarded.
export function middleware(request: NextRequest) {
  const secret = process.env.PROXY_SECRET;
  if (!secret) {
    console.error("[proxy] PROXY_SECRET is not set; refusing to proxy /api");
    return NextResponse.json({ detail: "Proxy misconfigured" }, { status: 500 });
  }

  const backendUrl = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");
  const target = new URL(request.nextUrl.pathname + request.nextUrl.search, backendUrl);

  const headers = new Headers(request.headers);
  const clientIp = ipAddress(request) ?? request.headers.get("x-real-ip");
  if (clientIp) {
    headers.set("x-client-ip", clientIp);
  } else {
    headers.delete("x-client-ip");
  }
  headers.set("x-proxy-secret", secret);

  return NextResponse.rewrite(target, { request: { headers } });
}

export const config = {
  matcher: "/api/:path*",
};
