import type { NextConfig } from "next";

// Server-side only: the browser talks to this app, which proxies to the backend,
// so the auth cookie stays first-party. Read at build time for rewrites.
const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` },
      // Used by warmupBackend() to wake a cold Render instance.
      { source: "/health", destination: `${BACKEND_URL}/health` },
    ];
  },
  images: {
    // Allow loading page images from the FastAPI backend
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/api/page-image/**",
      },
      {
        protocol: "https",
        hostname: "**",
        pathname: "/api/page-image/**",
      },
    ],
  },
};

export default nextConfig;
