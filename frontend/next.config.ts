import type { NextConfig } from "next";

// Server-side only. /api/* is proxied by middleware.ts (which adds the client IP
// and proxy secret); this rewrite only covers /health. Read at build time.
const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  experimental: {
    // Without R2 the browser PUTs PDFs through the /api proxy, which otherwise
    // silently truncates bodies over 10 MB. Keep at or above MAX_UPLOAD_MB.
    // (With R2 the file goes straight to the bucket and never passes through here.)
    proxyClientMaxBodySize: "51mb",
  },
  async rewrites() {
    return [
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
