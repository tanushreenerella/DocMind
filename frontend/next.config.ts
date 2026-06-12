import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
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
