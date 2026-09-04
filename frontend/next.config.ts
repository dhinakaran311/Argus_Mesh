import type { NextConfig } from 'next'

// BACKEND_URL must be set to the deployed backend URL in production
// (e.g. https://your-backend.onrender.com). Falls back to localhost for local dev.
const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000'

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
