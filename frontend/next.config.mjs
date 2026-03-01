/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    // Increase API route body size limit for large EDF files
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  async rewrites() {
    return [
      {
        source: '/results/:path*',
        destination: 'http://localhost:3000/api/serve-result/:path*',
      },
    ]
  },
}

export default nextConfig
