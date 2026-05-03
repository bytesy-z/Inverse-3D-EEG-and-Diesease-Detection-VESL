/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: process.env.NODE_ENV === 'development',
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
    const backendUrl = process.env.PHYSDEEPSIF_BACKEND_URL || 'http://localhost:8000'
    return [
      {
        source: '/api/analyze',
        destination: `${backendUrl}/api/analyze`,
      },
      {
        source: '/api/results/:path*',
        destination: `${backendUrl}/api/results/:path*`,
      },
      {
        source: '/api/job/:path*',
        destination: `${backendUrl}/api/job/:path*`,
      },
      {
        source: '/api/biomarkers',
        destination: `${backendUrl}/api/biomarkers`,
      },
      {
        source: '/api/eeg_waveform',
        destination: `${backendUrl}/api/eeg_waveform`,
      },
      {
        source: '/results/:path*',
        destination: `${backendUrl}/api/results/:path*`,
      },
    ]
  },
  async headers() {
    return [
      {
        source: '/ws/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
        ],
      },
    ]
  },
}

export default nextConfig
