"use client"

import Link from "next/link"
import { Construction } from "lucide-react"

export default function EEGSlidingWindowPage() {
  return (
    <main className="min-h-screen w-full flex flex-col bg-[#1f2937] text-foreground">
      {/* Navigation Bar */}
      <nav className="w-full bg-[#181c1f] border-b border-gray-800 shadow flex items-center px-6 py-3 z-20 sticky top-0">
        <div className="flex-1 flex items-center">
          <Link href="/" className="text-2xl font-bold tracking-tight text-[var(--chart-4)]">VESL</Link>
        </div>
        <div className="flex items-center gap-6">
          <Link href="/" className="font-semibold px-4 py-2 rounded hover:bg-gray-800 transition text-white">Home</Link>
          <Link href="/eeg-source-localization" className="font-semibold px-4 py-2 rounded hover:bg-gray-800 transition text-white">EEG Source Localization</Link>
          <span className="font-semibold px-4 py-2 rounded bg-gray-800 text-[var(--chart-4)]">EEG Sliding Window</span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center px-4">
        <div className="text-center max-w-2xl">
          <div className="mb-8 flex justify-center">
            <div className="p-6 bg-[#181c1f] rounded-full border border-gray-700">
              <Construction className="w-16 h-16 text-[var(--chart-4)]" />
            </div>
          </div>
          
          <h1 className="text-4xl font-bold mb-4 text-white">EEG Sliding Window Analysis</h1>
          <p className="text-xl text-gray-400 mb-8">
            This feature is currently under development
          </p>
          
          <div className="bg-[#181c1f] border border-gray-700 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold text-[var(--chart-4)] mb-3">Coming Soon</h2>
            <ul className="text-left text-gray-300 space-y-2">
              <li>• Real-time sliding window EEG analysis</li>
              <li>• Continuous source localization over time</li>
              <li>• Interactive timeline visualization</li>
              <li>• Window-by-window activity comparison</li>
            </ul>
          </div>
          
          <Link 
            href="/"
            className="inline-block font-semibold py-3 px-8 rounded-lg shadow-lg transition-all duration-200"
            style={{
              background: 'var(--chart-4)',
              color: 'var(--background)',
              border: '2px solid var(--chart-4)',
            }}
          >
            Return to Home
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="w-full bg-[#181c1f] border-t border-gray-800 py-4 flex flex-col items-center">
        <div className="text-gray-500 text-sm">
          &copy; {new Date().getFullYear()} VESL Project. All rights reserved.
        </div>
      </footer>
    </main>
  )
}
