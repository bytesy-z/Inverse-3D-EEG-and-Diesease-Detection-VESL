import { type NextRequest, NextResponse } from "next/server"
import { PHYSDEEPSIF_BACKEND_URL } from "@/lib/job-store"

export const dynamic = 'force-dynamic'

/**
 * GET /api/test-samples?mode=epileptogenic&limit=20
 *
 * Proxy to the Python backend's /api/test-samples endpoint.
 * Returns available synthetic test sample indices for the demo UI.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const mode = searchParams.get("mode") || "epileptogenic"
  const limit = searchParams.get("limit") || "20"

  try {
    const backendUrl = `${PHYSDEEPSIF_BACKEND_URL}/api/test-samples?mode=${mode}&limit=${limit}`
    const response = await fetch(backendUrl)

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Backend error" }))
      return NextResponse.json(
        { message: errorData.detail || "Failed to fetch test samples" },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error"
    if (msg.includes("ECONNREFUSED") || msg.includes("fetch failed")) {
      return NextResponse.json(
        { message: "PhysDeepSIF backend not running. Start with: cd backend && python server.py" },
        { status: 503 }
      )
    }
    return NextResponse.json({ message: msg }, { status: 500 })
  }
}
