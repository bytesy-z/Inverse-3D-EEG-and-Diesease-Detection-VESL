import { type NextRequest, NextResponse } from "next/server"
import { PHYSDEEPSIF_BACKEND_URL, physdeepsifJobs, type PhysDeepSIFResult } from "@/lib/job-store"

export const maxDuration = 300
export const dynamic = 'force-dynamic'

/**
 * POST /api/physdeepsif
 *
 * Proxy route that forwards analysis requests to the Python FastAPI backend.
 * Accepts either:
 *   - FormData with 'file' (EEG upload) and optional 'threshold_percentile'
 *   - FormData with 'sample_idx' (synthetic test sample) and optional 'threshold_percentile'
 *
 * The Python backend runs PhysDeepSIF inference, computes epileptogenicity
 * scores, and generates a 3D brain heatmap.  Results are returned synchronously
 * (inference takes ~5-15 seconds).
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now()
  console.log("[PHYSDEEPSIF API] Received request")

  try {
    const formData = await request.formData()

    // Build a new FormData to forward to the Python backend
    const backendForm = new FormData()

    const file = formData.get("file") as File | null
    const sampleIdx = formData.get("sample_idx") as string | null
    const thresholdPct = formData.get("threshold_percentile") as string | null
    const includeEeg = formData.get("include_eeg") as string | null

    if (file) {
      console.log(`[PHYSDEEPSIF API] Forwarding uploaded file: ${file.name} (${file.size} bytes)`)
      backendForm.append("file", file)
    }
    if (sampleIdx !== null && sampleIdx !== undefined) {
      console.log(`[PHYSDEEPSIF API] Using test sample idx: ${sampleIdx}`)
      backendForm.append("sample_idx", sampleIdx)
    }
    if (thresholdPct) {
      backendForm.append("threshold_percentile", thresholdPct)
    } else {
      backendForm.append("threshold_percentile", "87.5")
    }
    // Biomarkers page uses epileptogenicity detection mode
    backendForm.append("mode", "biomarkers")
    backendForm.append("include_eeg", includeEeg ?? "true")

    // Forward to Python FastAPI backend
    const backendUrl = `${PHYSDEEPSIF_BACKEND_URL}/api/analyze`
    console.log(`[PHYSDEEPSIF API] Forwarding to backend: ${backendUrl}`)

    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      body: backendForm,
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: "Backend error" }))
      console.error(`[PHYSDEEPSIF API] Backend error:`, errorData)
      return NextResponse.json(
        { message: errorData.detail || "Backend processing failed" },
        { status: backendResponse.status }
      )
    }

    const result: PhysDeepSIFResult = await backendResponse.json()
    console.log(`[PHYSDEEPSIF API] Result received: jobId=${result.jobId}, time=${result.processingTime}s`)

    // Store in memory for job-status polling compatibility
    physdeepsifJobs.set(result.jobId, {
      status: 'completed',
      startTime,
      result,
    })

    return NextResponse.json({
      success: true,
      ...result,
    })

  } catch (error) {
    console.error("[PHYSDEEPSIF API] Error:", error)

    // Check if this is a connection error (backend not running)
    const errorMessage = error instanceof Error ? error.message : "Unknown error"
    if (errorMessage.includes("ECONNREFUSED") || errorMessage.includes("fetch failed")) {
      return NextResponse.json(
        {
          message: "PhysDeepSIF backend is not running. Start it with: cd backend && python server.py",
          detail: errorMessage,
        },
        { status: 503 }
      )
    }

    return NextResponse.json(
      { message: errorMessage },
      { status: 500 }
    )
  }
}
