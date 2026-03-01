import { type NextRequest, NextResponse } from "next/server"
import { PHYSDEEPSIF_BACKEND_URL, matJobs } from "@/lib/job-store"

export const maxDuration = 300
export const dynamic = 'force-dynamic'

/**
 * POST /api/analyze-mat
 *
 * Proxy route for the EEG Source Localization page (MAT file upload).
 * Forwards uploaded MAT files to the FastAPI backend's /api/analyze endpoint
 * with mode=source_localization, which runs PhysDeepSIF inference and returns
 * a 3D heatmap of estimated source activity (ESI), NOT epileptogenicity scores.
 *
 * Returns results synchronously (inference takes ~0.2s on GPU).
 */
export async function POST(request: NextRequest) {
  const startTime = Date.now()
  console.log("[MAT API] Received request")

  try {
    const formData = await request.formData()
    const file = formData.get("file") as File | null

    if (!file) {
      return NextResponse.json({ message: "No file provided" }, { status: 400 })
    }

    if (!file.name.endsWith(".mat")) {
      return NextResponse.json(
        { message: "Invalid file format. Please upload a MAT file." },
        { status: 400 }
      )
    }

    console.log(`[MAT API] Forwarding MAT: ${file.name} (${file.size} bytes)`)

    // Forward the file to the Python FastAPI backend
    // mode=source_localization: ESI heatmap of brain activity (not epileptogenicity)
    const backendForm = new FormData()
    backendForm.append("file", file)
    backendForm.append("mode", "source_localization")

    const backendUrl = `${PHYSDEEPSIF_BACKEND_URL}/api/analyze`
    console.log(`[MAT API] Backend URL: ${backendUrl}`)

    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      body: backendForm,
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: "Backend error" }))
      console.error("[MAT API] Backend error:", errorData)
      return NextResponse.json(
        { message: errorData.detail || "Backend processing failed" },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    const processingTime = (Date.now() - startTime) / 1000
    console.log(`[MAT API] Done: ${processingTime.toFixed(1)}s, jobId=${result.jobId}`)

    // Store in memory for job-status compatibility
    matJobs.set(result.jobId, {
      status: 'completed',
      startTime,
      result: {
        plotHtml: result.plotHtml,
        outputDir: "",
        processingTime,
      },
    })

    // Return synchronously (no polling needed)
    return NextResponse.json({
      success: true,
      mode: "source_localization",
      plotHtml: result.plotHtml,
      processingTime,
      outputDir: "",
      nWindowsProcessed: result.nWindowsProcessed,
      hasAnimation: result.hasAnimation,
      source: result.source,
    })

  } catch (error) {
    console.error("[MAT API] Error:", error)
    const msg = error instanceof Error ? error.message : "Unknown error"
    if (msg.includes("ECONNREFUSED") || msg.includes("fetch failed")) {
      return NextResponse.json(
        { message: "PhysDeepSIF backend is not running. Start it with: ./start.sh" },
        { status: 503 }
      )
    }
    return NextResponse.json({ message: msg }, { status: 500 })
  }
}
