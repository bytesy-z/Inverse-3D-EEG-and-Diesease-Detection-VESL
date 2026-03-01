import { type NextRequest, NextResponse } from "next/server"
import { readFile } from "fs/promises"
import path from "path"

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const { path: pathSegments } = await params
    const repoRoot = path.join(process.cwd(), "..")
    const filePath = path.join(repoRoot, "results", ...pathSegments)

    // Security check: ensure the path is within the results directory
    const normalizedPath = path.normalize(filePath)
    const resultsDir = path.join(repoRoot, "results")
    if (!normalizedPath.startsWith(resultsDir)) {
      return NextResponse.json({ message: "Access denied" }, { status: 403 })
    }

    const fileBuffer = await readFile(filePath)

    // Determine content type based on file extension
    const ext = path.extname(filePath).toLowerCase()
    const contentTypeMap: Record<string, string> = {
      ".npz": "application/octet-stream",
      ".json": "application/json",
      ".html": "text/html",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
    }

    const contentType = contentTypeMap[ext] || "application/octet-stream"

    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    })
  } catch (error) {
    console.error("Error serving result file:", error)
    return NextResponse.json(
      { message: "File not found" },
      { status: 404 }
    )
  }
}
