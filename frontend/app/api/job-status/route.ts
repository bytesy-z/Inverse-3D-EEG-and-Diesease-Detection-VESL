import { type NextRequest, NextResponse } from "next/server"
import { readFile } from "fs/promises"
import path from "path"
import { eegJobs, matJobs } from "@/lib/job-store"

export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const jobId = searchParams.get("jobId")
  const outputDir = searchParams.get("outputDir")

  console.log(`[JOB STATUS] Checking status for jobId: ${jobId}, outputDir: ${outputDir}`)

  if (!jobId) {
    console.error("[JOB STATUS] Missing jobId parameter")
    return NextResponse.json({ message: "Missing jobId parameter" }, { status: 400 })
  }

  // Check in-memory job stores first
  console.log(`[JOB STATUS] Checking in-memory job stores...`)
  let job = eegJobs.get(jobId) || matJobs.get(jobId)
  console.log(`[JOB STATUS] In-memory job found:`, job ? "YES" : "NO")

  // If not found in memory, check disk (in case server restarted)
  if (!job && outputDir) {
    console.log(`[JOB STATUS] Job not in memory, checking disk...`)
    const repoRoot = path.join(process.cwd(), "..")
    const jobInfoPath = path.join(repoRoot, outputDir, "_job_info.json")
    console.log(`[JOB STATUS] Looking for job file at: ${jobInfoPath}`)
    
    try {
      const content = await readFile(jobInfoPath, "utf8")
      const diskJob = JSON.parse(content)
      console.log(`[JOB STATUS] Found disk job:`, diskJob)
      
      if (diskJob.status === 'completed') {
        console.log(`[JOB STATUS] Disk job completed, reading results...`)
        // Read the plot HTML
        const summaryPath = path.join(repoRoot, outputDir, "best_window_summary.json")
        console.log(`[JOB STATUS] Reading summary from: ${summaryPath}`)
        const summaryContent = await readFile(summaryPath, "utf8")
        const summary = JSON.parse(summaryContent)

        const plotPath = path.join(repoRoot, outputDir, summary.interactive_plot)
        console.log(`[JOB STATUS] Reading plot from: ${plotPath}`)
        const plotHtml = await readFile(plotPath, "utf8")

        const bodyMatch = plotHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i)
        const plotContent = bodyMatch ? bodyMatch[1] : plotHtml
        const styledPlotHtml = `<div style='width: 100%; height: 100%; display: flex; justify-content: center; align-items: center;'>${plotContent}</div>`

        const response = {
          status: 'completed',
          processingTime: diskJob.processingTime,
          result: {
            plotHtml: styledPlotHtml,
            outputDir: outputDir,
            processingTime: diskJob.processingTime,
          }
        }
        console.log(`[JOB STATUS] Returning completed response`)
        return NextResponse.json(response)
      } else if (diskJob.status === 'failed') {
        return NextResponse.json({
          status: 'failed',
          error: diskJob.error || 'Processing failed',
        })
      } else {
        return NextResponse.json({
          status: 'processing',
          elapsedSeconds: Math.round((Date.now() - diskJob.startTime) / 1000),
        })
      }
    } catch (error) {
      // File not found or parse error - job might still be running
      console.log(`[JOB STATUS] Could not read disk job file:`, error)
      return NextResponse.json({
        status: 'processing',
        message: 'Job still running or status file not yet created',
      })
    }
  }

  if (!job) {
    console.log(`[JOB STATUS] Job not found anywhere`)
    return NextResponse.json({ 
      status: 'not_found',
      message: "Job not found. It may have expired or server was restarted." 
    }, { status: 404 })
  }

  console.log(`[JOB STATUS] In-memory job status: ${job.status}`)

  if (job.status === 'completed' && job.result) {
    console.log(`[JOB STATUS] Job completed in memory, returning result`)
    return NextResponse.json({
      status: 'completed',
      processingTime: job.result.processingTime,
      result: job.result,
    })
  } else if (job.status === 'failed') {
    console.log(`[JOB STATUS] Job failed: ${job.error}`)
    return NextResponse.json({
      status: 'failed',
      error: job.error || 'Processing failed',
    })
  } else {
    const elapsed = Math.round((Date.now() - job.startTime) / 1000)
    console.log(`[JOB STATUS] Job still processing, elapsed: ${elapsed}s`)
    
    // Log memory usage periodically
    if (elapsed % 30 === 0) { // Every 30 seconds
      const memUsage = process.memoryUsage()
      console.log(`[JOB STATUS] Memory usage at ${elapsed}s:`, {
        rss: Math.round(memUsage.rss / 1024 / 1024) + 'MB',
        heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + 'MB',
        heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024) + 'MB'
      })
    }
    
    return NextResponse.json({
      status: 'processing',
      elapsedSeconds: elapsed,
    })
  }
}
