"use client"

import { useEffect, useRef, useState } from "react"
import { Maximize2, Minimize2, Play, Pause } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

const PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.3.min.js"

interface AnimationFrameData {
  scoresArray: number[]
  timestamp: number
}

interface BrainVisualizationProps {
  plotHtml: string
  animationData?: AnimationFrameData[]
  vertexRegion?: number[]
  label?: string
  className?: string
  playbackSpeed?: number
  currentFrame?: number
  onFrameChange?: (frameIndex: number) => void
}

function ensurePlotly(): Promise<void> {
  return new Promise((resolve) => {
    if ((window as any).Plotly) {
      console.debug("[BrainVis] Plotly already loaded globally")
      resolve()
      return
    }
    const script = document.createElement("script")
    script.src = PLOTLY_CDN
    script.onload = () => { console.debug("[BrainVis] Plotly CDN script loaded"); resolve() }
    script.onerror = () => { console.error("[BrainVis] Failed to load Plotly from CDN"); resolve() }
    document.head.appendChild(script)
  })
}

export function BrainVisualization({
  plotHtml,
  animationData,
  vertexRegion,
  label,
  className = "",
  playbackSpeed = 1,
  currentFrame,
  onFrameChange,
}: BrainVisualizationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotDivRef = useRef<any>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isPlayingRef = useRef(false)
  const localFrameRef = useRef(0)

  const [loading, setLoading] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [plotlyReady, setPlotlyReady] = useState(false)

  // Refs to hold latest animation data (avoids closure stale-capture issues)
  const animationDataRef = useRef(animationData)
  const vertexRegionRef = useRef(vertexRegion)
  const nFramesRef = useRef(0)

  useEffect(() => {
    animationDataRef.current = animationData
    vertexRegionRef.current = vertexRegion
    nFramesRef.current = animationData?.length ?? 0
  })

  const hasAnimation = !!(animationData && animationData.length > 1)
  const nFrames = animationData?.length ?? 0

  console.debug(
    "[BrainVis] Render:", { hasAnimation, nFrames, loading, currentFrame,
      animDataLen: animationData?.length, vertRegLen: vertexRegion?.length,
      plotHtmlLen: plotHtml?.length }
  )

  /* ---- Load Plotly ---- */
  useEffect(() => {
    let cancelled = false
    console.debug("[BrainVis] ensurePlotly start")
    ensurePlotly().then(() => {
      if (!cancelled) { console.debug("[BrainVis] ensurePlotly resolved"); setPlotlyReady(true) }
    })
    return () => { cancelled = true }
  }, [])

  /* ---- Inject static Plotly HTML ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container || !plotHtml || !plotlyReady) {
      console.debug("[BrainVis] Injection skipped", { hasContainer: !!container, hasHtml: !!plotHtml, plotlyReady })
      return
    }

    let cancelled = false
    console.debug("[BrainVis] Injection starting, setLoading(true)")
    setLoading(true)

    const run = async () => {
      container.innerHTML = ""
      console.debug("[BrainVis] Container cleared")

      const wrapper = document.createElement("div")
      wrapper.innerHTML = plotHtml

      const scripts = Array.from(wrapper.querySelectorAll("script"))
      scripts.forEach((s) => s.parentNode?.removeChild(s))

      while (wrapper.firstChild) {
        container.appendChild(wrapper.firstChild)
      }

      let scriptIdx = 0
      for (const oldScript of scripts) {
        if (cancelled) break
        const srcLabel = oldScript.src || "<inline>"
        console.debug(`[BrainVis] Executing script ${scriptIdx}/${scripts.length}: ${srcLabel}`)
        await new Promise<void>((resolve) => {
          const newScript = document.createElement("script")
          Array.from(oldScript.attributes).forEach((a) =>
            newScript.setAttribute(a.name, a.value)
          )
          if (oldScript.src) {
            newScript.onload = () => { console.debug(`[BrainVis] Script loaded: ${oldScript.src}`); resolve() }
            newScript.onerror = () => { console.error(`[BrainVis] Script failed: ${oldScript.src}`); resolve() }
            newScript.src = oldScript.src
          } else {
            newScript.textContent = oldScript.textContent
            container.appendChild(newScript)
            resolve()
          }
          container.appendChild(newScript)
        })
        scriptIdx++
      }

      if (!cancelled) {
        const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
        plotDivRef.current = plotDiv ?? null
        console.debug("[BrainVis] plotDiv found:", !!plotDiv)

        const Plotly = (window as any).Plotly
        if (plotDiv && Plotly?.Plots?.resize) {
          plotDiv.style.width = "100%"
          plotDiv.style.height = "100%"
          Plotly.Plots.resize(plotDiv)
          console.debug("[BrainVis] Plotly resized")
        }

        console.debug("[BrainVis] Injection complete, setLoading(false)")
        setLoading(false)
      } else {
        console.debug("[BrainVis] Injection cancelled mid-way")
      }
    }

    run().catch((err) => {
      console.error("[BrainVis] Injection failed:", err)
      setLoading(false)
    })

    return () => {
      cancelled = true
      console.debug("[BrainVis] Cleanup: cancelling injection, clearing timer")
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      isPlayingRef.current = false
      setIsPlaying(false)
      plotDivRef.current = null
      container.innerHTML = ""
    }
  }, [plotHtml, plotlyReady])

  /* ---- Compute vertex colors + switch frame (uses refs for data) ---- */
  const switchToFrame = (frameIndex: number) => {
    const plotDiv = plotDivRef.current
    const Plotly = (window as any).Plotly
    const ad = animationDataRef.current
    const vr = vertexRegionRef.current

    console.debug("[BrainVis] switchToFrame", { frameIndex, hasPlotDiv: !!plotDiv, hasRestyle: !!Plotly?.restyle,
      adLen: ad?.length, vrLen: vr?.length })

    if (!plotDiv || !Plotly?.restyle || !ad || !vr) {
      console.debug("[BrainVis] switchToFrame: SKIP — missing prerequisites")
      return
    }

    const clampedIdx = Math.max(0, Math.min(frameIndex, ad.length - 1))
    const frame = ad[clampedIdx]
    if (!frame) { console.debug("[BrainVis] switchToFrame: SKIP — no frame data at", clampedIdx); return }

    const nVerts = vr.length
    const scores = frame.scoresArray
    console.debug("[BrainVis] switchToFrame: computing vertex colors", { nVerts, scoresLen: scores.length, clampedIdx })

    const colors = new Float64Array(nVerts)
    for (let v = 0; v < nVerts; v++) {
      const rIdx = vr[v]
      colors[v] = rIdx >= 0 && rIdx < scores.length ? scores[rIdx] : 0
    }

    console.debug("[BrainVis] switchToFrame: calling Plotly.restyle with intensity array length", colors.length,
      "first 3 vals:", colors[0], colors[1], colors[2])

    Plotly.restyle(plotDiv, { intensity: [colors] }, [0])
      .then(() => console.debug("[BrainVis] Plotly.restyle SUCCESS for frame", frameIndex))
      .catch((err: unknown) => console.error("[BrainVis] Plotly.restyle FAILED:", err))
  }

  /* ---- Sync to external currentFrame ---- */
  useEffect(() => {
    console.debug("[BrainVis] currentFrame effect firing", { loading, currentFrame, hasAnimation,
      adAvailable: !!animationDataRef.current, vrAvailable: !!vertexRegionRef.current })

    if (!loading && currentFrame !== undefined) {
      switchToFrame(currentFrame)
    }
  }, [currentFrame, loading])

  /* ---- Play/pause toggle ---- */
  const togglePlay = () => {
    if (!hasAnimation) { console.debug("[BrainVis] togglePlay: SKIP — no animation"); return }

    if (isPlayingRef.current) {
      console.debug("[BrainVis] togglePlay: PAUSING")
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      isPlayingRef.current = false
      setIsPlaying(false)
    } else {
      const intervalMs = Math.round(200 / playbackSpeed)
      console.debug("[BrainVis] togglePlay: PLAYING, interval=", intervalMs, "nFrames=", nFrames)
      isPlayingRef.current = true
      setIsPlaying(true)

      timerRef.current = setInterval(() => {
        const prev = localFrameRef.current
        const nextIdx = prev + 1
        const wrapped = nextIdx >= nFrames ? 0 : nextIdx
        localFrameRef.current = wrapped
        console.debug("[BrainVis] Timer tick:", { prev, wrapped })
        switchToFrame(wrapped)
        onFrameChange?.(wrapped)
      }, intervalMs)
    }
  }

  /* ---- Rebuild timer on speed change ---- */
  useEffect(() => {
    if (!isPlayingRef.current || !timerRef.current) {
      console.debug("[BrainVis] Speed effect: not playing, skipping")
      return
    }
    console.debug("[BrainVis] Speed effect: rebuilding timer with speed", playbackSpeed)
    clearInterval(timerRef.current)
    const intervalMs = Math.round(200 / playbackSpeed)
    timerRef.current = setInterval(() => {
      const nextIdx = localFrameRef.current + 1
      const wrapped = nextIdx >= nFrames ? 0 : nextIdx
      localFrameRef.current = wrapped
      console.debug("[BrainVis] Speed-timer tick:", { prev: nextIdx - 1, wrapped })
      switchToFrame(wrapped)
      onFrameChange?.(wrapped)
    }, intervalMs)
  }, [playbackSpeed])

  /* ---- Resize ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const handleResize = () => {
      const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
      const Plotly = (window as any).Plotly
      if (plotDiv && Plotly?.Plots?.resize) {
        plotDiv.style.width = "100%"
        plotDiv.style.height = "100%"
        Plotly.Plots.resize(plotDiv)
      }
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  /* ---- Fullscreen ---- */
  const toggleFullscreen = () => {
    const el = containerRef.current?.parentElement
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.()
        .then(() => setIsFullscreen(true))
        .catch(() => {})
      setTimeout(() => {
        const plotDiv = containerRef.current?.querySelector<HTMLElement>(".plotly-graph-div")
        const Plotly = (window as any).Plotly
        if (plotDiv && Plotly?.Plots?.resize) Plotly.Plots.resize(plotDiv)
      }, 200)
    } else {
      document.exitFullscreen?.()
        .then(() => setIsFullscreen(false))
        .catch(() => {})
      setTimeout(() => {
        const plotDiv = containerRef.current?.querySelector<HTMLElement>(".plotly-graph-div")
        const Plotly = (window as any).Plotly
        if (plotDiv && Plotly?.Plots?.resize) Plotly.Plots.resize(plotDiv)
      }, 200)
    }
  }

  useEffect(() => {
    const handleFs = () => {
      setIsFullscreen(!!document.fullscreenElement)
      setTimeout(() => {
        const plotDiv = containerRef.current?.querySelector<HTMLElement>(".plotly-graph-div")
        const Plotly = (window as any).Plotly
        if (plotDiv && Plotly?.Plots?.resize) Plotly.Plots.resize(plotDiv)
      }, 200)
    }
    document.addEventListener("fullscreenchange", handleFs)
    return () => document.removeEventListener("fullscreenchange", handleFs)
  }, [])

  return (
    <Card
      className={`relative overflow-hidden ${isFullscreen ? "!rounded-none !border-none" : ""} ${className}`}
    >
      {label && (
        <div className="border-b border-border px-4 py-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
        </div>
      )}

      <Button
        variant="ghost"
        size="sm"
        className="absolute right-2 top-2 z-10 h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
        onClick={toggleFullscreen}
        aria-label={isFullscreen ? "Exit full screen" : "Full screen"}
      >
        {isFullscreen ? (
          <Minimize2 className="h-4 w-4" />
        ) : (
          <Maximize2 className="h-4 w-4" />
        )}
      </Button>

      {hasAnimation && (
        <Button
          variant="ghost"
          size="sm"
          className="absolute left-2 top-2 z-10 h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
          onClick={togglePlay}
          aria-label={isPlaying ? "Pause animation" : "Play animation"}
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>
      )}

      {loading && (
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-card">
          <Skeleton className="h-full w-full rounded-none" />
        </div>
      )}

      <div
        ref={containerRef}
        className="plotly-container w-full"
        style={{ height: isFullscreen ? "calc(100vh - 40px)" : undefined, minHeight: isFullscreen ? "100%" : 480 }}
      />
    </Card>
  )
}
