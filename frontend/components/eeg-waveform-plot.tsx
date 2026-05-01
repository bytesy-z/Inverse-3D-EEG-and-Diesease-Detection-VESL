"use client"

import { useRef, useState, useEffect, useCallback } from "react"
import { Maximize2, Minimize2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

interface EegData {
  channels: string[]
  samplingRate: number
  windowLength: number
  windows: Array<{
    startTime: number
    endTime: number
    data: number[][]
  }>
}

interface HighlightSegment {
  channel_idx: number
  start_time_sec: number
  end_time_sec: number
  importance: number
}

interface EegWaveformPlotProps {
  eegData: EegData
  selectedWindow?: number
  onSelectedWindowChange?: (index: number) => void
  className?: string
  highlightSegments?: HighlightSegment[]
}

const PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.3.min.js"

const CHANNEL_COLORS = [
  "#7ec8e3", "#a8d8ea", "#aa96da", "#fcbad3", "#a8e6cf",
  "#dcedc1", "#ffd3b6", "#ffaaa5", "#ff8b94", "#b8a9c9",
  "#c9c9ff", "#b5e7a0", "#e3eaa7", "#f7cac9", "#92a8d1",
  "#f0e68c", "#d5a6bd", "#c3e0e5", "#9b59b6",
]

function channelColor(idx: number): string {
  return CHANNEL_COLORS[idx % CHANNEL_COLORS.length]
}

function ensurePlotly(): Promise<void> {
  return new Promise((resolve, _reject) => {
    if ((window as any).Plotly) {
      resolve()
      return
    }
    const script = document.createElement("script")
    script.src = PLOTLY_CDN
    script.onload = () => resolve()
    script.onerror = () => {
      console.error("Failed to load Plotly from CDN")
      resolve()
    }
    document.head.appendChild(script)
  })
}

export function EegWaveformPlot({
  eegData,
  selectedWindow = 0,
  onSelectedWindowChange,
  className = "",
  highlightSegments,
}: EegWaveformPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [plotlyReady, setPlotlyReady] = useState(false)
  const [loading, setLoading] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [uVPerDiv, setUVPerDiv] = useState<number>(50)

  const windowCount = eegData.windows?.length ?? 0
  const effectiveWindowIndex = Math.min(Math.max(selectedWindow ?? 0, 0), Math.max(windowCount - 1, 0))
  const channels = eegData.channels ?? []
  const nChannels = channels.length

  const uVPerDivOptions = [5, 10, 15, 20, 25, 50, 100]

  const setWindowIndex = useCallback(
    (nextIndex: number) => {
      if (windowCount <= 0) return
      const clamped = Math.min(Math.max(nextIndex, 0), windowCount - 1)
      onSelectedWindowChange?.(clamped)
    },
    [windowCount, onSelectedWindowChange],
  )

  /* ---- Load Plotly ---- */
  useEffect(() => {
    let cancelled = false
    ensurePlotly().then(() => {
      if (!cancelled) setPlotlyReady(true)
    })
    return () => {
      cancelled = true
    }
  }, [])

  /* ---- Render Plotly figure ---- */
  useEffect(() => {
    if (!plotlyReady || windowCount === 0) return

    const container = containerRef.current
    if (!container) return

    let cancelled = false
    const Plotly = (window as any).Plotly
    if (!Plotly?.newPlot) return

    const win = eegData.windows[effectiveWindowIndex]
    if (!win?.data?.length) return

    const sr = eegData.samplingRate
    const nSamples = eegData.windowLength

    const timeArray: number[] = []
    for (let i = 0; i < nSamples; i++) {
      timeArray.push(win.startTime + i / sr)
    }

    let globalMax = 0
    for (let ch = 0; ch < nChannels; ch++) {
      const d = win.data[ch]
      if (!d) continue
      for (let s = 0; s < d.length; s++) {
        const abs = Math.abs(d[s])
        if (abs > globalMax) globalMax = abs
      }
    }
    if (globalMax === 0) globalMax = 1
    const channelOffset = globalMax * 3 * (50 / uVPerDiv)

    /* ---- Build traces ---- */
    const traces: Record<string, unknown>[] = []
    for (let ch = 0; ch < nChannels; ch++) {
      const yOffset = (nChannels - 1 - ch) * channelOffset
      const raw = win.data[ch]
      if (!raw) continue
      const yValues = raw.map((v) => v + yOffset)
      traces.push({
        x: timeArray,
        y: yValues,
        type: "scatter",
        mode: "lines",
        name: channels[ch],
        line: {
          color: channelColor(ch),
          width: 1,
          shape: "linear",
        },
        hoverinfo: "name",
        showlegend: false,
        hovertemplate: `${channels[ch]}<extra></extra>`,
      })
    }

    /* ---- Highlight shapes ---- */
    const shapes: Record<string, unknown>[] = []
    if (highlightSegments?.length) {
      for (const seg of highlightSegments) {
        if (seg.channel_idx < 0 || seg.channel_idx >= nChannels) continue
        const yCenter = (nChannels - 1 - seg.channel_idx) * channelOffset
        const halfBand = channelOffset * 0.48
        shapes.push({
          type: "rect",
          x0: seg.start_time_sec,
          x1: seg.end_time_sec,
          y0: yCenter - halfBand,
          y1: yCenter + halfBand,
          fillcolor: `rgba(239, 68, 68, ${Math.max(0.08, Math.min(0.35, seg.importance * 0.4))})`,
          line: { width: 0 },
          layer: "below",
        })
      }
    }

    /* ---- Layout ---- */
    const tickvals: number[] = []
    const ticktext: string[] = []
    for (let i = 0; i < nChannels; i++) {
      tickvals.push((nChannels - 1 - i) * channelOffset)
      ticktext.push(channels[i])
    }

    const layout: Record<string, unknown> = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#a1a1aa", size: 10 },
      margin: { l: 58, r: 16, t: 5, b: 36 },
      showlegend: false,
      hovermode: "closest",
      dragmode: "pan",
      xaxis: {
        title: { text: "Time (s)", font: { color: "#a1a1aa", size: 10 } },
        tickfont: { color: "#a1a1aa", size: 9 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
        range: [win.startTime, win.endTime],
      },
      yaxis: {
        tickvals,
        ticktext,
        tickfont: { color: "#a1a1aa", size: 8 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
        range: [-channelOffset * 0.7, (nChannels - 0.3) * channelOffset],
        fixedrange: false,
      },
      shapes,
    }

    const config: Record<string, unknown> = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      displaylogo: false,
      scrollZoom: true,
    }

    Plotly.purge(container)
    Plotly.newPlot(container, traces, layout, config)
      .then(() => {
        if (!cancelled) setLoading(false)
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      if (containerRef.current) {
        const P = (window as any).Plotly
        if (P?.purge) P.purge(containerRef.current)
      }
    }
  }, [plotlyReady, eegData, effectiveWindowIndex, uVPerDiv, highlightSegments, channels, nChannels, windowCount])

  /* ---- Resize on window resize ---- */
  useEffect(() => {
    if (!plotlyReady) return
    const container = containerRef.current
    if (!container) return

    const handleResize = () => {
      const P = (window as any).Plotly
      if (P?.Plots?.resize && containerRef.current) {
        P.Plots.resize(containerRef.current)
      }
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [plotlyReady])

  /* ---- Fullscreen toggle ---- */
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current?.parentElement
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => {
        setIsFullscreen(true)
        setTimeout(() => {
          if (containerRef.current) {
            const P = (window as any).Plotly
            P?.Plots?.resize?.(containerRef.current)
          }
        }, 200)
      })
    } else {
      document.exitFullscreen?.().then(() => {
        setIsFullscreen(false)
        setTimeout(() => {
          if (containerRef.current) {
            const P = (window as any).Plotly
            P?.Plots?.resize?.(containerRef.current)
          }
        }, 200)
      })
    }
  }, [])

  useEffect(() => {
    const handleFs = () => {
      setIsFullscreen(!!document.fullscreenElement)
      setTimeout(() => {
        if (containerRef.current) {
          const P = (window as any).Plotly
          P?.Plots?.resize?.(containerRef.current)
        }
      }, 200)
    }
    document.addEventListener("fullscreenchange", handleFs)
    return () => document.removeEventListener("fullscreenchange", handleFs)
  }, [])

  return (
    <Card
      className={`relative overflow-hidden ${isFullscreen ? "!rounded-none !border-none" : ""} ${className}`}
    >
      {/* Header bar */}
      <div className="border-b border-border px-4 py-2">
        <div className="flex items-center justify-between w-full">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            EEG Waveform{windowCount > 1 ? ` (Window ${effectiveWindowIndex + 1}/${windowCount})` : ""}
          </span>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Scale</span>
            <select
              className="bg-transparent border border-border rounded px-2 py-1 text-sm focus:outline-none"
              value={uVPerDiv}
              onChange={(e) => setUVPerDiv(Number(e.target.value))}
            >
              {uVPerDivOptions.map((opt) => (
                <option key={opt} value={opt} className="bg-background text-foreground">
                  {opt} µV/div
                </option>
              ))}
            </select>
          </div>
        </div>
        {windowCount > 1 && (
          <div className="mt-2 flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => setWindowIndex(effectiveWindowIndex - 1)}
              disabled={effectiveWindowIndex <= 0}
            >
              Prev
            </Button>
            <input
              type="range"
              min={0}
              max={windowCount - 1}
              step={1}
              value={effectiveWindowIndex}
              onChange={(e) => setWindowIndex(Number(e.target.value))}
              className="h-7 w-full"
              aria-label="Select EEG window"
            />
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => setWindowIndex(effectiveWindowIndex + 1)}
              disabled={effectiveWindowIndex >= windowCount - 1}
            >
              Next
            </Button>
          </div>
        )}
      </div>

      {/* Fullscreen toggle button */}
      <Button
        variant="ghost"
        size="sm"
        className="absolute right-2 top-2 z-10 h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
        onClick={toggleFullscreen}
        aria-label={isFullscreen ? "Exit full screen" : "Full screen"}
      >
        {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
      </Button>

      {/* Loading skeleton */}
      {loading && (
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-card">
          <Skeleton className="h-full w-full rounded-none" />
        </div>
      )}

      {/* Plotly container */}
      <div
        ref={containerRef}
        className="w-full"
        style={{
          minHeight: Math.max(400, nChannels * 28),
          height: isFullscreen ? "calc(100vh - 40px)" : undefined,
        }}
      />
    </Card>
  )
}

export function EegWaveformSkeleton({ className = "" }: { className?: string }) {
  return (
    <Card className={`relative overflow-hidden ${className}`}>
      <div className="border-b border-border px-4 py-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          EEG Waveform
        </span>
      </div>
      <div className="flex items-center justify-center p-8">
        <Skeleton className="h-[400px] w-full" />
      </div>
    </Card>
  )
}
