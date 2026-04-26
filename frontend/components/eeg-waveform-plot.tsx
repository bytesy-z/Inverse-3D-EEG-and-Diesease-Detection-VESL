"use client"

import { useEffect, useRef, useState } from "react"
import type { Data, Layout, Config } from "plotly.js-dist-min"
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

interface EegWaveformPlotProps {
  eegData: EegData
  selectedWindow?: number
  onSelectedWindowChange?: (windowIndex: number) => void
  className?: string
}

export function EegWaveformPlot({
  eegData,
  selectedWindow = 0,
  onSelectedWindowChange,
  className = "",
}: EegWaveformPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [uVPerDiv, setUVPerDiv] = useState<number>(50)

  const windowCount = eegData.windows?.length ?? 0
  const effectiveWindowIndex = Math.min(Math.max(selectedWindow ?? 0, 0), Math.max(windowCount - 1, 0))
  const currentWindow = eegData.windows?.[effectiveWindowIndex]
  const channels = eegData.channels ?? []
  const samplingRate = eegData.samplingRate ?? 200
  const windowLength = eegData.windowLength ?? 400

  const setWindowIndex = (nextIndex: number) => {
    if (windowCount <= 0) return
    const clamped = Math.min(Math.max(nextIndex, 0), windowCount - 1)
    onSelectedWindowChange?.(clamped)
  }

  const uVPerDivOptions = [5, 10, 15, 20, 25, 50, 100]

  /* ---- Render Plotly waveform ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container || !currentWindow) return

    let cancelled = false
    setLoading(true)

    const run = async () => {
      if (cancelled) return

      let Plotly: any;
      try {
        Plotly = await import("plotly.js-dist-min").then(m => m.default)
      } catch (err) {
        console.warn("Failed to load plotly.js module, falling back to window.Plotly", err)
        Plotly = (window as any).Plotly
      }
      if (cancelled || !Plotly) return

      const { data: eegValues, startTime, endTime } = currentWindow
      const numChannels = channels.length
      const numSamples = eegValues[0]?.length ?? 0
      const timeAxis = Array.from({ length: numSamples }, (_, i) => startTime + i / samplingRate)

      const spacing = 100 // vertical spacing per channel
      const gain = spacing / uVPerDiv // compute multiplier from selected scale
      const traces: Data[] = channels.map((channel, chIdx) => {
        const values = eegValues[chIdx] ?? []
        const offset = (numChannels - 1 - chIdx) * spacing
        const yCoords = values.map((v) => v * gain + offset)
        return {
          x: timeAxis,
          y: yCoords,
          mode: "lines" as const,
          name: channel,
          line: {
            color: "#000000",
            width: 1,
          },
          hovertemplate: `<b>${channel}</b><br>Time: %{x}s<br>Value: %{customdata} µV<extra></extra>`,
          customdata: values,
          yaxis: "y",
        }
      })

      const layout: Partial<Layout> = {
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {
          color: "#000000",
          size: 10,
        },
        margin: { l: 50, r: 20, t: 30, b: 40 },
        xaxis: {
          title: "Time (s)",
          titlefont: { color: "#000000", size: 11 },
          tickfont: { color: "#000000", size: 9 },
          gridcolor: "#e2e8f0",
          zerolinecolor: "#cbd5e1",
          zerolinewidth: 1,
          dtick: 0.5,
          range: [startTime, endTime],
        },
        yaxis: {
          title: "",
          tickfont: { color: "#000000", size: 9 },
          gridcolor: "#e2e8f0",
          zerolinecolor: "#cbd5e1",
          zerolinewidth: 1,
          showticklabels: true,
          // Keep ticks aligned with per-channel offsets; range sized to accommodate scaled waveforms
          tickvals: channels.map((_, i) => (numChannels - 1 - i) * spacing),
          ticktext: channels,
          dtick: spacing,
          range: [-spacing, (numChannels - 1) * spacing + spacing],
        },
        showlegend: false,
        hovermode: "x unified" as const,
        dragmode: "pan" as const,
        annotations: [
          {
            xref: 'paper', yref: 'paper', x: 0.02, y: 0.95,
            text: `Scale: ${uVPerDiv} µV / div`, showarrow: false,
            font: { color: '#000000', size: 12 }
          }
        ],
        height: Math.max(400, numChannels * 28),
      }

      const config: Partial<Config> = {
        responsive: true,
        displayModeBar: false,
        scrollZoom: true,
      }
      // UI: simple gain control
      // Insert a small gain control UI in header (handled in JSX below)
      // Use react-like update if already initialized, otherwise create
      if ((container as any)._plotlyInitialized) {
        Plotly.react(container, traces, layout, config)
      } else {
        Plotly.newPlot(container, traces, layout, config)
        ;(container as any)._plotlyInitialized = true
      }
      setLoading(false)
    }

    run().catch(console.error)

    return () => {
      cancelled = true
      // Purge Plotly instance to free memory and avoid canvas leaks
      import("plotly.js-dist-min").then(m => {
        const Plotly = m.default
        if (container) Plotly.purge(container)
      }).catch(() => {/* ignore — container may already be gone */})
    }
  }, [eegData, effectiveWindowIndex, currentWindow, channels, samplingRate, uVPerDiv])

  /* ---- Fullscreen toggle ---- */
  const toggleFullscreen = () => {
    const el = containerRef.current?.parentElement
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => setIsFullscreen(true))
    } else {
      document.exitFullscreen?.().then(() => setIsFullscreen(false))
    }
  }

  useEffect(() => {
    const handleFs = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener("fullscreenchange", handleFs)
    return () => document.removeEventListener("fullscreenchange", handleFs)
  }, [])

  return (
    <Card
      className={`relative overflow-hidden ${isFullscreen ? "!rounded-none !border-none" : ""} ${className}`}
    >
      <div className="border-b border-border px-4 py-2">
        <div className="flex items-center justify-between w-full">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            EEG Waveform {windowCount > 1 ? `(Window ${effectiveWindowIndex + 1}/${windowCount})` : ""}
          </span>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Scale</span>
            <select
              className="bg-transparent border border-border rounded px-2 py-1 text-sm focus:outline-none"
              value={uVPerDiv}
              onChange={(e) => setUVPerDiv(Number(e.target.value))}
            >
              {uVPerDivOptions.map(opt => (
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
        {isFullscreen ? (
          <Minimize2 className="h-4 w-4" />
        ) : (
          <Maximize2 className="h-4 w-4" />
        )}
      </Button>

      {loading && (
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-card">
          <Skeleton className="h-full w-full rounded-none" />
        </div>
      )}

      <div
        ref={containerRef}
        className="w-full"
        style={{ minHeight: Math.max(400, channels.length * 28) }}
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
