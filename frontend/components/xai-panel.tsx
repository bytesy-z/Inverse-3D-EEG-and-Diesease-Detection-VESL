"use client"

import { useEffect, useRef, useState } from "react"

interface XaiSegment {
  channel_idx: number
  start_time_sec: number
  end_time_sec: number
  importance: number
  window_idx?: number
}

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

interface XaiPanelProps {
  channelImportance: number[]
  timeImportance: number[]
  channelNames: string[]
  topSegments?: XaiSegment[]
  samplingRate?: number
  strideSamples?: number
  eegData?: EegData
  selectedWindow?: number
}

const uVPerDivOptions = [5, 10, 15, 20, 25, 50, 100]

export function XaiPanel({
  channelImportance,
  timeImportance,
  channelNames,
  topSegments = [],
  samplingRate = 200,
  strideSamples = 20,
  eegData,
  selectedWindow = 0,
}: XaiPanelProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const [plotlyReady, setPlotlyReady] = useState(() => {
    return typeof window !== "undefined" && !!(window as any).Plotly
  })
  const [uVPerDiv, setUVPerDiv] = useState<number>(50)

  const hasEeg = eegData && eegData.windows.length > 0

  useEffect(() => {
    if (plotlyReady) return
    const script = document.createElement("script")
    script.src = "https://cdn.plot.ly/plotly-2.35.3.min.js"
    script.onload = () => setPlotlyReady(true)
    script.onerror = () => console.error("Failed to load Plotly")
    document.head.appendChild(script)
  }, [plotlyReady])

  useEffect(() => {
    if (!plotlyReady || !hasEeg || !eegData || !plotRef.current) return

    let cancelled = false
    const Plotly = (window as any).Plotly
    if (!Plotly?.newPlot) return

    const numChannels = eegData.channels.length
    const sr = eegData.samplingRate
    const windows = eegData.windows

    // Concatenate all windows into full EEG
    const fullTime: number[] = []
    const fullData: number[][] = new Array(numChannels).fill(null).map(() => [])
    let totalTime = 0
    for (const w of windows) {
      const nSamples = w.data[0]?.length ?? 0
      for (let s = 0; s < nSamples; s++) {
        fullTime.push(totalTime + s / sr)
      }
      for (let ch = 0; ch < numChannels; ch++) {
        if (w.data[ch]) fullData[ch].push(...w.data[ch])
      }
      totalTime += w.endTime - w.startTime
    }
    const fullDuration = totalTime

    // Global max across all windows
    let globalMax = 0
    for (let ch = 0; ch < numChannels; ch++) {
      const d = fullData[ch]
      if (!d) continue
      for (let s = 0; s < d.length; s++) {
        const abs = Math.abs(d[s])
        if (abs > globalMax) globalMax = abs
      }
    }
    if (globalMax === 0) globalMax = 1
    const channelSpacing = globalMax * 3
    const gain = 50 / uVPerDiv

    // One trace per channel across full recording
    const traces: Record<string, unknown>[] = []
    for (let ch = 0; ch < numChannels; ch++) {
      const yOffset = (numChannels - 1 - ch) * channelSpacing
      const raw = fullData[ch]
      if (!raw) continue
      const yValues = raw.map((v) => v * gain + yOffset)
      traces.push({
        x: fullTime,
        y: yValues,
        type: "scatter",
        mode: "lines",
        name: eegData.channels[ch],
        line: { color: "#60a5fa", width: 0.6, shape: "linear" },
        hoverinfo: "name",
        showlegend: false,
        hovertemplate: `${eegData.channels[ch]}<extra></extra>`,
      })
    }

    // Green bars at absolute recording time (backend returns absolute timestamps)
    const shapes: Record<string, unknown>[] = topSegments
      .slice(0, 3)
      .filter((seg) => seg.channel_idx >= 0 && seg.channel_idx < numChannels)
      .map((seg) => {
        const yCenter = (numChannels - 1 - seg.channel_idx) * channelSpacing
        const halfBand = channelSpacing * 0.48
        return {
          type: "rect",
          x0: seg.start_time_sec,
          x1: seg.end_time_sec,
          y0: yCenter - halfBand,
          y1: yCenter + halfBand,
          fillcolor: "rgba(16, 185, 129, 0.25)",
          line: { width: 1, color: "rgba(16, 185, 129, 0.6)" },
          layer: "below",
        }
      })

    const layout: Record<string, unknown> = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#a1a1aa", size: 10 },
      height: 280,
      margin: { l: 58, r: 10, t: 5, b: 30 },
      showlegend: false,
      hovermode: "closest",
      dragmode: "pan",
      xaxis: {
        title: { text: "Time (s)", font: { color: "#a1a1aa", size: 10 } },
        tickfont: { color: "#a1a1aa", size: 9 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
        range: fullDuration > 0 ? [0, Math.min(fullDuration, 10)] : undefined,
      },
      yaxis: {
        tickvals: eegData.channels.map((_, i) => (numChannels - 1 - i) * channelSpacing),
        ticktext: eegData.channels,
        tickfont: { color: "#a1a1aa", size: 7 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
        range: [-channelSpacing * 0.7, (numChannels - 0.3) * channelSpacing],
        fixedrange: false,
      },
      shapes,
    }

    const config: Record<string, unknown> = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      displaylogo: false,
      scrollZoom: true,
    }

    Plotly.purge(plotRef.current)
    Plotly.newPlot(plotRef.current, traces, layout, config).catch(() => {})

    return () => {
      cancelled = true
      if (plotRef.current) Plotly.purge(plotRef.current)
    }
  }, [plotlyReady, hasEeg, eegData, selectedWindow, topSegments, uVPerDiv])

  useEffect(() => {
    if (!plotlyReady) return
    const container = plotRef.current
    if (!container) return
    const handleResize = () => {
      const Plotly = (window as any).Plotly
      if (Plotly?.Plots?.resize && container) Plotly.Plots.resize(container)
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [plotlyReady])

  const hasTop = topSegments.length > 0

  return (
    <div className="space-y-3 p-4 rounded-lg border">
      <h3 className="text-sm font-medium">Explainability (XAI)</h3>

      {hasTop && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">
            Top influential segments (from full recording):
          </p>
          {topSegments.slice(0, 3).map((seg, i) => (
            <div key={i} className="text-xs">
              <strong>{channelNames[seg.channel_idx] ?? `Ch${seg.channel_idx}`}</strong>
              {" "}{seg.start_time_sec.toFixed(2)}s–{seg.end_time_sec.toFixed(2)}s
              {seg.window_idx != null ? ` (W${seg.window_idx + 1})` : ""}
            </div>
          ))}
        </div>
      )}

      {hasEeg && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>uV/div:</span>
          {uVPerDivOptions.map((v) => (
            <button
              key={v}
              onClick={() => setUVPerDiv(v)}
              className={`rounded px-1.5 py-0.5 transition-colors ${
                uVPerDiv === v ? "bg-primary text-primary-foreground" : "bg-muted hover:text-foreground"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      )}

      {hasEeg && (
        <div ref={plotRef} className="w-full overflow-x-auto" style={{ minHeight: 280 }} />
      )}
    </div>
  )
}
