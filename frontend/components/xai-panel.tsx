"use client"

import { useEffect, useRef, useState } from "react"

interface XaiSegment {
  channel_idx: number
  start_time_sec: number
  end_time_sec: number
  importance: number
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
  const topChannels = channelImportance
    .map((val, idx) => ({ idx, val }))
    .sort((a, b) => b.val - a.val)
  const maxVal = topChannels.length > 0 ? topChannels[0].val : 1

  const plotRef = useRef<HTMLDivElement>(null)
  const [plotlyReady, setPlotlyReady] = useState(false)

  const hasEeg = eegData && eegData.windows.length > 0

  // Dynamically load Plotly if not already available via window.Plotly
  useEffect(() => {
    if (typeof window !== "undefined" && (window as any).Plotly) {
      setPlotlyReady(true)
      return
    }
    const script = document.createElement("script")
    script.src = "https://cdn.plot.ly/plotly-2.35.3.min.js"
    script.onload = () => setPlotlyReady(true)
    script.onerror = () => console.error("Failed to load Plotly")
    document.head.appendChild(script)
  }, [])

  // Render EEG waveform chart
  useEffect(() => {
    if (!plotlyReady || !hasEeg || !eegData || !plotRef.current) return

    let cancelled = false
    const Plotly = (window as any).Plotly
    if (!Plotly?.newPlot) return

    const winIdx = Math.min(selectedWindow, eegData.windows.length - 1)
    const win = eegData.windows[winIdx]
    if (!win || !win.data.length) return

    const numChannels = eegData.channels.length
    const numSamples = win.data[0]?.length ?? 0
    if (numSamples === 0) return

    const timeAxis: number[] = []
    for (let i = 0; i < numSamples; i++) {
      timeAxis.push(win.startTime + i / eegData.samplingRate)
    }

    // Compute offset for vertically stacked channels
    let globalMax = 0
    for (let ch = 0; ch < numChannels; ch++) {
      const d = win.data[ch]
      if (!d) continue
      for (let s = 0; s < d.length; s++) {
        const abs = Math.abs(d[s])
        if (abs > globalMax) globalMax = abs
      }
    }
    if (globalMax === 0) globalMax = 1
    const channelOffset = globalMax * 3

    // Build one trace per channel (no smoothing, linear line shape)
    const traces: Record<string, unknown>[] = []
    for (let ch = 0; ch < numChannels; ch++) {
      const yOffset = (numChannels - 1 - ch) * channelOffset
      const raw = win.data[ch]
      if (!raw) continue
      const yValues = raw.map((v) => v + yOffset)
      traces.push({
        x: timeAxis,
        y: yValues,
        type: "scatter",
        mode: "lines",
        name: eegData.channels[ch],
        line: {
          color: "#60a5fa",
          width: 0.8,
          shape: "linear",
        },
        hoverinfo: "name+x+y",
        showlegend: false,
        hovertemplate: "%{y:.2f}",
      })
    }

    // Build occlusion-segment highlight shapes
    const shapes: Record<string, unknown>[] = topSegments
      .filter((seg) => seg.channel_idx >= 0 && seg.channel_idx < numChannels)
      .map((seg) => {
        const yCenter = (numChannels - 1 - seg.channel_idx) * channelOffset
        const halfBand = channelOffset * 0.48
        return {
          type: "rect",
          x0: seg.start_time_sec,
          x1: seg.end_time_sec,
          y0: yCenter - halfBand,
          y1: yCenter + halfBand,
          fillcolor: "rgba(16, 185, 129, 0.15)",
          line: { width: 0 },
          layer: "below",
        }
      })

    const layout: Record<string, unknown> = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#a1a1aa", size: 10 },
      height: 380,
      margin: { l: 58, r: 10, t: 5, b: 30 },
      showlegend: false,
      hovermode: "x unified",
      dragmode: "pan",
      xaxis: {
        title: { text: "Time (s)", font: { color: "#a1a1aa", size: 10 } },
        tickfont: { color: "#a1a1aa", size: 9 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
      },
      yaxis: {
        tickvals: eegData.channels.map(
          (_, i) => (numChannels - 1 - i) * channelOffset
        ),
        ticktext: eegData.channels,
        tickfont: { color: "#a1a1aa", size: 7 },
        gridcolor: "rgba(255,255,255,0.06)",
        zeroline: false,
        showgrid: true,
        range: [
          -channelOffset * 0.7,
          (numChannels - 0.3) * channelOffset,
        ],
        fixedrange: false,
      },
      shapes,
    }

    const config: Record<string, unknown> = {
      responsive: true,
      displayModeBar: false,
      scrollZoom: false,
    }

    Plotly.purge(plotRef.current)
    Plotly.newPlot(plotRef.current, traces, layout, config).catch(() => {})

    return () => {
      cancelled = true
      if (plotRef.current) {
        Plotly.purge(plotRef.current)
      }
    }
  }, [plotlyReady, hasEeg, eegData, selectedWindow, topSegments])

  // Resize Plotly on window resize
  useEffect(() => {
    if (!plotlyReady) return
    const container = plotRef.current
    if (!container) return

    const handleResize = () => {
      const Plotly = (window as any).Plotly
      if (Plotly?.Plots?.resize && container) {
        Plotly.Plots.resize(container)
      }
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [plotlyReady])

  return (
    <div className="space-y-3 p-4 rounded-lg border">
      <h3 className="text-sm font-medium">Explainability (XAI)</h3>

      {topSegments.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Top influential segments:</p>
          {topSegments.slice(0, 3).map((seg, i) => (
            <div key={i} className="text-xs">
              <strong>{channelNames[seg.channel_idx] ?? `Ch${seg.channel_idx}`}</strong>
              {" "}{seg.start_time_sec.toFixed(2)}s–{seg.end_time_sec.toFixed(2)}s
              {" "}(&Delta;score: {seg.importance.toFixed(4)})
            </div>
          ))}
        </div>
      )}

      <div className="space-y-1 mt-3">
        <p className="text-xs text-muted-foreground">Top channels by importance:</p>
        {topChannels.slice(0, 5).map((ch) => {
          const pct = maxVal > 0 ? (ch.val / maxVal) * 100 : 0
          return (
            <div key={ch.idx} className="flex items-center gap-2 text-xs">
              <span className="w-8 text-right font-mono">{channelNames[ch.idx] ?? `Ch${ch.idx}`}</span>
              <div className="flex-1 h-2 bg-muted rounded">
                <div
                  className="h-full bg-emerald-500 rounded"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-12 text-right font-mono text-muted-foreground">{ch.val.toFixed(3)}</span>
            </div>
          )
        })}
      </div>

      {hasEeg && (
        <div
          ref={plotRef}
          className="w-full overflow-x-auto"
          style={{ minHeight: 380 }}
        />
      )}
    </div>
  )
}
