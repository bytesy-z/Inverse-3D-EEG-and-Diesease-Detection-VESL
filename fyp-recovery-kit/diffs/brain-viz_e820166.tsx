"use client"

import { useEffect, useRef, useState } from "react"
import { Maximize2, Minimize2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

interface BrainVisualizationProps {
  /** Raw Plotly HTML string returned by the backend  */
  plotHtml: string
  /** Optional label shown above the card              */
  label?: string
  /** Extra Tailwind classes for the outer card        */
  className?: string
  /** Playback speed multiplier (0.5, 1, 2, 4) for animation */
  playbackSpeed?: number
  /** Controlled current frame index for external sync */
  currentFrame?: number
  /**
   * Called whenever the Plotly animation advances to a new frame.
   * frameIndex is the 0-based index of the frame that just started playing.
   * Use this to synchronize external UI (e.g. EEG waveform window) with the animation.
   */
  onFrameChange?: (frameIndex: number) => void
}

/**
 * Renders the Plotly 3D brain visualisation inside a Card.
 *
 * Key behaviours:
 *   - Injects the Plotly HTML into a container div (NOT an iframe)
 *     so that Plotly's native play/pause and slider controls work directly.
 *   - Shows a skeleton loader until scripts finish loading.
 *   - Provides a full-screen toggle button.
 *   - Adds the `.plotly-container` CSS class so the global stylesheet
 *     can enforce 100% width/height on the inner Plotly elements.
 *   - Resizes Plotly on window resize and full-screen toggle.
 */
export function BrainVisualization({
  plotHtml,
  label,
  className = "",
  playbackSpeed = 1,
  currentFrame,
  onFrameChange,
}: BrainVisualizationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotDivRef = useRef<PlotlyGraphDiv | null>(null)
  const suppressFrameEventRef = useRef<number | null>(null)
  const lastNotifiedFrameRef = useRef<number | null>(null)
  const lastKnownFrameRef = useRef<number | null>(null)
  const emittedFramesRef = useRef<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  // Guard against feedback from programmatic seeks
  const isProgrammaticSeekRef = useRef(false)

  /* ---- Inject & execute the Plotly HTML ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container || !plotHtml) return

    let cancelled = false
    setLoading(true)

    const run = async () => {
      // Clear previous render
      container.innerHTML = ""

      // Parse HTML string
      const wrapper = document.createElement("div")
      wrapper.innerHTML = plotHtml

      // Separate script tags (they won't execute via innerHTML)
      const scripts = Array.from(wrapper.querySelectorAll("script"))
      scripts.forEach((s) => s.parentNode?.removeChild(s))

      // Append non-script content
      while (wrapper.firstChild) {
        container.appendChild(wrapper.firstChild)
      }

      // Execute scripts sequentially
      for (const oldScript of scripts) {
        if (cancelled) break
        await new Promise<void>((resolve, reject) => {
          const newScript = document.createElement("script")
          Array.from(oldScript.attributes).forEach((a) =>
            newScript.setAttribute(a.name, a.value)
          )
          if (oldScript.src) {
            newScript.onload = () => resolve()
            newScript.onerror = () =>
              reject(new Error(`Failed to load: ${oldScript.src}`))
            newScript.src = oldScript.src
          } else {
            newScript.textContent = oldScript.textContent
            container.appendChild(newScript)
            return resolve()
          }
          container.appendChild(newScript)
        })
      }

      if (!cancelled) {
        const plotDiv = container.querySelector<PlotlyGraphDiv>(".plotly-graph-div")
        plotDivRef.current = plotDiv ?? null

        resizePlotly(container)
        setLoading(false)
      }
    }

    run().catch(console.error)

    return () => {
      cancelled = true
      plotDivRef.current = null
      container.innerHTML = ""
    }
  }, [plotHtml])

  /* ---- Bidirectional frame sync with parent ---- */
  useEffect(() => {
    const plotDiv = plotDivRef.current
    if (!plotDiv || !onFrameChange) return

    const emitFrameChange = (frameIndex: number | null) => {
      if (frameIndex === null) return
      lastKnownFrameRef.current = frameIndex

      if (isProgrammaticSeekRef.current) {
        // ignore events coming from programmatic seeks
        return
      }
      if (suppressFrameEventRef.current === frameIndex) {
        suppressFrameEventRef.current = null
        return
      }

      if (lastNotifiedFrameRef.current === frameIndex) return
      lastNotifiedFrameRef.current = frameIndex
      emittedFramesRef.current.add(frameIndex)
      onFrameChange(frameIndex)
    }

    const handleAnimatingFrame = (eventData: unknown) => {
      emitFrameChange(extractFrameIndex(eventData))
    }

    const handleSliderChange = (eventData: unknown) => {
      // Avoid acting on programmatic seeks to prevent loops
      if (isProgrammaticSeekRef.current) return
      emitFrameChange(extractFrameIndex(eventData))
    }

    plotDiv.on?.("plotly_animatingframe", handleAnimatingFrame)
    plotDiv.on?.("plotly_sliderchange", handleSliderChange)

    return () => {
      plotDiv.removeListener?.("plotly_animatingframe", handleAnimatingFrame)
      plotDiv.removeListener?.("plotly_sliderchange", handleSliderChange)
    }
  }, [loading, onFrameChange])

  useEffect(() => {
    // Programmatic seek from parent; jump brain to given frame without looping
    if (currentFrame === undefined || currentFrame < 0) return
    if (lastKnownFrameRef.current === currentFrame) return // Prevent feedback loop during native playback

    // Echo cancellation: if this currentFrame is one we recently emitted to the parent, ignore it!
    if (emittedFramesRef.current.has(currentFrame)) {
      emittedFramesRef.current.delete(currentFrame)
      return
    }
    // Genuine external seek; clear the emitted history to prevent stale matches
    emittedFramesRef.current.clear()

    const plotDiv = plotDivRef.current
    if (!plotDiv) return
    const Plotly = (window as any).Plotly
    if (!Plotly?.animate) return
    isProgrammaticSeekRef.current = true
    lastKnownFrameRef.current = currentFrame
    Plotly.animate(plotDiv, [String(currentFrame)], {
      mode: "immediate",
      transition: { duration: 0 },
      frame: { duration: 0, redraw: true },
    }).catch(() => {
      // swallow
    }).finally(() => {
      // allow subsequent user interactions to emit events
      isProgrammaticSeekRef.current = false
    })
  }, [currentFrame, loading])

  /* ---- Resize handler ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const handleResize = () => resizePlotly(container)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  /* ---- Fullscreen toggle ---- */
  const toggleFullscreen = () => {
    const el = containerRef.current?.parentElement
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => {
        setIsFullscreen(true)
        // Resize Plotly after entering fullscreen so it fills the screen
        setTimeout(() => {
          if (containerRef.current) resizePlotly(containerRef.current)
        }, 200)
      })
    } else {
      document.exitFullscreen?.().then(() => {
        setIsFullscreen(false)
        setTimeout(() => {
          if (containerRef.current) resizePlotly(containerRef.current)
        }, 200)
      })
    }
  }

  useEffect(() => {
    const handleFs = () => {
      const fs = !!document.fullscreenElement
      setIsFullscreen(fs)
      // Resize on fullscreen state change
      setTimeout(() => {
        if (containerRef.current) resizePlotly(containerRef.current)
      }, 200)
    }
    document.addEventListener("fullscreenchange", handleFs)
    return () => document.removeEventListener("fullscreenchange", handleFs)
  }, [])

  /* ---- Update playback speed by rewriting Plotly animation buttons ---- */
  useEffect(() => {
    const plotDiv = plotDivRef.current
    if (!plotDiv || loading) return

    // Access the Plotly global and the graph div
    const Plotly = (window as Window & { Plotly?: PlotlyClientAPI }).Plotly
    if (!Plotly?.relayout || !plotDiv) return

    // The base frame duration is 200ms — adjust by speed multiplier
    const frameDuration = Math.round(200 / playbackSpeed)

    // Update the Play button args to use the new frame duration
    // Plotly stores updatemenus in layout — we update via relayout
    try {
      Plotly.relayout(plotDiv, {
        "updatemenus[0].buttons[0].args[1].frame.duration": frameDuration,
      })
    } catch {
      // Silently ignore if figure has no animation updatemenus
    }
  }, [playbackSpeed, loading])

  return (
    <Card
      className={`relative overflow-hidden ${isFullscreen ? "!rounded-none !border-none" : ""} ${className}`}
    >
      {/* Optional heading */}
      {label && (
        <div className="border-b border-border px-4 py-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
        </div>
      )}

      {/* Fullscreen button */}
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

      {/* Skeleton loader shown while scripts are loading */}
      {loading && (
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-card">
          <Skeleton className="h-full w-full rounded-none" />
        </div>
      )}

      {/* Plotly container — fills available space */}
      <div
        ref={containerRef}
        className="plotly-container w-full"
        style={{ height: isFullscreen ? "calc(100vh - 40px)" : undefined, minHeight: isFullscreen ? "100%" : 480 }}
      />
    </Card>
  )
}

/* ---- Helpers ---- */

type PlotlyGraphDiv = HTMLElement & {
  on?: (event: string, cb: (eventData: unknown) => void) => void
  removeListener?: (event: string, cb: (eventData: unknown) => void) => void
}

type PlotlyClientAPI = {
  relayout?: (el: HTMLElement, update: Record<string, unknown>) => void
  animate?: (
    el: HTMLElement,
    frameOrGroupNameOrFrameList?: unknown,
    animationOpts?: {
      mode?: "immediate" | "next" | "afterall"
      transition?: { duration?: number }
      frame?: { duration?: number; redraw?: boolean }
    },
  ) => Promise<unknown>
  Plots?: { resize: (el: HTMLElement) => void }
}

function extractFrameIndex(eventData: unknown): number | null {
  if (!eventData || typeof eventData !== "object") return null

  const data = eventData as {
    name?: unknown
    step?: { label?: unknown; args?: unknown[] }
  }

  // Prefer explicit frame name like 'frame12' or '12' (which Plotly uses natively in Play)
  const nameVal = typeof data.name === 'string' ? data.name : null
  if (nameVal) {
    const n = parseFrameIndex(nameVal)
    if (n !== null) return n
  }

  // Look into step args for a numeric frame index
  // Note: Plotly slider passes an array in step.args where args[0] might be an array of frame names
  // e.g. step: { args: [["12"], { ... }] }
  const args = Array.isArray(data.step?.args) ? data.step!.args! : []
  for (const a of args) {
    if (typeof a === 'number') return a
    if (Array.isArray(a) && a.length > 0 && typeof a[0] === 'string') {
      const n = parseFrameIndex(a[0])
      if (n !== null) return n
    }
    if (typeof a === 'string') {
      const n = parseFrameIndex(a)
      if (n !== null) return n
    }
  }

  // Fallback to a direct numeric name in possible nested fields
  const frameName = (data as any).frame?.name
  if (typeof frameName === 'string') {
    const n = parseFrameIndex(frameName)
    if (n !== null) return n
  }

  return null
}

function parseFrameIndex(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value !== "string") return null
  // Avoid parsing things like "0:20.0" as frame 0 or frame 20.
  // Frames are passed as exactly the stringified integer, e.g., "12".
  if (/^\d+$/.test(value)) return parseInt(value, 10)
  
  // If it's something like "frame12", match that:
  const match = value.match(/frame(\d+)$/i)
  if (match) return parseInt(match[1], 10)

  return null
}

function resizePlotly(container: HTMLElement) {
  const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
  const Plotly = (window as Window & { Plotly?: PlotlyClientAPI }).Plotly
  if (plotDiv && Plotly?.Plots?.resize) {
    plotDiv.style.width = "100%"
    plotDiv.style.height = "100%"
    Plotly.Plots.resize(plotDiv)
  }
}
