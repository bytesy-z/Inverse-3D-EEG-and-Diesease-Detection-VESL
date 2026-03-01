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
}: BrainVisualizationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [loading, setLoading] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)

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
        resizePlotly(container)
        setLoading(false)
      }
    }

    run().catch(console.error)

    return () => {
      cancelled = true
      container.innerHTML = ""
    }
  }, [plotHtml])

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
    const container = containerRef.current
    if (!container || loading) return

    // Access the Plotly global and the graph div
    const Plotly = (window as Window & { Plotly?: Record<string, unknown> }).Plotly as
      | { relayout?: (el: HTMLElement, update: Record<string, unknown>) => void }
      | undefined
    const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
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

function resizePlotly(container: HTMLElement) {
  const plotDiv = container.querySelector<HTMLElement>(".plotly-graph-div")
  const Plotly = (window as Window & { Plotly?: { Plots?: { resize: (el: HTMLElement) => void } } }).Plotly
  if (plotDiv && Plotly?.Plots?.resize) {
    plotDiv.style.width = "100%"
    plotDiv.style.height = "100%"
    Plotly.Plots.resize(plotDiv)
  }
}
