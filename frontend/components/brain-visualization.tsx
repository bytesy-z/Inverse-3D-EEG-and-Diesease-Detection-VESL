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
 *   - Wraps Plotly.animate to catch all promise rejections from native buttons.
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
  // Track whether animation is currently playing (to avoid clashing relayout calls)
  const isAnimatingRef = useRef(false)

  /* ---- Inject & execute the Plotly HTML ---- */
  useEffect(() => {
    const container = containerRef.current
    if (!container || !plotHtml) {
      console.log("[BrainVis] Skipping injection: no container or plotHtml", { hasContainer: !!container, hasHtml: !!plotHtml })
      return
    }

    let cancelled = false
    setLoading(true)
    console.log("[BrainVis] Starting Plotly HTML injection")

    const run = async () => {
      // Clear previous render
      container.innerHTML = ""
      console.log("[BrainVis] Cleared container")

      // Parse HTML string
      const wrapper = document.createElement("div")
      wrapper.innerHTML = plotHtml

      // Separate script tags (they won't execute via innerHTML)
      const scripts = Array.from(wrapper.querySelectorAll("script"))
      scripts.forEach((s) => s.parentNode?.removeChild(s))
      console.log(`[BrainVis] Found ${scripts.length} script(s) in HTML`, scripts.map(s => s.src || "<inline>"))

      // Append non-script content
      while (wrapper.firstChild) {
        container.appendChild(wrapper.firstChild)
      }

      // Execute scripts sequentially
      for (const [i, oldScript] of scripts.entries()) {
        if (cancelled) break
        const src = oldScript.src || "<inline>"
        console.log(`[BrainVis] Executing script ${i}/${scripts.length}: ${src}`)
        await new Promise<void>((resolve, reject) => {
          const newScript = document.createElement("script")
          Array.from(oldScript.attributes).forEach((a) =>
            newScript.setAttribute(a.name, a.value)
          )
          if (oldScript.src) {
            newScript.onload = () => {
              console.log(`[BrainVis] Script loaded OK: ${oldScript.src}`)
              resolve()
            }
            newScript.onerror = () => {
              console.error(`[BrainVis] Script load FAILED: ${oldScript.src}`)
              reject(new Error(`Failed to load: ${oldScript.src}`))
            }
            newScript.src = oldScript.src
          } else {
            newScript.textContent = oldScript.textContent
            container.appendChild(newScript)
            console.log(`[BrainVis] Inline script ${i} executed`)
            return resolve()
          }
          container.appendChild(newScript)
        })
      }

      if (!cancelled) {
        const plotDiv = container.querySelector<PlotlyGraphDiv>(".plotly-graph-div")
        plotDivRef.current = plotDiv ?? null
        console.log("[BrainVis] plotDiv found:", !!plotDiv)

        // Monkey-patch Plotly.relayout to catch promise rejections (used for playback speed)
        const Plotly = (window as any).Plotly
        console.log("[BrainVis] Plotly global available:", !!Plotly, !!Plotly?.animate, !!Plotly?.relayout)

        if (Plotly?.animate) {
          const origAnimate = Plotly.animate.bind(Plotly)
          Plotly.animate = (...args: unknown[]) => {
            console.log(`[BrainVis] Plotly.animate called with args:`, JSON.stringify(args.map(a => {
              if (a === null) return "null"
              if (Array.isArray(a)) return `[${a.join(",")}]`
              if (typeof a === "object") return "{...}"
              return String(a)
            })))
            const result = origAnimate(...args)
            if (result && typeof result.catch === "function") {
              // Swallow on original promise (don't return the caught promise)
              // Plotly's internal promise queue tracks the original promise reference.
              // If we return result.catch(()=>{}), Plotly's queue gets a different promise
              // and the original rejects unhandled internally.
              result.catch((err: unknown) => {
                console.error("[BrainVis] Plotly.animate rejected (swallowed):", err)
              })
              return result
            }
            console.log("[BrainVis] Plotly.animate returned non-promise:", result)
            return result
          }
        }

        if (Plotly?.relayout) {
          const origRelayout = Plotly.relayout.bind(Plotly)
          Plotly.relayout = (...args: unknown[]) => {
            const result = origRelayout(...args) as unknown
            if (result && typeof (result as Promise<unknown>).catch === "function") {
              (result as Promise<unknown>).catch((err: unknown) => {
                console.error("[BrainVis] Plotly.relayout rejected (swallowed):", err)
              })
              return result
            }
            return result
          }
        }


  // Module-level unhandledrejection handler for Plotly rejections.
  // Registered once globally (outside component lifecycle) so React Strict Mode
  // double-mounting cannot create gaps in coverage.
  // Initialised lazily on first BrainVisualization mount.
  if (typeof window !== "undefined" && !(window as any).__brainvis_rejection_handler) {
    const handler = (event: PromiseRejectionEvent) => {
      const reason = event.reason
      if (
        reason === undefined ||
        reason === null ||
        (typeof reason === "object" && reason !== null &&
         "message" in reason && typeof (reason as any).message === "string" &&
         (reason as any).message.includes("animate"))
      ) {
        event.preventDefault()
      }
    }
    window.addEventListener("unhandledrejection", handler)
    ;(window as any).__brainvis_rejection_handler = handler
  }


        resizePlotly(container)
        setLoading(false)

        // Debug: compare frame intensities to check if they differ
        if (plotDivRef.current) {
          const gd = plotDivRef.current as any
          const frames = gd._fullLayout?._frames || gd.layout?.frames || gd.frames || []
          const n = frames.length
          if (n > 1) {
            const frameMeans: number[] = []
            for (let i = 0; i < n; i++) {
              const f = frames[i]
              const intensity = f?.data?.[0]?.intensity ||
                f?.traces?.[0]?.intensity ||
                f?.trace?.intensity
              if (intensity) {
                let arr: number[] = []
                if (Array.isArray(intensity)) {
                  arr = intensity
                } else if (Array.isArray((intensity as any)?.[0])) {
                  arr = (intensity as any)[0]
                }
                if (arr.length > 0) {
                  const sum = arr.reduce((a: number, b: number) => a + b, 0)
                  frameMeans.push(sum / arr.length)
                }
              }
            }
            if (frameMeans.length > 1) {
              const uniqueMeans = new Set(frameMeans.map(m => m.toFixed(6)))
              console.log(
                `[BrainVis] FRAME CHECK: ${n} frames, ${frameMeans.length} with intensity, ${uniqueMeans.size} unique means, ` +
                `range=[${Math.min(...frameMeans).toFixed(6)}, ${Math.max(...frameMeans).toFixed(6)}]`
              )
              if (uniqueMeans.size <= 1) {
                console.warn("[BrainVis] ALL FRAMES HAVE IDENTICAL INTENSITY — animation will appear static!")
              }
            }
          } else {
            console.log(`[BrainVis] FRAME CHECK: ${n} frames (static plot or frames not found)`)
          }
        }

        console.log("[BrainVis] Injection complete, loading=false")
      } else {
        console.log("[BrainVis] Injection cancelled mid-way")
      }
    }

    run().catch((err) => {
      console.error("[BrainVis] Injection run failed:", err)
    })

    return () => {
      console.log("[BrainVis] Cleanup: cancelling injection and clearing container")
      cancelled = true
      // Stop Plotly animation timer before clearing DOM
      isAnimatingRef.current = false
      if (plotDivRef.current) {
        const Plotly = (window as any).Plotly
        console.log("[BrainVis] Cleanup: stopping animation")
        Plotly?.animate?.(plotDivRef.current, [null], { mode: "immediate" }).catch(() => {})
      }
      plotDivRef.current = null
      container.innerHTML = ""
      console.log("[BrainVis] Cleanup complete")
    }
  }, [plotHtml])

  /* ---- Bidirectional frame sync with parent ---- */
  useEffect(() => {
    const plotDiv = plotDivRef.current
    if (!plotDiv || !onFrameChange) {
      console.log("[BrainVis] Frame sync: skipping (no plotDiv or no onFrameChange)", { hasDiv: !!plotDiv, hasFn: !!onFrameChange })
      return
    }
    console.log("[BrainVis] Frame sync: attaching event listeners")

    const emitFrameChange = (source: string, frameIndex: number | null) => {
      if (frameIndex === null) {
        console.log(`[BrainVis] emitFrameChange(${source}): frameIndex is null, ignoring`)
        return
      }
      console.log(`[BrainVis] emitFrameChange(${source}): frameIndex=${frameIndex}, isProgrammaticSeek=${isProgrammaticSeekRef.current}, lastNotified=${lastNotifiedFrameRef.current}, suppress=${suppressFrameEventRef.current}`)
      lastKnownFrameRef.current = frameIndex

      if (isProgrammaticSeekRef.current) {
        console.log(`[BrainVis] emitFrameChange(${source}): ignoring during programmatic seek`)
        return
      }
      if (suppressFrameEventRef.current === frameIndex) {
        console.log(`[BrainVis] emitFrameChange(${source}): suppressing frame ${frameIndex} (echo cancel)`)
        suppressFrameEventRef.current = null
        return
      }

      if (lastNotifiedFrameRef.current === frameIndex) {
        console.log(`[BrainVis] emitFrameChange(${source}): frame ${frameIndex} already notified, dedup`)
        return
      }
      lastNotifiedFrameRef.current = frameIndex
      emittedFramesRef.current.add(frameIndex)
      console.log(`[BrainVis] emitFrameChange(${source}): CALLING onFrameChange(${frameIndex})`)
      onFrameChange(frameIndex)
    }

    const animTimeoutRef = { current: null as ReturnType<typeof setTimeout> | null }

    const handleAnimatingFrame = (eventData: unknown) => {
      const idx = extractFrameIndex(eventData)
      console.log(`[BrainVis] Event: plotly_animatingframe, extracted=${idx}`)
      // Mark animation as active; clear when frames stop arriving
      isAnimatingRef.current = true
      if (animTimeoutRef.current) clearTimeout(animTimeoutRef.current)
      animTimeoutRef.current = setTimeout(() => {
        isAnimatingRef.current = false
      }, 600)
      emitFrameChange("animatingframe", idx)
    }

    const handleSliderChange = (eventData: unknown) => {
      if (isProgrammaticSeekRef.current) {
        console.log("[BrainVis] Event: plotly_sliderchange (ignored, programmatic seek)")
        return
      }
      const idx = extractFrameIndex(eventData)
      console.log(`[BrainVis] Event: plotly_sliderchange, extracted=${idx}`)
      emitFrameChange("sliderchange", idx)
    }

    plotDiv.on?.("plotly_animatingframe", handleAnimatingFrame)
    plotDiv.on?.("plotly_sliderchange", handleSliderChange)
    console.log("[BrainVis] Frame sync: listeners attached")

    return () => {
      console.log("[BrainVis] Frame sync: removing event listeners")
      if (animTimeoutRef.current) clearTimeout(animTimeoutRef.current)
      isAnimatingRef.current = false
      plotDiv.removeListener?.("plotly_animatingframe", handleAnimatingFrame)
      plotDiv.removeListener?.("plotly_sliderchange", handleSliderChange)
    }
  }, [loading, onFrameChange])

  useEffect(() => {
    // Programmatic seek from parent; jump brain to given frame without looping
    if (currentFrame === undefined || currentFrame < 0) {
      console.log(`[BrainVis] Seek: skipping (currentFrame=${currentFrame})`)
      return
    }
    console.log(`[BrainVis] Seek: attempt frame=${currentFrame}, lastKnown=${lastKnownFrameRef.current}, emittedHas=${emittedFramesRef.current.has(currentFrame)}`)

    // Skip seek if figure hasn't reported any frame yet (not fully initialized)
    if (lastKnownFrameRef.current === null) {
      console.log(`[BrainVis] Seek: skipped (figure not initialized yet, no frame seen)`)
      return
    }

    if (lastKnownFrameRef.current === currentFrame) {
      console.log(`[BrainVis] Seek: skipped (already at frame ${currentFrame})`)
      return
    }

    // Echo cancellation: if this currentFrame is one we recently emitted to the parent, ignore it!
    if (emittedFramesRef.current.has(currentFrame)) {
      console.log(`[BrainVis] Seek: echo cancellation for frame ${currentFrame}`)
      emittedFramesRef.current.delete(currentFrame)
      return
    }
    // Genuine external seek; clear the emitted history to prevent stale matches
    emittedFramesRef.current.clear()

    const plotDiv = plotDivRef.current
    if (!plotDiv) {
      console.log("[BrainVis] Seek: no plotDiv, aborting")
      return
    }
    const Plotly = (window as any).Plotly
    if (!Plotly?.animate) {
      console.log("[BrainVis] Seek: Plotly.animate not available")
      return
    }
    isProgrammaticSeekRef.current = true
    lastKnownFrameRef.current = currentFrame
    console.log(`[BrainVis] Seek: executing Plotly.animate to frame ${currentFrame}`)
    // Pass plain string (not array) — Plotly 3.3.1 single-frame seek requires string
    const seekResult = Plotly.animate(plotDiv, String(currentFrame), {
      mode: "immediate",
      transition: { duration: 0 },
      frame: { duration: 0, redraw: true },
    })
    if (seekResult && typeof seekResult.catch === "function") {
      seekResult.then(() => {
        console.log(`[BrainVis] Seek: completed to frame ${currentFrame}`)
      }).catch((err: unknown) => {
        console.error(`[BrainVis] Seek: rejected for frame ${currentFrame}:`, err)
      }).finally(() => {
        console.log(`[BrainVis] Seek: finally, resetting programmaticSeek flag`)
        isProgrammaticSeekRef.current = false
      })
    } else {
      isProgrammaticSeekRef.current = false
    }
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

  /* ---- Update playback speed by rewriting Plotly animation button args ---- */
  useEffect(() => {
    const plotDiv = plotDivRef.current
    if (!plotDiv || loading) {
      console.log(`[BrainVis] Speed change: skipped (plotDiv=${!!plotDiv}, loading=${loading})`)
      return
    }

    const Plotly = (window as Window & { Plotly?: PlotlyClientAPI }).Plotly
    if (!Plotly?.relayout) {
      console.log("[BrainVis] Speed change: Plotly.relayout not available")
      return
    }

    const frameDuration = Math.round(200 / playbackSpeed)
    console.log(`[BrainVis] Speed change: playbackSpeed=${playbackSpeed} -> frameDuration=${frameDuration}ms, isAnimating=${isAnimatingRef.current}`)

    // Skip relayout while animation is playing — avoids mid-animation
    // promise rejections that can corrupt Plotly's internal state.
    if (isAnimatingRef.current) {
      console.log("[BrainVis] Speed change: deferred (animation in progress)")
      return
    }

    // Check if plot has updatemenus (animation buttons); skip if static plot
    const fullLayout = (plotDiv as any)?._fullLayout
    if (!fullLayout?._updatemenus || fullLayout._updatemenus.length === 0) {
      console.log("[BrainVis] Speed change: skipped (static plot, no animation controls)")
      return
    }

    try {
      const result = Plotly.relayout(plotDiv, {
        "updatemenus[0].buttons[0].args[1].frame.duration": frameDuration,
      }) as Promise<unknown> | undefined
      if (result && typeof result.catch === "function") {
        result.then(() => {
          console.log(`[BrainVis] Speed change: relayout completed`)
        }).catch((err: unknown) => {
          console.error(`[BrainVis] Speed change: relayout rejected:`, err)
        })
      }
    } catch (err) {
      console.error("[BrainVis] Speed change: relayout threw synchronously:", err)
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
  relayout?: (el: HTMLElement, update: Record<string, unknown>) => Promise<unknown> | void
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
  if (!eventData || typeof eventData !== "object") {
    console.log("[BrainVis] extractFrameIndex: invalid eventData", eventData)
    return null
  }

  const data = eventData as {
    name?: unknown
    step?: { label?: unknown; args?: unknown[] }
  }

  // Prefer explicit frame name like 'frame12' or '12' (which Plotly uses natively in Play)
  const nameVal = typeof data.name === 'string' ? data.name : null
  if (nameVal) {
    const n = parseFrameIndex(nameVal)
    if (n !== null) {
      console.log(`[BrainVis] extractFrameIndex: from name="${nameVal}" -> ${n}`)
      return n
    }
    console.log(`[BrainVis] extractFrameIndex: name="${nameVal}" did not parse as frame index`)
  }

  // Look into step args for a numeric frame index
  const args = Array.isArray(data.step?.args) ? data.step!.args! : []
  console.log(`[BrainVis] extractFrameIndex: step.args.length=${args.length}`)
  for (const a of args) {
    if (typeof a === 'number') {
      console.log(`[BrainVis] extractFrameIndex: from numeric arg ${a}`)
      return a
    }
    if (Array.isArray(a) && a.length > 0 && typeof a[0] === 'string') {
      const n = parseFrameIndex(a[0])
      if (n !== null) {
        console.log(`[BrainVis] extractFrameIndex: from arg[0]="${a[0]}" -> ${n}`)
        return n
      }
    }
    if (typeof a === 'string') {
      const n = parseFrameIndex(a)
      if (n !== null) {
        console.log(`[BrainVis] extractFrameIndex: from string arg "${a}" -> ${n}`)
        return n
      }
    }
  }

  // Fallback to a direct numeric name in possible nested fields
  const frameName = (data as any).frame?.name
  if (typeof frameName === 'string') {
    const n = parseFrameIndex(frameName)
    if (n !== null) {
      console.log(`[BrainVis] extractFrameIndex: from frame.name="${frameName}" -> ${n}`)
      return n
    }
  }

  console.log("[BrainVis] extractFrameIndex: FAILED to extract frame index")
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
