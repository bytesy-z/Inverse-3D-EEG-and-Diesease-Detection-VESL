"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Loader2, Check } from "lucide-react"

interface ProcessingWindowProps {
  /** Real elapsed time from the parent */
  elapsedTime?: number
  /** Real progress percentage (0-100) from the parent */
  progress?: number
  /** Status message from the parent */
  status?: string
}

const pipelineSteps = [
  { label: "Validating file format",       threshold: 1 },
  { label: "Loading signal data",          threshold: 3 },
  { label: "Preprocessing EEG signals",    threshold: 8 },
  { label: "Running source localization",  threshold: 20 },
  { label: "Generating 3D brain map",      threshold: 40 },
]

/**
 * Compact processing card shown while inference is running.
 *
 * Shows:
 *  - Elapsed timer (mono font)
 *  - A smooth progress bar
 *  - Pipeline step checklist that ticks off over time
 */
export function ProcessingWindow({ elapsedTime = 0, progress = 0, status = "" }: ProcessingWindowProps) {
  const [elapsed, setElapsed] = useState(0)

  // Internal timer fallback when no elapsedTime is provided
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed((e) => e + 1)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const displayElapsed = elapsedTime > 0 ? elapsedTime : elapsed
  const displayProgress = progress

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  return (
    <Card className="mx-auto max-w-md p-8" aria-live="polite">
      {/* Spinner + title */}
      <div className="flex items-center justify-center gap-3 mb-6">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
        <h2 className="text-lg font-semibold text-foreground">
          Processing EEG
        </h2>
      </div>

      {/* Timer */}
      <p className="text-center text-2xl font-mono text-primary tabular-nums mb-6">
        {fmt(displayElapsed)}
      </p>

      {/* Progress bar */}
      <Progress value={displayProgress} className="mb-6 h-1.5" />

      {status && (
        <p className="text-center text-sm text-muted-foreground mb-4">{status}</p>
      )}

      {/* Pipeline steps */}
      <ul className="space-y-2">
        {pipelineSteps.map((step) => {
          const done = displayElapsed >= step.threshold
          return (
            <li key={step.label} className="flex items-center gap-2 text-sm">
              {done ? (
                <Check className="h-4 w-4 text-primary" />
              ) : (
                <span className="flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] text-muted-foreground">
                  &bull;
                </span>
              )}
              <span className={done ? "text-foreground" : "text-muted-foreground"}>
                {step.label}
              </span>
            </li>
          )
        })}
      </ul>

      <p className="mt-6 text-center text-xs text-muted-foreground">
        This page will update automatically when processing completes.
      </p>
    </Card>
  )
}
