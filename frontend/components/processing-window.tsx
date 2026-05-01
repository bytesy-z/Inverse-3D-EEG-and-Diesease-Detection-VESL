"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Loader2 } from "lucide-react"

interface ProcessingWindowProps {
  /** Real elapsed time from the parent */
  elapsedTime?: number
  /** Real progress percentage (0-100) from the parent */
  progress?: number
  /** Status message from the parent */
  status?: string
}

/** Compact processing card shown while inference is running. */
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


    </Card>
  )
}
