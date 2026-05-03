"use client"

import { useEffect, useRef, useState, useCallback } from "react"

interface JobStatus {
  status: string
  progress: number
  message: string
  result?: any
  cmaes_phase?: string
  cmaes_generation?: number
  cmaes_max_generations?: number
  cmaes_best_score?: number
}

interface UseWebSocketReturn {
  status: JobStatus | null
  connected: boolean
  phaseAComplete: boolean
  cmaesRunning: boolean
  cmaesProgress: { current: number; max: number; bestScore: number | null }
  result: any
  error: string | null
}

export function useWebSocket(jobId: string | null): UseWebSocketReturn {
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    queueMicrotask(() => setConnected(false))
  }, [])

  useEffect(() => {
    if (!jobId) {
      cleanup()
      return
    }

    let closed = false
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host =
      process.env.NEXT_PUBLIC_PHYSDEEPSIF_BACKEND_WS ||
      (window.location.host.endsWith(":3000") ? window.location.host.replace(":3000", ":8000") : window.location.host)
    const url = `${protocol}//${host}/ws/${jobId}`

    function connect() {
      if (closed) return
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)

      ws.onmessage = (event) => {
        try {
          const data: JobStatus = JSON.parse(event.data)
          setStatus(data)
          if (data.status === "completed" || data.status === "failed") {
            ws.close()
            setConnected(false)
          }
        } catch {
          // ignore parse errors
        }
      }

      ws.onerror = () => {
        setError("WebSocket connection error")
        setConnected(false)
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
      }
    }

    connect()

    return () => {
      closed = true
      cleanup()
    }
  }, [jobId, cleanup])

  const terminalStatus = status?.status === "completed" || status?.status === "failed"
  const phaseAComplete = status?.status === "phase_a_complete" || terminalStatus
  const cmaesRunning = status?.cmaes_phase === "running" || status?.cmaes_phase === "queued"
  const result = status?.result ?? null

  const cmaesProgress = {
    current: status?.cmaes_generation ?? 0,
    max: status?.cmaes_max_generations ?? 30,
    bestScore: status?.cmaes_best_score ?? null,
  }

  return { status, connected, phaseAComplete, cmaesRunning, cmaesProgress, result, error }
}
