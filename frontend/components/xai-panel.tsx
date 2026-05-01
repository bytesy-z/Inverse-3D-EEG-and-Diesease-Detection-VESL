"use client"

interface XaiPanelProps {
  channelImportance: number[]
  timeImportance: number[]
  channelNames: string[]
  onToggleOverlay: () => void
  showOverlay: boolean
}

export function XaiPanel({ channelImportance, timeImportance, channelNames }: XaiPanelProps) {
  const topChanIdx = channelImportance.length > 0
    ? channelImportance.indexOf(Math.max(...channelImportance))
    : -1
  const topTimeIdx = timeImportance.length > 0
    ? timeImportance.indexOf(Math.max(...timeImportance))
    : -1
  const topChan = topChanIdx >= 0 && topChanIdx < channelNames.length ? channelNames[topChanIdx] : "N/A"
  const topTimeMs = topTimeIdx >= 0 ? Math.round(topTimeIdx * 0.2 * 100) : 0

  return (
    <div className="space-y-3 p-4 rounded-lg border">
      <h3 className="text-sm font-medium">Explainability (XAI)</h3>
      <p className="text-sm text-muted-foreground">
        Most influential channel: <span className="font-semibold text-foreground">{topChan}</span>
      </p>
      <p className="text-sm text-muted-foreground">
        Most important time segment: <span className="font-semibold text-foreground">~{topTimeMs}ms</span>
      </p>
      <p className="text-xs text-muted-foreground">
        (Window contributed most to the epileptogenicity classification)
      </p>
    </div>
  )
}
