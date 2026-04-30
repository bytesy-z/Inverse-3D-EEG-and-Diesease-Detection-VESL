"use client"

interface XaiPanelProps {
  channelImportance: number[]
  timeImportance: number[]
  channelNames: string[]
  onToggleOverlay: () => void
  showOverlay: boolean
}

export function XaiPanel({ channelImportance, timeImportance, channelNames, onToggleOverlay, showOverlay }: XaiPanelProps) {
  const maxChan = Math.max(...channelImportance.map(Math.abs), 1)
  const maxTime = Math.max(...timeImportance.map(Math.abs), 1)

  return (
    <div className="space-y-4 p-4 rounded-lg border">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Occlusion Attribution</h3>
        <button
          onClick={onToggleOverlay}
          className={`px-3 py-1 text-xs rounded-md border transition-colors ${
            showOverlay ? "bg-primary text-primary-foreground" : "bg-background"
          }`}
        >
          {showOverlay ? "Hide Overlay" : "Show Overlay"}
        </button>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-2">Channel Importance</p>
        <div className="space-y-1">
          {channelImportance.map((imp, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs w-8 text-right font-mono">{channelNames[i]}</span>
              <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.abs(imp) / maxChan * 100}%`,
                    backgroundColor: imp >= 0 ? "hsl(142, 76%, 36%)" : "hsl(0, 84%, 60%)",
                  }}
                />
              </div>
              <span className="text-xs w-12 text-right font-mono text-muted-foreground">
                {imp.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-2">Time Segment Importance</p>
        <div className="flex items-end gap-px h-16">
          {timeImportance.map((imp, i) => (
            <div
              key={i}
              className="flex-1 rounded-t-sm transition-all"
              style={{
                height: `${Math.abs(imp) / maxTime * 100}%`,
                backgroundColor: imp >= 0 ? "hsl(142, 76%, 36%)" : "hsl(0, 84%, 60%)",
                opacity: 0.7 + 0.3 * (Math.abs(imp) / maxTime),
              }}
            />
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Most influential channels/segments: {channelNames[channelImportance.indexOf(Math.max(...channelImportance))]} at ~{Math.round(timeImportance.indexOf(Math.max(...timeImportance)) * 0.2 * 100)}ms
        </p>
      </div>
    </div>
  )
}
