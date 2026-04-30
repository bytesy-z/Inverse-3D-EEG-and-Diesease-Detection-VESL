"use client"

import { AlertTriangle, CheckCircle, HelpCircle } from "lucide-react"

interface ConcordanceBadgeProps {
  tier: string | null
  overlap?: number
  description?: string
  sharedRegions?: string[]
}

export function ConcordanceBadge({ tier, overlap, description, sharedRegions }: ConcordanceBadgeProps) {
  if (!tier) {
    return (
      <div className="rounded-lg border border-border bg-muted/50 p-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <HelpCircle className="h-4 w-4" />
          <span className="font-medium">Concordance: Pending</span>
        </div>
        <p className="mt-1 text-xs">Biophysical validation not yet available.</p>
      </div>
    )
  }

  const tierConfig = {
    HIGH: {
      icon: CheckCircle,
      color: "text-green-500",
      border: "border-green-500/30",
      bg: "bg-green-500/5",
      label: "HIGH CONCORDANCE",
    },
    MODERATE: {
      icon: AlertTriangle,
      color: "text-yellow-500",
      border: "border-yellow-500/30",
      bg: "bg-yellow-500/5",
      label: "MODERATE CONCORDANCE",
    },
    LOW: {
      icon: AlertTriangle,
      color: "text-red-500",
      border: "border-red-500/30",
      bg: "bg-red-500/5",
      label: "LOW CONCORDANCE",
    },
  }[tier] ?? {
    icon: HelpCircle,
    color: "text-muted-foreground",
    border: "border-border",
    bg: "bg-muted/50",
    label: "UNKNOWN",
  }

  const Icon = tierConfig.icon

  return (
    <div className={`rounded-lg border ${tierConfig.border} ${tierConfig.bg} p-4`}>
      <div className="flex items-center gap-2">
        <Icon className={`h-5 w-5 ${tierConfig.color}`} />
        <span className={`text-sm font-semibold ${tierConfig.color}`}>{tierConfig.label}</span>
        {overlap !== undefined && (
          <span className="ml-auto text-xs text-muted-foreground">
            Overlap: {overlap}/10
          </span>
        )}
      </div>
      {description && (
        <p className="mt-2 text-xs text-muted-foreground">{description}</p>
      )}
      {sharedRegions && sharedRegions.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
            Shared regions ({sharedRegions.length})
          </summary>
          <ul className="mt-1 space-y-0.5">
            {sharedRegions.map((r) => (
              <li key={r} className="text-xs text-muted-foreground pl-4">
                {r}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
