"use client"

import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

/* ================================================================
   ResultsMeta — compact row of key-value stats
   ================================================================ */

interface MetaItem {
  label: string
  value: string | number
}

export function ResultsMeta({ items }: { items: MetaItem[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5">
          <span className="text-muted-foreground">{item.label}:</span>
          <span className="font-medium text-foreground">{item.value}</span>
        </div>
      ))}
    </div>
  )
}

/* ================================================================
   DetectedRegions — badge list of epileptogenic regions
   ================================================================ */

interface DetectedRegionsProps {
  /** Full anatomical names, e.g. "rCCA (Right Cingulate Cortex Anterior)" */
  regions: string[]
  /** Whether to use destructive / warning colours (epileptogenic = true)  */
  variant?: "clinical" | "neutral"
}

export function DetectedRegions({
  regions,
  variant = "clinical",
}: DetectedRegionsProps) {
  if (regions.length === 0) {
    return (
      <Card className="p-6 text-center text-sm text-muted-foreground">
        No regions above threshold.
      </Card>
    )
  }

  return (
    <Card className="p-6">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">
        Detected Epileptogenic Regions
      </h3>
      <Separator className="mb-4" />
      <div className="flex flex-wrap gap-2">
        {regions.slice(0, 3).map((region) => (
          <Badge
            key={region}
            variant={variant === "clinical" ? "destructive" : "secondary"}
            className={`
              text-xs font-medium px-3 py-1
              ${variant === "clinical"
                ? "bg-red-100 text-red-800 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-800/40"
                : ""
              }
            `}
          >
            {region}
          </Badge>
        ))}
      </div>
      <p className="mt-4 text-xs text-muted-foreground">
        {regions.length > 3 
          ? `Top 3 of ${regions.length} identified regions shown`
          : `${regions.length} region${regions.length !== 1 ? "s" : ""} identified`}
        .
      </p>
    </Card>
  )
}
