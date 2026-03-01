"use client"

import { Check } from "lucide-react"

/**
 * Three-step workflow indicator: Upload → Analyze → Results.
 *
 * All three steps are always visible. Steps before `currentStep` show
 * a green check-mark; the current step is highlighted with primary colour;
 * future steps are muted.
 */

export type StepId = "upload" | "analyze" | "results"

interface StepDefinition {
  id: StepId
  label: string
  num: number
}

const steps: StepDefinition[] = [
  { id: "upload",  label: "Upload",  num: 1 },
  { id: "analyze", label: "Analyze", num: 2 },
  { id: "results", label: "Results", num: 3 },
]

interface StepIndicatorProps {
  current: StepId
}

export function StepIndicator({ current }: StepIndicatorProps) {
  const currentIdx = steps.findIndex((s) => s.id === current)

  return (
    <nav aria-label="Workflow steps" className="mb-8">
      <ol className="flex items-center justify-center gap-2">
        {steps.map((step, idx) => {
          const isCompleted = idx < currentIdx
          const isCurrent   = idx === currentIdx

          return (
            <li key={step.id} className="flex items-center gap-2">
              {/* Connector line before this step (skip first) */}
              {idx > 0 && (
                <div
                  className={`h-px w-10 ${
                    isCompleted ? "bg-primary" : "bg-border"
                  }`}
                />
              )}

              {/* Step circle */}
              <div
                className={`
                  flex h-7 w-7 shrink-0 items-center justify-center rounded-full
                  text-xs font-semibold transition-colors
                  ${isCompleted
                    ? "bg-primary text-primary-foreground"
                    : isCurrent
                      ? "border-2 border-primary text-primary"
                      : "border border-border text-muted-foreground"
                  }
                `}
                aria-current={isCurrent ? "step" : undefined}
              >
                {isCompleted ? <Check className="h-3.5 w-3.5" /> : step.num}
              </div>

              {/* Label */}
              <span
                className={`text-sm font-medium ${
                  isCurrent
                    ? "text-foreground"
                    : isCompleted
                      ? "text-primary"
                      : "text-muted-foreground"
                }`}
              >
                {step.label}
              </span>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
