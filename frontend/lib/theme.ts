/**
 * Design tokens for the VESL application.
 *
 * Single source of truth for spacing, colours, and layout
 * constants used across both dark (ESI) and light (Biomarkers) themes.
 * Import these tokens instead of using magic numbers in components.
 */

// ---- Accent palette (sage-green family, oklch h≈160) ----
export const accent = {
  DEFAULT: "var(--primary)",          // oklch(0.52 0.12 160)
  light:   "var(--chart-4)",          // oklch(0.65 0.15 160) – CTA bg
  dark:    "var(--secondary)",        // oklch(0.35 0.06 160)
  muted:   "var(--accent)",           // oklch(0.52 0.12 160)
} as const

// ---- Layout constants ----
export const layout = {
  /** Max-width of the main content container */
  maxWidth:    "max-w-6xl",
  /** Standard horizontal padding */
  px:          "px-6",
  /** Vertical page padding */
  py:          "py-8",
  /** Card border radius */
  radius:      "rounded-xl",
  /** Consistent gap between major sections */
  sectionGap:  "space-y-8",
  /** Gap inside cards */
  cardPadding: "p-6",
} as const

// ---- Visualisation canvas ----
export const canvas = {
  /** Default height for the brain visualisation card */
  height:     "h-[640px]",
  /** Minimum height to prevent squishing */
  minHeight:  "min-h-[480px]",
  /** Background for ESI (dark) mode canvas */
  bgDark:     "bg-[#0f1117]",
  /** Background for Biomarkers (light) mode canvas */
  bgLight:    "bg-white",
} as const

// ---- Animation ----
export const animation = {
  fadeIn: "animate-in fade-in duration-300",
  slideUp: "animate-in slide-in-from-bottom-2 duration-300",
} as const
