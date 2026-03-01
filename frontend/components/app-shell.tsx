"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Brain } from "lucide-react"

/**
 * Shared application header with a single navigation tab.
 *
 * The "Analyze EEG" tab leads to the unified analysis dashboard
 * at /analysis, where users upload one EEG file and switch between
 * Source Localization and Biomarker Detection views.
 */

interface NavTab {
  href: string
  label: string
  icon: React.ReactNode
}

const tabs: NavTab[] = [
  {
    href: "/analysis",
    label: "Analyze EEG",
    icon: <Brain className="w-4 h-4" />,
  },
]

export function AppHeader() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-card/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        {/* Brand */}
        <Link
          href="/"
          className="text-xl font-bold tracking-tight text-primary"
          aria-label="VESL home"
        >
          VESL
        </Link>

        {/* Mode tabs */}
        <nav className="flex items-center gap-1" aria-label="Primary navigation">
          {tabs.map((tab) => {
            const isActive = pathname === tab.href
            return (
              <Link
                key={tab.href}
                href={tab.href}
                className={`
                  inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium
                  transition-colors focus-visible:outline-2 focus-visible:outline-offset-2
                  focus-visible:outline-ring
                  ${isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }
                `}
                aria-current={isActive ? "page" : undefined}
              >
                {tab.icon}
                {tab.label}
              </Link>
            )
          })}
        </nav>
      </div>
    </header>
  )
}

/**
 * Wraps page content in a max-width container with consistent padding.
 */
export function AppContainer({
  children,
  className = "",
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={`mx-auto w-full max-w-6xl px-6 py-8 ${className}`}>
      {children}
    </div>
  )
}

/**
 * Standard page title block with heading + optional subtitle.
 */
export function PageTitle({
  title,
  subtitle,
}: {
  title: string
  subtitle?: string
}) {
  return (
    <div className="mb-8">
      <h1 className="text-2xl font-semibold tracking-tight text-foreground">
        {title}
      </h1>
      {subtitle && (
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      )}
    </div>
  )
}

/**
 * Footer with team credits.
 */
export function AppFooter() {
  return (
    <footer className="mt-auto border-t border-border/40 py-6">
      <div className="mx-auto max-w-6xl px-6 text-center text-xs text-muted-foreground">
        <p>&copy; {new Date().getFullYear()} VESL &mdash; Virtual Epileptogenic Source Localizer</p>
        <p className="mt-1">Hira Sardar &middot; Muhammad Zikrullah Rehman &middot; Shahliza Ahmad</p>
      </div>
    </footer>
  )
}
