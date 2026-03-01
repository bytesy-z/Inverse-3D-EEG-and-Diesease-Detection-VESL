import { redirect } from "next/navigation"

/**
 * Legacy route — redirects to the unified analysis dashboard.
 */
export default function EEGSourceLocalizationRedirect() {
  redirect("/analysis")
}
