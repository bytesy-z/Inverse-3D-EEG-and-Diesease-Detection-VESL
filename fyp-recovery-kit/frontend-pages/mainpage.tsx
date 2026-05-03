"use client"

import Link from "next/link"
import { Brain, Zap, Globe, Eye, ArrowRight } from "lucide-react"
import { Card } from "@/components/ui/card"
import { AppHeader, AppFooter } from "@/components/app-shell"

/**
 * Landing page — problem statement, solution benefits, and CTA.
 *
 * Draws content from the original VESL landing page but uses a
 * cleaner, more clinical tone. Dark theme throughout.
 */

/* ---- Content data ---- */

const keyPoints = [
  {
    icon: <Brain className="h-6 w-6 text-primary" />,
    title: "A Growing Crisis",
    body: "Neurological disorders — epilepsy, Alzheimer's, Parkinson's — affect nearly one in three people worldwide, yet diagnosis often depends on expensive imaging like MRI or PET.",
  },
  {
    icon: <Zap className="h-6 w-6 text-primary" />,
    title: "Why EEG?",
    body: "Electroencephalography is low-cost, portable, and non-invasive. It captures brain activity in real-time, but conventional analysis cannot pinpoint the 3D source of abnormal signals.",
  },
  {
    icon: <Eye className="h-6 w-6 text-primary" />,
    title: "Our Approach",
    body: "VESL uses a physics-informed neural network trained on biophysically realistic simulations to reconstruct 3D brain source activity from standard 19-channel EEG recordings.",
  },
  {
    icon: <Globe className="h-6 w-6 text-primary" />,
    title: "Accessible Diagnosis",
    body: "By replacing costly imaging with EEG-based source localization, VESL brings early neurological diagnosis within reach of clinics worldwide — including resource-limited settings.",
  },
]

const capabilities = [
  {
    title: "3D Source Localization",
    body: "Visualize time-resolved brain source activity mapped onto a realistic cortical surface.",
  },
  {
    title: "Epileptogenic Zone Detection",
    body: "Identify the brain regions most likely driving epileptic activity, highlighted with clinical confidence.",
  },
  {
    title: "Standard EEG Input",
    body: "Works with standard 10-20 montage EDF recordings — no specialized hardware required.",
  },
  {
    title: "Physics-Informed Model",
    body: "Grounded in biophysical forward modeling (Epileptor neural mass model + BEM leadfield), not black-box pattern matching.",
  },
]

export default function MainPage() {
  return (
    <div className="dark flex min-h-screen flex-col bg-background text-foreground">
      <AppHeader />

      <main className="flex-1">
        {/* ---- Hero Section ---- */}
        <section className="mx-auto flex max-w-4xl flex-col items-center px-6 pt-24 pb-16 text-center">
          <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">
            Inverse EEG-Based 3D Brain Source Localization
          </h1>
          <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
            Affordable, real-time 3D brain activity reconstruction with
            explainable biomarker visualization for early and accessible
            neurological diagnosis.
          </p>
          <Link
            href="/analysis"
            className="mt-8 inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Start Analysis
            <ArrowRight className="h-4 w-4" />
          </Link>
        </section>

        {/* ---- Problem & Solution (Key Points) ---- */}
        <section className="mx-auto max-w-5xl px-6 py-12">
          <h2 className="mb-8 text-center text-2xl font-semibold tracking-tight">
            The Problem &amp; Our Solution
          </h2>
          <div className="grid gap-6 sm:grid-cols-2">
            {keyPoints.map((kp, i) => (
              <Card
                key={i}
                className="flex flex-col gap-3 p-6 transition-colors hover:border-primary/30"
              >
                {kp.icon}
                <h3 className="text-lg font-semibold">{kp.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {kp.body}
                </p>
              </Card>
            ))}
          </div>
        </section>

        {/* ---- Capabilities ---- */}
        <section className="mx-auto max-w-5xl px-6 py-12">
          <h2 className="mb-8 text-center text-2xl font-semibold tracking-tight">
            What VESL Does
          </h2>
          <div className="grid gap-6 sm:grid-cols-2">
            {capabilities.map((cap, i) => (
              <Card
                key={i}
                className="flex flex-col gap-2 p-6 transition-colors hover:border-primary/30"
              >
                <h3 className="font-semibold">{cap.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {cap.body}
                </p>
              </Card>
            ))}
          </div>
        </section>

        {/* ---- CTA Banner ---- */}
        <section className="mx-auto max-w-3xl px-6 py-16 text-center">
          <h2 className="text-xl font-semibold tracking-tight">
            Ready to analyze an EEG recording?
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Upload a standard 19-channel EDF file and get source localization
            plus epileptogenic zone detection in seconds.
          </p>
          <Link
            href="/analysis"
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Go to Analysis
            <ArrowRight className="h-4 w-4" />
          </Link>
        </section>
      </main>

      <AppFooter />
    </div>
  )
}
