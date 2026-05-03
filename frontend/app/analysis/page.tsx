"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { RotateCcw, Brain, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { AppHeader, AppContainer, AppFooter } from "@/components/app-shell"
import { StepIndicator, type StepId } from "@/components/step-indicator"
import { FileUploadSection } from "@/components/file-upload-section"
import { ProcessingWindow } from "@/components/processing-window"
import { BrainVisualization } from "@/components/brain-visualization"
import { EegWaveformPlot } from "@/components/eeg-waveform-plot"
import { ResultsMeta, DetectedRegions } from "@/components/results-summary"
import { ErrorAlert } from "@/components/error-alert"
import { ErrorBoundary } from "@/components/error-boundary"
import { AnalysisSkeleton } from "@/components/analysis-skeleton"
import { XaiPanel } from "@/components/xai-panel"
import { ConcordanceBadge } from "@/components/concordance-badge"
import type { PhysDeepSIFResult } from "@/lib/job-store"

const DEFAULT_CHANNEL_NAMES = [
  "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
  "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
  "Fz", "Cz", "Pz",
]

/* ---- Result types for both modes ---- */
interface EegWindowData {
  channels: string[]
  samplingRate: number
  windowLength: number
  windows: Array<{
    startTime: number
    endTime: number
    data: number[][]
  }>
}

interface AnimationFrameData {
  scoresArray: number[]
  timestamp: number
}

interface ESIResult {
  jobId: string
  fileName: string
  processingTime: number
  plotHtml: string
  nWindowsProcessed?: number
  nWindowsTotal?: number
  windowsTruncated?: boolean
  hasAnimation?: boolean
  eegData?: EegWindowData | null
  animationData?: AnimationFrameData[] | null
  vertexRegion?: number[] | null
}

type ViewMode = "source" | "biomarkers"

/**
 * /analysis — Unified EEG Analysis Dashboard
 *
 * Single page with shared upload + dual-mode results.
 * Workflow: Upload EEG → Process both modes → Switch between views.
 *
 * The backend runs inference twice (source_localization + biomarkers),
 * then the user can seamlessly tab between the two visualisations.
 */
export default function AnalysisPage() {
  const [step, setStep] = useState<StepId>("upload")
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>("source")
  const [maxWindows, setMaxWindows] = useState<number>(5)
  const [cmaesGens, setCmaesGens] = useState<number>(20)

  // Results for both modes
  const [esiResult, setEsiResult] = useState<ESIResult | null>(null)
  const [bioResult, setBioResult] = useState<PhysDeepSIFResult | null>(null)

  // Playback speed state — passed to BrainVisualization for Plotly control
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1)

  // EEG waveform window selection
  const [selectedWindow, setSelectedWindow] = useState<number>(0)

  /* ---- File selected from upload ---- */
  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file)
    setError(null)
  }, [])

  /* ---- Run analysis ---- */
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return
    setStep("analyze")
    setError(null)

    try {
      // Parallel requests: source_localization + biomarkers
      const fdSource = new FormData()
      fdSource.append("file", selectedFile)
      fdSource.append("include_eeg", "true")
      fdSource.append("max_windows", String(maxWindows))
      fdSource.append("cmaes_generations", String(cmaesGens))
      fdSource.append("xai_window_idx", String(selectedWindow))
      fdSource.append("ws", "false")
      fdSource.append("mode", "source_localization")

      const fdBio = new FormData()
      fdBio.append("file", selectedFile)
      fdBio.append("include_eeg", "true")
      fdBio.append("max_windows", String(maxWindows))
      fdBio.append("cmaes_generations", String(cmaesGens))
      fdBio.append("xai_window_idx", String(selectedWindow))
      fdBio.append("ws", "false")
      fdBio.append("mode", "biomarkers")

      const [sourceRes, bioRes] = await Promise.all([
        fetch("/api/analyze", { method: "POST", body: fdSource }),
        fetch("/api/analyze", { method: "POST", body: fdBio }),
      ])

      if (!sourceRes.ok) {
        const errData = await sourceRes.json().catch(() => ({ message: "Source localization failed" }))
        throw new Error(errData.detail || errData.message || `Source localization failed (${sourceRes.status})`)
      }
      if (!bioRes.ok) {
        const errData = await bioRes.json().catch(() => ({ message: "Biomarker analysis failed" }))
        throw new Error(errData.detail || errData.message || `Biomarker analysis failed (${bioRes.status})`)
      }

      const [sourceData, bioData] = await Promise.all([
        sourceRes.json(),
        bioRes.json(),
      ])

      console.log("[AnalysisPage] Source result:", {
        hasAnimation: sourceData?.hasAnimation,
        nWindowsProcessed: sourceData?.nWindowsProcessed,
        nWindowsTotal: sourceData?.nWindowsTotal,
        plotHtmlLen: sourceData?.plotHtml?.length,
        hasEegData: !!sourceData?.eegData,
        eegWindowsLen: sourceData?.eegData?.windows?.length,
        windowsTruncated: sourceData?.windowsTruncated,
      })
      console.log("[AnalysisPage] Bio result:", {
        hasEegData: !!bioData?.eegData,
        eegWindowsLen: bioData?.eegData?.windows?.length,
      })

      console.debug("[Page] ESIResult received:", {
        hasAnimation: sourceData?.hasAnimation,
        nWindows: sourceData?.nWindowsProcessed,
        animDataLen: sourceData?.animationData?.length,
        vertexRegionLen: sourceData?.vertexRegion?.length,
        firstFrameScores: sourceData?.animationData?.[0]?.scoresArray?.slice(0, 3),
        eegWindows: sourceData?.eegData?.windows?.length,
      })

      setEsiResult(sourceData)
      setBioResult(bioData)
      setStep("results")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during analysis")
      setStep("upload")
    }
  }, [selectedFile, maxWindows, cmaesGens])

  /* ---- Reset to upload state ---- */
  const handleReset = useCallback(() => {
    setStep("upload")
    setEsiResult(null)
    setBioResult(null)
    setError(null)
    setSelectedFile(null)
    setPlaybackSpeed(1)
    setSelectedWindow(0)
  }, [])

  /* ---- CMA-ES polling — when backend launches background CMA-ES, poll for results ---- */
  const cmaesPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => {
    if (!bioResult?.cmaes || bioResult.cmaes.status !== "running" || !bioResult.jobId) {
      return
    }

    const jobId = bioResult.jobId
    const poll = async () => {
      try {
        console.debug("[Page] CMA-ES polling", { jobId })
        const res = await fetch(`/api/job/${jobId}/cmaes`)
        if (!res.ok) { console.debug("[Page] CMA-ES poll not OK", res.status); return }
        const data = await res.json()
        console.debug("[Page] CMA-ES response:", data)
        if (data.status === "completed") {
          setBioResult((prev) => {
            if (!prev) return prev
            return {
              ...prev,
              concordance: data.concordance ?? prev.concordance,
              cmaes: data.cmaes ?? prev.cmaes,
            }
          })
        } else if (data.status === "running") {
          setBioResult((prev) => {
            if (!prev) return prev
            return {
              ...prev,
              cmaes: {
                status: "running",
                generations: data.generation ?? prev.cmaes?.generations ?? 0,
                max_generations: data.max_generations ?? prev.cmaes?.max_generations,
                best_score: data.best_score ?? prev.cmaes?.best_score,
              },
            }
          })
        } else if (data.status === "failed") {
          setBioResult((prev) => {
            if (!prev) return prev
            return {
              ...prev,
              cmaes: { status: "failed", error: data.cmaes?.error ?? "Unknown error" },
            }
          })
        }
      } catch {
        // Ignore transient network errors during polling
      }
    }

    poll() // immediate first poll
    cmaesPollRef.current = setInterval(poll, 2000)

    return () => {
      if (cmaesPollRef.current) {
        clearInterval(cmaesPollRef.current)
        cmaesPollRef.current = null
      }
    }
  }, [bioResult?.cmaes?.status, bioResult?.jobId])

  // Keep selectedWindow valid when result data changes.
  const setClampedWindow = useCallback(
    (value: number | ((prev: number) => number)) => {
      setSelectedWindow((prev) => {
        const total = esiResult?.eegData?.windows?.length ?? 0
        const raw = typeof value === "function" ? value(prev) : value
        if (total <= 0) return 0
        return Math.max(0, Math.min(raw, total - 1))
      })
    },
    [esiResult?.eegData?.windows?.length],
  )

  const handleBrainFrameChange = useCallback((frameIndex: number) => {
    setClampedWindow(frameIndex)
  }, [setClampedWindow])

  /* ---- Re-analyze XAI when selected window changes ---- */
  const xaiWindowRef = useRef<number>(0)
  useEffect(() => {
    if (!bioResult?.jobId || selectedWindow === xaiWindowRef.current) return
    xaiWindowRef.current = selectedWindow
    const jobId = bioResult.jobId
    const poll = async () => {
      try {
        const res = await fetch(`/api/xai/${jobId}/${selectedWindow}`)
        if (!res.ok) return
        const data = await res.json()
        if (data.status === "ok" && data.xai) {
          setBioResult((prev) => prev ? { ...prev, xai: data.xai } : prev)
        }
      } catch { /* ignore */ }
    }
    poll()
  }, [selectedWindow, bioResult?.jobId])

  /* ---- Derive detected regions from biomarker result ---- */
  const roiDetected = bioResult?.epileptogenicity?.roi_detected ?? true
  const detectedRegions: string[] =
    bioResult?.epileptogenicity?.regions_of_interest_full ??
    bioResult?.epileptogenicity?.epileptogenic_regions_full ??
    bioResult?.epileptogenicity?.epileptogenic_regions ??
    []

  /* ---- Active result data for metadata bar ---- */
  const activeProcessingTime =
    viewMode === "source"
      ? esiResult?.processingTime ?? 0
      : bioResult?.processingTime ?? 0

  return (
    <div className="dark flex min-h-screen flex-col bg-background text-foreground">
      <AppHeader />

      <main id="main-content" className="flex-1">
        <ErrorBoundary>
        <AppContainer>
          {/* Page header — only shown before results */}
          {step !== "results" && (
            <div className="mb-8">
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                EEG Analysis
              </h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Upload a 19-channel EEG recording for source localization and epileptogenic zone detection
              </p>
            </div>
          )}

          <StepIndicator current={step} />

          {/* ---- Error banner ---- */}
          {error && (
            <div className="mb-6">
              <ErrorAlert message={error} onRetry={handleReset} />
            </div>
          )}

          {/* ---- Step 1: Upload ---- */}
          {step === "upload" && (
            <div className="mx-auto max-w-lg space-y-6">
              <FileUploadSection
                onFileSelect={handleFileSelect}
                accept={[".edf"]}
                hint="19-channel EEG recording in EDF format"
              />

              <div className="flex items-center gap-3">
                <label className="text-sm text-muted-foreground whitespace-nowrap">
                  Max windows:
                </label>
                <Input
                  type="number"
                  min={1}
                  max={90}
                  value={maxWindows}
                  onChange={(e) => setMaxWindows(Number(e.target.value) || 1)}
                  className="w-24"
                />
                <span className="text-xs text-muted-foreground">
                  (first N windows, 2s each)
                </span>
              </div>

              <div className="flex items-center gap-3">
                <label className="text-sm text-muted-foreground whitespace-nowrap">
                  CMA-ES gens:
                </label>
                <Input
                  type="number"
                  min={1}
                  max={30}
                  value={cmaesGens}
                  onChange={(e) => setCmaesGens(Number(e.target.value) || 1)}
                  className="w-24"
                />
                <span className="text-xs text-muted-foreground">
                  (1-30, lower = faster)
                </span>
              </div>

              <Button
                className="w-full"
                size="lg"
                disabled={!selectedFile}
                onClick={handleAnalyze}
              >
                Analyze EEG
              </Button>
            </div>
          )}

          {/* ---- Step 2: Processing ---- */}
          {step === "analyze" && (
            <>
              <ProcessingWindow
                elapsedTime={0}
                progress={0}
                status={"Running inference..."}
              />
              <AnalysisSkeleton />
            </>
          )}

          {/* ---- Step 3: Results Dashboard ---- */}
          {step === "results" && (esiResult || bioResult) && (
            <div className="space-y-4 animate-fade-in">
              {/* Top bar: metadata + view toggle + reset */}
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <ResultsMeta
                  items={[
                    { label: "File", value: selectedFile?.name ?? "Unknown" },
                    { label: "Time", value: `${activeProcessingTime.toFixed(1)}s` },
                    ...(viewMode === "source" && esiResult?.nWindowsProcessed && esiResult.nWindowsProcessed > 1
                      ? [{ label: "Windows", value: esiResult.nWindowsProcessed }]
                      : []),
                  ]}
                />
                <Button variant="outline" size="sm" onClick={handleReset}>
                  <RotateCcw className="mr-2 h-3.5 w-3.5" />
                  New Analysis
                </Button>
              </div>

              {/* View mode toggle tabs */}
              <div className="flex gap-1 rounded-lg bg-muted/50 p-1" role="tablist">
                <button
                  onClick={() => setViewMode("source")}
                  role="tab"
                  aria-selected={viewMode === "source"}
                  className={`
                    flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2.5
                    text-sm font-medium transition-all
                    ${viewMode === "source"
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }
                  `}
                >
                  <Brain className="h-4 w-4" />
                  Source Localization
                </button>
                <button
                  onClick={() => setViewMode("biomarkers")}
                  role="tab"
                  aria-selected={viewMode === "biomarkers"}
                  className={`
                    flex flex-1 items-center justify-center gap-2 rounded-md px-4 py-2.5
                    text-sm font-medium transition-all
                    ${viewMode === "biomarkers"
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }
                  `}
                >
                  <Activity className="h-4 w-4" />
                  Biomarker Detection
                </button>
              </div>

              {/* Playback speed control — only shown when animation exists */}
              {viewMode === "source" && esiResult?.hasAnimation && esiResult.nWindowsProcessed && esiResult.nWindowsProcessed > 1 && (
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-muted-foreground">Speed:</span>
                  {[0.5, 1, 2, 4].map((speed) => (
                    <button
                      key={speed}
                      onClick={() => setPlaybackSpeed(speed)}
                      className={`
                        rounded-md px-3 py-1 text-xs font-medium transition-colors
                        ${playbackSpeed === speed
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover:text-foreground"
                        }
                      `}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
              )}

              {/* ---- Source Localization View ---- */}
              {viewMode === "source" && (esiResult || bioResult) && (
                <div className="grid w-full grid-cols-1 gap-4 lg:grid-cols-2">
                  {/* EEG Waveform */}
                  {(() => {
                    const eeg = esiResult?.eegData ?? bioResult?.eegData
                    if (!eeg) return null
                    return (
                      <EegWaveformPlot
                        eegData={eeg}
                        selectedWindow={selectedWindow}
                        onSelectedWindowChange={setClampedWindow}
                        className="w-full"
                      />
                    )
                  })()}

                  {/* 3D Brain Visualization */}
                  {(() => {
                    const html = esiResult?.plotHtml ?? bioResult?.plotHtml
                    if (!html) return null
                    const isSourceView = viewMode === "source"
                    const animData = isSourceView ? esiResult?.animationData ?? undefined : undefined
                    const vertReg = isSourceView ? esiResult?.vertexRegion ?? undefined : undefined
                    console.debug("[Page] Passing to BrainVisualization:", {
                      isSourceView, animLen: animData?.length, vertRegLen: vertReg?.length,
                      currentFrame: selectedWindow, htmlLen: html?.length,
                    })
                    return (
                      <BrainVisualization
                        plotHtml={html}
                        animationData={animData}
                        vertexRegion={vertReg}
                        label="Source Activity"
                        className="h-[640px] w-full"
                        playbackSpeed={playbackSpeed}
                        currentFrame={selectedWindow}
                        onFrameChange={handleBrainFrameChange}
                      />
                    )
                  })()}
                </div>
              )}

              {/* ---- Biomarker Detection View ---- */}
              {viewMode === "biomarkers" && bioResult && (
                <div className="grid w-full grid-cols-1 lg:grid-cols-[40%_60%] gap-4">
                  {/* Left column: Brain heatmap — always visible, sticky */}
                  <div className="lg:sticky lg:top-4 self-start">
                    <BrainVisualization
                      plotHtml={bioResult.plotHtml}
                      label="Epileptogenicity Map"
                      className="h-[500px] w-full"
                      currentFrame={selectedWindow}
                      onFrameChange={handleBrainFrameChange}
                    />
                  </div>

                  {/* Right column: results stacked vertically */}
                  <div className="space-y-4">
                    {/* Section 1: Detected Regions */}
                    <Card>
                      <div className="px-4 py-2 border-b">
                        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Epileptogenic Regions</span>
                      </div>
                      <div className="p-4">
                        <DetectedRegions regions={detectedRegions} variant="clinical" />
                      </div>
                    </Card>

                    {/* Section 2: CMA-ES Biophysical Validation */}
                    {bioResult?.cmaes && (
                      <Card>
                        <div className="px-4 py-2 border-b">
                          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Biophysical Validation</span>
                        </div>
                        <div className="p-4 space-y-3">
                          {bioResult.cmaes.status === "running" && (
                            <div className="space-y-2">
                              <div className="flex items-center gap-2 text-sm">
                                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                                <span>CMA-ES biophysical validation in progress...</span>
                              </div>
                              {bioResult.cmaes.generations != null && bioResult.cmaes.max_generations && (
                                <div className="space-y-1">
                                  <div className="flex justify-between text-xs text-muted-foreground">
                                    <span>Generation {bioResult.cmaes.generations} / {bioResult.cmaes.max_generations}</span>
                                  </div>
                                  <div className="h-2 bg-muted rounded overflow-hidden">
                                    <div
                                      className="h-full bg-primary transition-all duration-500"
                                      style={{ width: `${((bioResult.cmaes.generations ?? 0) / (bioResult.cmaes.max_generations || 1)) * 100}%` }}
                                    />
                                  </div>
                                </div>
                              )}
                              <p className="text-xs text-muted-foreground">
                                Simulates neural propagation across brain regions using The Virtual Brain to verify whether detected activity patterns are physiologically plausible.
                              </p>
                            </div>
                          )}
                          {bioResult.cmaes.status === "completed" && bioResult.concordance && (
                            <ConcordanceBadge
                              tier={bioResult.concordance.tier}
                              overlap={bioResult.concordance.overlap}
                              description={bioResult.concordance.tier_description}
                              sharedRegions={bioResult.concordance.shared_regions}
                            />
                          )}
                          {bioResult.cmaes.status === "completed" && !bioResult.concordance && (
                            <p className="text-xs text-muted-foreground">Validation complete — no concordance data available.</p>
                          )}
                          {bioResult.cmaes.status === "failed" && (
                            <p className="text-xs text-red-400">Biophysical validation failed: {bioResult.cmaes.error ?? "Unknown error"}</p>
                          )}
                          {bioResult.cmaes.status === "debug_skip" && (
                            <p className="text-xs text-muted-foreground">Debug mode — biophysical validation skipped.</p>
                          )}
                        </div>
                      </Card>
                    )}

                    {/* Section 3: XAI Panel */}
                    {bioResult?.xai && (
                      <Card>
                        <div className="px-4 py-2 border-b">
                          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Explainability (XAI)</span>
                        </div>
                        <div className="p-4">
                          <XaiPanel
                            channelImportance={bioResult.xai.channel_importance ?? []}
                            timeImportance={bioResult.xai.time_importance ?? []}
                            channelNames={esiResult?.eegData?.channels ?? bioResult?.eegData?.channels ?? DEFAULT_CHANNEL_NAMES}
                            topSegments={bioResult.xai.top_segments ?? []}
                            eegData={esiResult?.eegData ?? bioResult?.eegData ?? undefined}
                            selectedWindow={selectedWindow}
                          />
                        </div>
                      </Card>
                    )}
                  </div>
                </div>
              )}

              {/* EEG Waveform — full width below the above grid, collapsible */}
              {viewMode === "biomarkers" && bioResult?.eegData && (
                <details className="mt-4 group">
                  <summary className="text-xs font-medium uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-foreground">
                    EEG Waveform {(() => { const n = esiResult?.eegData?.windows?.length; if (n && n > 1) return `(${n} windows)`; return null })()}
                  </summary>
                  <div className="mt-3">
                    <EegWaveformPlot
                      eegData={bioResult.eegData}
                      selectedWindow={selectedWindow}
                      onSelectedWindowChange={setClampedWindow}
                      className="h-[400px]"
                    />
                  </div>
                </details>
              )}

            </div>
          )}
        </AppContainer>
        </ErrorBoundary>
      </main>

      <AppFooter />
    </div>
  )
}
