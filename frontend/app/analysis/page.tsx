"use client"

import { useState, useCallback, useEffect } from "react"
import { RotateCcw, Brain, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { AppHeader, AppContainer, AppFooter } from "@/components/app-shell"
import { StepIndicator, type StepId } from "@/components/step-indicator"
import { FileUploadSection } from "@/components/file-upload-section"
import { ProcessingWindow } from "@/components/processing-window"
import { BrainVisualization } from "@/components/brain-visualization"
import { EegWaveformPlot } from "@/components/eeg-waveform-plot"
import { ResultsMeta, DetectedRegions } from "@/components/results-summary"
import { ErrorAlert } from "@/components/error-alert"
import type { PhysDeepSIFResult } from "@/lib/job-store"

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

interface ESIResult {
  fileName: string
  processingTime: number
  plotHtml: string
  nWindowsProcessed?: number
  nWindowsTotal?: number
  windowsTruncated?: boolean
  hasAnimation?: boolean
  eegData?: EegWindowData | null
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

  /* ---- Run both analyses in parallel ---- */
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return
    setStep("analyze")
    setError(null)

    try {
      // Build form data for each request
      const makeForm = (mode: string) => {
        const fd = new FormData()
        fd.append("file", selectedFile)
        if (mode === "source_localization") {
          fd.append("include_eeg", "true")
        }
        if (mode === "biomarkers") {
          fd.append("threshold_percentile", "87.5")
          // Avoid duplicating huge EEG window payload in the second request.
          // Biomarker view reuses the EEG data from source-localization result.
          fd.append("include_eeg", "false")
        }
        return fd
      }

      // Run requests sequentially to avoid memory spikes on large EDF uploads.
      const esiRes = await fetch("/api/analyze-eeg", {
        method: "POST",
        body: makeForm("source_localization"),
      })

      // Check for errors on either response
      if (!esiRes.ok) {
        const errData = await esiRes.json().catch(() => ({ message: "ESI request failed" }))
        throw new Error(errData.message || `Source localization failed (${esiRes.status})`)
      }
      const esiData = await esiRes.json()
      const bioRes = await fetch("/api/physdeepsif", {
        method: "POST",
        body: makeForm("biomarkers"),
      })
      if (!bioRes.ok) {
        const errData = await bioRes.json().catch(() => ({ message: "Biomarker request failed" }))
        throw new Error(errData.message || `Biomarker detection failed (${bioRes.status})`)
      }
      const bioData: PhysDeepSIFResult = await bioRes.json()

      setEsiResult({
        fileName: selectedFile.name,
        processingTime: esiData.processingTime || 0,
        plotHtml: esiData.plotHtml,
        nWindowsProcessed: esiData.nWindowsProcessed,
        nWindowsTotal: esiData.nWindowsTotal,
        windowsTruncated: esiData.windowsTruncated,
        hasAnimation: esiData.hasAnimation,
        eegData: esiData.eegData ?? null,
      })
      setBioResult(bioData)
      setViewMode("source") // Default to source localization view
      setSelectedWindow(0)
      setStep("results")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during analysis")
      setStep("upload")
    }
  }, [selectedFile])

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

  const handleBrainFrameChange = useCallback((frameIndex: number) => {
    const total = esiResult?.eegData?.windows?.length ?? 0
    if (total <= 0) return
    const normalizedIndex = ((frameIndex % total) + total) % total
    setSelectedWindow(normalizedIndex)
  }, [esiResult?.eegData?.windows?.length])

  // Keep selectedWindow valid when result data changes.
  useEffect(() => {
    const total = esiResult?.eegData?.windows?.length ?? 0
    if (total <= 0) {
      if (selectedWindow !== 0) {
        setSelectedWindow(0)
      }
      return
    }
    if (selectedWindow >= total) {
      setSelectedWindow(total - 1)
    }
  }, [esiResult?.eegData?.windows?.length, selectedWindow])

  /* ---- Derive detected regions from biomarker result ---- */
  const detectedRegions: string[] =
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

      <main className="flex-1">
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
          {step === "analyze" && <ProcessingWindow />}

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
              <div className="flex gap-1 rounded-lg bg-muted/50 p-1">
                <button
                  onClick={() => setViewMode("source")}
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

              {/* EEG window is synchronized with brain animation via shared selectedWindow state. */}
              {/* ---- Source Localization View (layout adjusted) ---- */}
              {viewMode === "source" && esiResult && (
                <div className="grid w-full grid-cols-1 gap-4 lg:grid-cols-2">
                  {/* EEG Waveform */}
                  {esiResult.eegData && (
                    <EegWaveformPlot
                      eegData={esiResult.eegData}
                      selectedWindow={selectedWindow}
                      onSelectedWindowChange={setSelectedWindow}
                      className="w-full"
                    />
                  )}

                  {/* 3D Brain Visualization (guarded by presence of plotHtml) */}
                  {esiResult.plotHtml && (
                    <BrainVisualization
                      plotHtml={esiResult.plotHtml}
                      label="3D Source Activity"
                      className="h-[640px] w-full"
                      playbackSpeed={playbackSpeed}
                      currentFrame={selectedWindow}
                      onFrameChange={handleBrainFrameChange}
                    />
                  )}
                </div>
              )}

              {/* ---- Biomarker Detection View ---- */}
              {viewMode === "biomarkers" && bioResult && (
                <div className="space-y-4">
                  {/* Window selector for biomarker view */}
                  {(esiResult?.eegData?.windows && esiResult.eegData.windows.length > 1) && (
                    <div className="flex items-center gap-3 text-sm">
                      <span className="text-muted-foreground">Window:</span>
                      <div className="flex gap-1">
                        {esiResult.eegData.windows.map((_, idx) => (
                          <button
                            key={idx}
                            onClick={() => setSelectedWindow(idx)}
                            className={`
                              rounded-md px-3 py-1 text-xs font-medium transition-colors
                              ${selectedWindow === idx
                                ? "bg-primary text-primary-foreground"
                                : "bg-muted text-muted-foreground hover:text-foreground"
                              }
                            `}
                          >
                            {idx + 1}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* EEG Waveform */}
                  {esiResult?.eegData && (
                    <EegWaveformPlot
                      eegData={esiResult.eegData}
                      selectedWindow={selectedWindow}
                      onSelectedWindowChange={setSelectedWindow}
                      className="h-[500px]"
                    />
                  )}

                  <BrainVisualization
                    plotHtml={bioResult.plotHtml}
                    label="Epileptogenicity Map"
                    className="h-[640px]"
                    currentFrame={selectedWindow}
                    onFrameChange={handleBrainFrameChange}
                  />

                  <DetectedRegions
                    regions={detectedRegions}
                    variant="clinical"
                  />
                </div>
              )}
            </div>
          )}
        </AppContainer>
      </main>

      <AppFooter />
    </div>
  )
}
