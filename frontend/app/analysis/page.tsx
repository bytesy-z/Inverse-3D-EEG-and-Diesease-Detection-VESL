"use client"

import { useState, useCallback, useEffect } from "react"
import { RotateCcw, Brain, Activity, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
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
import { useWebSocket } from "@/hooks/use-websocket"
import { ConcordanceBadge } from "@/components/concordance-badge"
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
  const [maxWindows, setMaxWindows] = useState<number>(5)
  const [cmaesGens, setCmaesGens] = useState<number>(2)
  const [debugMode, setDebugMode] = useState<boolean>(true)

  // Results for both modes
  const [esiResult, setEsiResult] = useState<ESIResult | null>(null)
  const [bioResult, setBioResult] = useState<PhysDeepSIFResult | null>(null)

  // Playback speed state — passed to BrainVisualization for Plotly control
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1)

  // EEG waveform window selection
  const [selectedWindow, setSelectedWindow] = useState<number>(0)

  // XAI overlay toggle
  const [showOverlay, setShowOverlay] = useState<boolean>(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [useWs, setUseWs] = useState(false)
  const cmaesStatus = useWebSocket(useWs ? jobId : null)

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
      const fd = new FormData()
      fd.append("file", selectedFile)
      fd.append("mode", "biomarkers")
      fd.append("include_eeg", "true")
      fd.append("max_windows", String(maxWindows))
      fd.append("cmaes_generations", String(cmaesGens))
      fd.append("debug", debugMode ? "true" : "false")

      if (debugMode) {
        fd.append("ws", "false")
        const res = await fetch("/api/analyze", { method: "POST", body: fd })
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ message: "Analysis failed" }))
          throw new Error(errData.detail || errData.message || `Analysis failed (${res.status})`)
        }
        const data = await res.json()
        setEsiResult({ ...data, plotHtml: data.sourcePlotHtml || data.plotHtml })
        setBioResult(data)
        setStep("results")
      } else {
        fd.append("ws", "true")
        const res = await fetch("/api/analyze", { method: "POST", body: fd })
        if (!res.ok) {
          const errData = await res.json().catch(() => ({ message: "Analysis request failed" }))
          throw new Error(errData.message || `Analysis failed (${res.status})`)
        }
        const data = await res.json()
        setJobId(data.job_id)
        setUseWs(true)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during analysis")
      setStep("upload")
    }
  }, [selectedFile, maxWindows, cmaesGens, debugMode])

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
    const total = esiResult?.eegData?.windows?.length ?? 0
    if (total <= 0) return
    const normalizedIndex = ((frameIndex % total) + total) % total
    setClampedWindow(normalizedIndex)
  }, [esiResult?.eegData?.windows?.length, setClampedWindow])

  /* ---- Handle WebSocket job status updates ---- */
  useEffect(() => {
    if (!cmaesStatus.result) return
    const r = cmaesStatus.result

    queueMicrotask(() => {
      // Phase A: preliminary biomarker results
      if (r.epileptogenicity && !bioResult) {
        setBioResult({
          jobId: r.jobId,
          status: r.status,
          processingTime: 0,
          source: selectedFile?.name ?? "upload",
          plotHtml: "",
          fullHtmlPath: r.fullHtmlPath,
          epileptogenicity: r.epileptogenicity,
          concordance: r.concordance ?? null,
          cmaes: r.cmaes ?? null,
          xai: r.xai ?? null,
          groundTruth: null,
          eegData: null,
        })
      }

      // Completed: update with concordance if present
      if (r.status === "completed" && bioResult) {
        setBioResult((prev) =>
          prev
            ? {
                ...prev,
                status: "completed",
                concordance: r.concordance ?? null,
                cmaes: r.cmaes ?? null,
              }
            : prev,
        )
        setStep("results")
        setUseWs(false)
      }
    })
  }, [cmaesStatus, bioResult, selectedFile])

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

              <div className="flex items-center gap-2">
                <Checkbox
                  id="debug-mode"
                  checked={debugMode}
                  onCheckedChange={(v) => setDebugMode(v === true)}
                />
                <label htmlFor="debug-mode" className="text-sm text-muted-foreground cursor-pointer select-none flex items-center gap-1">
                  <Zap className="h-3.5 w-3.5 text-amber-500" />
                  Debug mode (skip CMA-ES, dummy concordance)
                </label>
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
                progress={cmaesStatus.status?.progress ?? 0}
                status={
                  cmaesStatus.cmaesRunning
                    ? `CMA-ES generation ${cmaesStatus.cmaesProgress.current}/${cmaesStatus.cmaesProgress.max}`
                    : cmaesStatus.status?.message ?? "Starting analysis..."
                }
              />
              <AnalysisSkeleton />
              {cmaesStatus.phaseAComplete && cmaesStatus.result?.fullHtmlPath && (
                <div className="mt-4">
                  <iframe
                    src={cmaesStatus.result.fullHtmlPath}
                    className="h-[400px] w-full rounded-lg opacity-60"
                    title="Preliminary brain heatmap"
                  />
                  <p className="mt-2 text-center text-xs text-muted-foreground">
                    Preliminary results — biophysical validation running in background
                  </p>
                </div>
              )}
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
                  {(esiResult?.eegData || bioResult?.eegData) && (
                    <EegWaveformPlot
                      eegData={esiResult?.eegData || bioResult?.eegData}
                      selectedWindow={selectedWindow}
                      onSelectedWindowChange={setClampedWindow}
                      className="w-full"
                    />
                  )}

                  {/* 3D Brain Visualization */}
                  {(esiResult?.plotHtml || bioResult?.plotHtml) && (
                    <BrainVisualization
                      plotHtml={esiResult?.plotHtml || bioResult?.plotHtml}
                      label="Source Activity"
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
                            onClick={() => setClampedWindow(idx)}
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
                      onSelectedWindowChange={setClampedWindow}
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

                  {bioResult?.concordance && (
                    <ConcordanceBadge
                      tier={bioResult.concordance.tier}
                      overlap={bioResult.concordance.overlap}
                      description={bioResult.concordance.tier_description}
                      sharedRegions={bioResult.concordance.shared_regions}
                    />
                  )}

                  <DetectedRegions
                    regions={detectedRegions}
                    variant="clinical"
                  />

                  {bioResult && (
                    <XaiPanel
                      channelImportance={bioResult.xai?.channel_importance ?? []}
                      timeImportance={bioResult.xai?.time_importance ?? []}
                      channelNames={bioResult.epileptogenicity?.region_labels ?? []}
                      onToggleOverlay={() => setShowOverlay((v) => !v)}
                      showOverlay={showOverlay}
                    />
                  )}
                </div>
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
