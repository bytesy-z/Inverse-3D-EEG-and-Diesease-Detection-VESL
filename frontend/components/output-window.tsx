"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RotateCcw } from "lucide-react"
import { BrainVisualization } from "./brain-visualization"
import { AnimatedBrainVisualization } from "./animated-brain-visualization"

interface EvaluationMetrics {
  mean_nmse: number
  mean_auc: number
  mean_localization_error_mm: number
  mean_time_error_ms?: number
  best_window_nmse?: number
  best_window_auc?: number
  best_window_localization_error_mm?: number
  n_windows?: number
}

interface SourceLocalizationData {
  scores?: Record<string, number>
  scores_array?: number[]
  top_active_regions?: string[]
  max_activity_region?: string
  max_activity_score?: number
  region_labels?: string[]
  summary?: {
    mean_rms: number
    max_rms: number
    mean_variance: number
    max_variance: number
    spatial_cv: number
  }
}

interface OutputWindowProps {
  result: {
    fileName: string
    fileType?: "edf" | "mat"
    uploadTime: string
    processingTime: number
    plotHtml: string
    animationFile?: string
    outputDir?: string
    mode?: "source_localization" | "biomarkers"
    sourceLocalization?: SourceLocalizationData
    hasGroundTruth?: boolean
    metrics?: EvaluationMetrics
    nWindowsProcessed?: number
    sourceFile?: string
  }
  onNewAnalysis: () => void
}

export function OutputWindow({ result, onNewAnalysis }: OutputWindowProps) {
  const [activeTab, setActiveTab] = useState<"visualization" | "details">("visualization")

  const isSimulation = result.fileType === "mat"
  const hasMetrics = result.hasGroundTruth && result.metrics
  const isESI = result.mode === "source_localization"

  return (
    <div className="space-y-6">
      {/* Header with Actions */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-semibold text-white">Analysis Complete</h2>
          <p className="text-sm text-slate-300 mt-1">
            File: <span className="font-medium text-white">{result.fileName}</span>
            {isSimulation && (
              <span className="ml-2 px-2 py-0.5 bg-emerald-500/20 text-emerald-300 text-xs rounded-full">
                Simulation
              </span>
            )}
          </p>
        </div>
        <div className="flex gap-3 w-full sm:w-auto">
          <Button onClick={onNewAnalysis} variant="outline" className="flex-1 sm:flex-none bg-transparent border-border/60 text-emerald-300 hover:text-white hover:bg-emerald-500/10">
            <RotateCcw className="w-4 h-4 mr-2" />
            New Analysis
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border">
        <button
          onClick={() => setActiveTab("visualization")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${
            activeTab === "visualization"
              ? "text-emerald-300 border-b-2 border-emerald-300"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {isESI ? "Source Activity Map" : "3D Brain Map"}
        </button>
        <button
          onClick={() => setActiveTab("details")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${
            activeTab === "details"
              ? "text-emerald-300 border-b-2 border-emerald-300"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Analysis Details
        </button>
      </div>

      {/* Content */}
      {activeTab === "visualization" && (
        <Card className="bg-card border border-border h-[600px] overflow-hidden">
          {result.animationFile && result.outputDir ? (
            <AnimatedBrainVisualization 
              animationFilePath={`${result.outputDir}/${result.animationFile}`}
            />
          ) : (
            <BrainVisualization plotHtml={result.plotHtml} />
          )}
        </Card>
      )}

      {activeTab === "details" && (
        <Card className="p-6 bg-background/40 border border-border/60">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Input Information */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Input Information</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">File Name</p>
                  <p className="text-sm text-slate-200 mt-1">{result.fileName}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">File Type</p>
                  <p className="text-sm text-slate-200 mt-1">
                    {isSimulation ? "Simulation (MAT)" : "Recording (EDF)"}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Upload Time</p>
                  <p className="text-sm text-slate-200 mt-1">{result.uploadTime}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Analysis Mode</p>
                  <p className="text-sm text-slate-200 mt-1">
                    {isESI ? "Source Imaging (ESI)" : "Biomarker Detection"}
                  </p>
                </div>
              </div>
            </div>

            {/* Processing Information */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Processing Information</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Processing Time</p>
                  <p className="text-sm text-slate-200 mt-1">{result.processingTime.toFixed(2)}s</p>
                </div>
                {result.nWindowsProcessed && (
                  <div>
                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Windows Processed</p>
                    <p className="text-sm text-slate-200 mt-1">{result.nWindowsProcessed}</p>
                  </div>
                )}
                <div>
                  <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Status</p>
                  <p className="text-sm text-emerald-300 font-medium mt-1">Completed Successfully</p>
                </div>
              </div>
            </div>
          </div>

          {/* ESI Source Activity Metrics */}
          {isESI && result.sourceLocalization && (
            <div className="mt-8 pt-6 border-t border-border/60">
              <h3 className="text-lg font-semibold text-white mb-4">
                Source Activity Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {result.sourceLocalization.summary && (
                  <>
                    <MetricCard
                      label="Spatial CV"
                      value={result.sourceLocalization.summary.spatial_cv.toFixed(3)}
                      description="Spatial coefficient of variation"
                      highlight={result.sourceLocalization.summary.spatial_cv > 0.1}
                    />
                    <MetricCard
                      label="Max RMS"
                      value={result.sourceLocalization.summary.max_rms.toFixed(4)}
                      description="Peak RMS amplitude"
                    />
                    <MetricCard
                      label="Mean RMS"
                      value={result.sourceLocalization.summary.mean_rms.toFixed(4)}
                      description="Average RMS across regions"
                    />
                    <MetricCard
                      label="Max Variance"
                      value={result.sourceLocalization.summary.max_variance.toFixed(4)}
                      description="Peak temporal variance"
                    />
                  </>
                )}
              </div>

              {/* Top Active Regions */}
              {result.sourceLocalization.top_active_regions && (
                <div className="mt-6">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Most Active Regions</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.sourceLocalization.top_active_regions.slice(0, 10).map((region, idx) => (
                      <span
                        key={region}
                        className={`px-3 py-1 rounded-full text-xs font-medium ${
                          idx < 3
                            ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                            : idx < 6
                            ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                            : 'bg-slate-700 text-slate-300 border border-slate-600'
                        }`}
                      >
                        #{idx + 1} {region}
                        {result.sourceLocalization?.scores?.[region] !== undefined &&
                          ` (${result.sourceLocalization.scores[region].toFixed(3)})`
                        }
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Evaluation Metrics (only for simulations with ground truth, non-ESI mode) */}
          {!isESI && hasMetrics && result.metrics && (
            <div className="mt-8 pt-6 border-t border-border/60">
              <h3 className="text-lg font-semibold text-white mb-4">
                Evaluation Metrics
                <span className="ml-2 text-xs font-normal text-slate-400">(Ground Truth Available)</span>
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  label="Mean nMSE"
                  value={result.metrics.mean_nmse.toFixed(4)}
                  description="Normalized Mean Squared Error"
                />
                <MetricCard
                  label="Mean AUC"
                  value={result.metrics.mean_auc.toFixed(4)}
                  description="Area Under ROC Curve"
                  highlight={result.metrics.mean_auc > 0.7}
                />
                <MetricCard
                  label="Localization Error"
                  value={`${result.metrics.mean_localization_error_mm.toFixed(2)} mm`}
                  description="Mean Distance to True Source"
                  highlight={result.metrics.mean_localization_error_mm < 20}
                />
                {result.metrics.best_window_localization_error_mm !== undefined && (
                  <MetricCard
                    label="Best Window Error"
                    value={`${result.metrics.best_window_localization_error_mm.toFixed(2)} mm`}
                    description="Localization Error at Peak"
                  />
                )}
              </div>
              {result.sourceFile && (
                <p className="text-xs text-slate-500 mt-4">
                  Ground truth source: {result.sourceFile}
                </p>
              )}
            </div>
          )}

          {/* About ESI */}
          {isESI && (
            <div className="mt-8 pt-6 border-t border-border/60">
              <p className="text-sm text-slate-400">
                <span className="text-slate-300 font-medium">About:</span> EEG Source Imaging (ESI)
                estimates which brain regions generated the recorded scalp EEG signals. The 3D heatmap
                shows relative activity levels across 76 Desikan-Killiany atlas regions. Warmer colors
                (red/orange) indicate higher estimated neural activity. The model uses the PhysDeepSIF
                physics-informed deep learning architecture.
              </p>
            </div>
          )}

          {/* Note for EDF files (non-ESI mode) */}
          {!isESI && !isSimulation && (
            <div className="mt-8 pt-6 border-t border-border/60">
              <p className="text-sm text-slate-400">
                <span className="text-slate-300 font-medium">Note:</span> Evaluation metrics are only available for simulation files (MAT format) where ground truth source locations are known.
              </p>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string
  description: string
  highlight?: boolean
}

function MetricCard({ label, value, description, highlight }: MetricCardProps) {
  return (
    <div className={`p-4 rounded-lg ${highlight ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-slate-800/50'}`}>
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">{label}</p>
      <p className={`text-xl font-bold mt-1 ${highlight ? 'text-emerald-300' : 'text-white'}`}>{value}</p>
      <p className="text-xs text-slate-500 mt-1">{description}</p>
    </div>
  )
}
