// Job store for tracking async processing jobs
// This module is shared between analyze-eeg, analyze-mat, job-status,
// and the new PhysDeepSIF biomarker/source-localization routes.

interface EegJobResult {
  plotHtml: string
  outputDir: string
  processingTime: number
}

interface MatJobResult extends EegJobResult {
  hasGroundTruth?: boolean
  metrics?: unknown
  bestWindow?: unknown
  nWindowsProcessed?: number
  sourceFile?: string
}

// PhysDeepSIF analysis result — returned by the Python FastAPI backend
export interface PhysDeepSIFResult {
  jobId: string
  status: string
  processingTime: number
  source: string
  plotHtml: string
  fullHtmlPath: string
  nWindowsProcessed?: number
  hasAnimation?: boolean
  epileptogenicity?: {
    scores: Record<string, number>
    scores_array: number[]
    epileptogenic_regions: string[]
    epileptogenic_regions_full?: string[]  // Full anatomical names
    threshold: number
    threshold_percentile: number
    max_score_region: string
    max_score: number
    region_labels: string[]
  }
  // Optional raw EEG windows returned by the backend for waveform display
  eegData?: {
    channels: string[]
    samplingRate: number
    windowLength: number
    windows: Array<{
      startTime: number
      endTime: number
      data: number[][]
    }>
  } | null
  sourceLocalization?: {
    scores: Record<string, number>
    scores_array: number[]
    top_active_regions: string[]
    top_active_regions_full?: string[]  // Full anatomical names
    max_activity_region: string
    max_activity_score: number
    region_labels: string[]
    summary?: Record<string, number>
  }
  groundTruth?: {
    available: boolean
    regions: string[]
    n_epileptogenic: number
    recall: number | null
    precision: number | null
    top5_recall: number | null
    top10_recall: number | null
  } | null
  sourceActivity?: {
    shape: number[]
    min: number
    max: number
    mean: number
    std: number
  }
  heuristic_ei_scores?: number[]
  concordance?: {
    tier: string
    overlap: number
    shared_regions?: string[]
    tier_description?: string
  } | null
  cmaes?: {
    status: string
    best_score?: number
    generations?: number
    biophysical_ei?: number[]
    error?: string
  } | null
  xai?: {
    channel_importance: number[]
    time_importance: number[]
    attribution_map?: number[][]
    top_segments?: Array<{
      channel_idx: number
      start_sample: number
      end_sample: number
      start_time_sec: number
      end_time_sec: number
      importance: number
    }>
    target_region_idx?: number
    baseline_score?: number
  } | null
}

interface Job<T> {
  status: 'processing' | 'completed' | 'failed'
  startTime: number
  error?: string
  result?: T
}

export const eegJobs = new Map<string, Job<EegJobResult>>()
export const matJobs = new Map<string, Job<MatJobResult>>()
export const physdeepsifJobs = new Map<string, Job<PhysDeepSIFResult>>()

// URL of the Python FastAPI backend that hosts the PhysDeepSIF model
export const PHYSDEEPSIF_BACKEND_URL = process.env.PHYSDEEPSIF_BACKEND_URL || "http://localhost:8000"
