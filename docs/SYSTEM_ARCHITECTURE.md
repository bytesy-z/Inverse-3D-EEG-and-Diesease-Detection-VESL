# PhysDeepSIF — System Architecture

## 1. Overview

PhysDeepSIF is a physics-informed deep learning EEG inverse solver that reconstructs 76-region brain source activity from 19-channel scalp EEG and maps epileptogenicity. The system accepts clinical EDF uploads or synthetic test samples, runs inference through a SpatialModule + BiLSTM network, performs power-based biomarker detection, validates findings via CMA-ES biophysical concordance using TVB Epileptor simulations, and generates occlusion-based XAI attribution. A FastAPI backend serves a Next.js frontend with real-time WebSocket progress, interactive 3D brain heatmaps, and color-coded concordance badges.

## 2. Pipeline Diagram

```
Patient EDF (19ch, 200Hz)
  → [Preprocessing: MNE load, channel map, bandpass 0.5-70Hz, notch 50Hz, 2s windows]
  → [PhysDeepSIF Inference: SpatialModule(19→128→256→256→128→76) + BiLSTM(hidden=76)]
  → [Biomarker Detection: AC variance → z-score → sigmoid → EI ∈ [0,1]]
  → [CMA-ES Concordance: x₀ optimization via TVB Epileptor, 30 gens, pop=14]
  → [XAI Occlusion: 200ms windows, channel+time importance]
  → [WebSocket → Frontend: real-time progress, heatmap, concordance badge]
```

## 3. Component Architecture

### 3.1 Backend (FastAPI, port 8000)

| Module | Files | Key Classes/Functions | Purpose |
|--------|-------|-----------------------|---------|
| API Server | `backend/server.py` (2222 lines) | `analyze_eeg()`, `biomarker_detection()`, `run_inference()` | Routes, WebSocket, preprocessing, inference orchestration |
| PhysDeepSIF Network | `src/phase2_network/physdeepsif.py` | `build_physdeepsif()`, `SpatialModule`, `TemporalModule` | 19→76 source imaging MLP + 2-layer BiLSTM |
| Loss Functions | `src/phase2_network/loss_functions.py` | `PhysDeepSIFLoss`, `compute_auc_epileptogenicity` | Source MSE + epi loss, physics regularization |
| Training | `src/phase2_network/trainer.py` | `train_one_epoch()`, `validate()` | AdamW, ReduceLROnPlateau, gradient clipping |
| Metrics | `src/phase2_network/metrics.py` | `compute_dle()`, `compute_spatial_dispersion()` | DLE, SD, AUC, temporal correlation |
| CMA-ES Inversion | `src/phase4_inversion/__init__.py` | `fit_patient()`, `ProgressCallback` | Exports public API for parameter inversion |
| Objective | `src/phase4_inversion/objective_function.py` | `compute_eeg_psd()`, `build_objective()` | PSD-MSE + L2 regularization on x₀ |
| CMA-ES Optimizer | `src/phase4_inversion/cmaes_optimizer.py` | `fit_patient()` | `cmaes` library wrapper, 30 generations, pop=14 |
| Biophysical EI | `src/phase4_inversion/epileptogenicity_index.py` | `compute_biophysical_ei()` | Sigmoid(x₀) → biophysical EI |
| Concordance | `src/phase4_inversion/concordance.py` | `compute_concordance()` | Overlap of top-10 heuristic vs biophysical → HIGH/MODERATE/LOW |
| XAI Occlusion | `src/xai/eeg_occlusion.py` | `explain_biomarker()` | 200ms sliding mask, channel+time importance |
| Validation | `src/phase5_validation/generate_figures.py` | `generate_all_figures()` | DLE histogram, AUC vs SNR, top-K recall, hemisphere accuracy, learning curve, concordance heatmap |

### 3.2 Frontend (Next.js, port 3000)

| File | Key Exports | Purpose |
|------|-------------|---------|
| `frontend/app/analysis/page.tsx` | default export | Main analysis page: upload, parameter controls, result display |
| `frontend/components/concordance-badge.tsx` | `ConcordanceBadge` | Color-coded tier (HIGH green / MODERATE yellow / LOW red) with overlap count and shared region names |
| `frontend/hooks/use-websocket.ts` | `useWebSocket` | WebSocket hook returning `status`, `connected`, `phaseAComplete`, `cmaesRunning`, `cmaesProgress`, `result` |
| `frontend/components/processing-window.tsx` | `ProcessingWindow` | Pipeline step checklist with elapsed timer and real progress bar |
| `frontend/components/xai-visualization.tsx` | (XAI overlay component) | Channel bar chart + temporal heatmap toggled on EEG waveform |

### 3.3 Infrastructure

| Resource | File | Details |
|----------|------|---------|
| Docker Compose | `deploy/docker-compose.yml` | 2 services (backend + frontend), bridge network `physdeepsif-net`, healthcheck on backend (curl /api/health every 30s), frontend depends_on backend service_healthy |
| Backend Dockerfile | `deploy/Dockerfile.backend` | python:3.9-slim, copies all data files from `data/` and `outputs/models/` |
| Frontend Dockerfile | `deploy/Dockerfile.frontend` | Node-based Next.js build |
| Orchestrator | `start.sh` | `--check`, `--backend`, `--frontend`, `--kill` flags; file validation; SWC binary fix helper |
| CI/CD | `.github/workflows/test.yml` | Runs on push/PR, excludes system+integration tests |

## 4. Data Flow (Detailed)

### 4.1 Upload Path (EDF)

1. `POST /api/analyze` with `file` → `analyze_eeg()` validates file extension (.edf/.npy/.mat), reads via MNE
2. `_process_edf_raw()`: bandpass 0.5-70 Hz, notch 50 Hz, map channel names (FP1→Fp1, drop A1/A2), segment into 2s windows (400 samples at 200 Hz)
3. `run_inference()`: per-channel temporal de-mean → global z-score with AC stats → model forward pass → denormalize source predictions
4. `compute_epileptogenicity_index()`: AC variance → z-score → sigmoid → EI ∈ [0,1] per region
5. `generate_heatmap_html()`: Plotly Mesh3d on fsaverage5 cortical surface, color-mapped by EI
6. `_run_cmaes_inversion()` via `asyncio.to_thread()`: TVB Epileptor simulation with CMA-ES (30 gens, pop=14), PSD-MSE objective, concordance engine → tier + shared regions
7. `_run_xai()`: occlusion attribution on top concordant region, returns channel_importance (19×1) + time_importance (1×T)
8. Response: job_id, source_activity (base64 npy), heatmap (Plotly HTML body), EI scores + top regions, concordance result, XAI data

### 4.2 Test Sample Path

1. `POST /api/analyze` with `sample_idx` → `_load_test_sample()` loads from `data/synthetic3/test_dataset.h5`
2. Same inference/preprocessing pipeline (steps 3-7 above), no EDF preprocessing

### 4.3 WebSocket Progress

1. Client connects to `ws://host/ws/{job_id}` after receiving job_id from POST response
2. Server-side `websocket_endpoint()` polls `active_jobs` dict (thread-safe, `threading.Lock`) every 0.5s
3. `_make_cmaes_callback()` writes generation, best_score, phase to `active_jobs[job_id]` after each CMA-ES generation
4. Flow: `queued` → `phase_a_complete` (instant heatmap ready) → `cmaes_running` (with generation X/30) → `completed` (with full result)
5. Frontend `useWebSocket` hook renders phases: phase A badge + "CMA-ES running in background" during cmaes_running, then concordance badge replaces preliminary badge on completion

## 5. Key Configuration (from config.yaml)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `training.batch_size` | 64 | Training batch size |
| `training.loss_weights.beta_forward` | 0.0 | Forward physics loss disabled (confirmed superior) |
| `training.physics_sub_weights.lambda_laplacian` | 0.0 | Laplacian regularization disabled |
| `training.physics_sub_weights.lambda_temporal` | 0.3 | Temporal smoothness regularization weight |
| `training.physics_sub_weights.lambda_amplitude` | 0.2 | Amplitude bound regularization weight |
| `network.temporal_module.hidden_dim` | 76 | BiLSTM hidden size |
| `parameter_inversion.max_generations` | 30 | CMA-ES generations |
| `parameter_inversion.population_size` | 14 | CMA-ES population size |
| `parameter_inversion.bounds` | [-2.4, -1.0] | x₀ search bounds |
| `parameter_inversion.initial_x0` | -2.1 | CMA-ES initial mean |
| `parameter_inversion.initial_sigma` | 0.3 | CMA-ES initial step size |
| `preprocessing.bandpass` | [0.5, 70.0] | Hz |
| `preprocessing.notch_freq` | 50.0 | Hz |
| `preprocessing.epoch_length_sec` | 2.0 | Window length |

## 6. Security & Error Handling

| Scenario | HTTP / Behavior | Location in `server.py` |
|----------|-----------------|------------------------|
| Model not loaded | 503 Service Unavailable | `run_inference()` line 433 |
| NaN/Inf input | 400 Bad Request | `run_inference()` line 436 |
| Rate limit (100 req/min/IP) | 429 Too Many Requests | `rate_limit_middleware()` line 283 |
| Disk full (<100 MB free) | 507 Insufficient Storage | Before result write |
| Request timeout | `asyncio.wait_for(60s)` raised | `_process_analysis_async()` |
| File not provided | 422 Unprocessable Entity | `analyze_eeg()` |
| Sample index out of range | 400 Bad Request | `_load_test_sample()` |
| CORS | Explicit `GET,POST,OPTIONS` + `Content-Type,Authorization` | `add_middleware()` line 266 |
| Queue full (max 10 concurrent) | 503 Service Unavailable | `_process_analysis_async()` semaphore |

## 7. Dependencies

| Category | Packages |
|----------|----------|
| Core ML | torch, numpy, scipy |
| Neural Mass | tvb-library, tvb-data |
| EEG | mne, pyedflib, mne-bids |
| Optimization | cmaes |
| Web | fastapi, uvicorn, websockets, python-multipart, pydantic |
| Data | h5py, nibabel |
| Visualization | plotly, matplotlib |
| Infrastructure | watchfiles |

## 8. File Layout

```
data/
├── leadfield_19x76.npy           # Forward matrix (19ch × 76 regions)
├── connectivity_76.npy           # Structural connectivity (76×76)
├── region_labels_76.json         # DK region names
├── region_centers_76.npy         # MNI coordinates (76×3)
├── tract_lengths_76.npy          # Fiber tract lengths (76×76)
└── synthetic3/
    ├── train_dataset.h5          # 80k samples (80,000, 19, 400)
    ├── val_dataset.h5            # 10k samples
    └── test_dataset.h5           # 10k samples

src/
├── phase1_forward/               # TVB simulation, leadfield construction
├── phase2_network/               # PhysDeepSIF model, training, losses, metrics
│   ├── physdeepsif.py            # build_physdeepsif(), SpatialModule, TemporalModule
│   ├── loss_functions.py         # PhysDeepSIFLoss, epileptogenicity AUC
│   ├── trainer.py                # train_one_epoch(), validate()
│   └── metrics.py                # DLE, SD, AUC, temporal correlation
├── phase4_inversion/             # CMA-ES, biophysical EI, concordance
│   ├── objective_function.py     # PSD computation, objective builder
│   ├── cmaes_optimizer.py        # fit_patient() wrapper
│   ├── epileptogenicity_index.py # Sigmoid EI from fitted x₀
│   └── concordance.py            # Top-K overlap → tier
├── phase5_validation/            # Baselines, validation figures
│   └── generate_figures.py       # 6 validation plots (775 lines)
└── xai/                          # Explainability
    └── eeg_occlusion.py          # Occulusion-based attribution (111 lines)

backend/
├── server.py                     # FastAPI application (2222 lines)
├── requirements.txt              # 18 runtime dependencies
└── region_names.py               # DK region name resolution

scripts/
├── 01_generate_data.py           # Synthetic dataset generation
├── 02_build_leadfield.py         # Leadfield matrix construction
└── 03_train_network.py           # Training entry point

outputs/
├── models/
│   ├── checkpoint_best.pt        # Best model weights (epoch 47)
│   └── normalization_stats.json  # Global z-score stats
├── figures/                      # Validation figures (6 plots)
├── logs/                         # Rotating JSON logs (10 MB, 5 backups)
└── frontend_results/             # Per-job inference outputs (24h TTL)

deploy/
├── docker-compose.yml            # 2 services, bridge network
├── Dockerfile.backend            # python:3.9-slim
├── Dockerfile.frontend           # Node/Next.js build
├── .dockerignore
└── .env.example

tests/
├── unit/                         # 18 files, 92 tests
├── functional/                   # 4 files, 16 tests
├── integration/                  # 3 files (inference lifecycle, EDF pipeline, full pipeline)
├── system/                       # 4 files
└── regression/                   # 3 files (amplitude, DC, normstats)

frontend/
├── app/analysis/page.tsx         # Main analysis page
├── components/
│   ├── concordance-badge.tsx     # Concordance tier display
│   ├── processing-window.tsx     # Progress tracking checklist
│   └── xai-visualization.tsx     # XAI channel/time overlay
├── hooks/
│   └── use-websocket.ts          # WebSocket hook for job progress
└── package.json
```

## 9. Inference Behaviour (Do Not Modify)

- Preprocessing applied before model inference: per-channel temporal de-meaning (remove DC per channel) THEN global z-score normalization using `normalization_stats.json`. Training used this exact order — do not reorder or omit.
- When returning predictions the backend denormalizes outputs to original scale.
- Sliding-window segmentation for EDF uploads: window_length=400 samples (2s at 200 Hz) with 50% overlap (step=200). The backend builds Plotly animation HTML for multi-window outputs and embeds only the body + inline Plotly script when returning to the frontend.
- CMA-ES runs in a background thread via `asyncio.to_thread()` with a progress callback updating `active_jobs`.
- XAI occlusion runs on the top concordant region after CMA-ES completes.

## 10. Concordance Engine

The concordance engine compares the top-10 regions from instantaneous biomarker detection (Phase A) against the top-10 regions from CMA-ES biophysical fitting (Phase B):

| Overlap | Tier | Description |
|---------|------|-------------|
| ≥5/10 | HIGH | Both methods independently agree: strong evidence |
| 2-4/10 | MODERATE | Partial agreement: correlate with clinical findings |
| ≤1/10 | LOW | Methods disagree: consider longer recording or stereo-EEG |

The result replaces the preliminary biomarker badge with a color-coded concordance badge on the frontend.
