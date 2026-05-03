# PhysDeepSIF — Physics-Informed Deep Source Imaging Framework

Physics-constrained deep learning for EEG source localization and patient-specific epileptogenicity mapping. Uses a 76-region brain parcellation (Desikan-Killiany atlas) and the Epileptor neural mass model in The Virtual Brain (TVB) to produce biophysically grounded epileptogenicity heatmaps from 19-channel scalp EEG (10-20 montage, linked-ear reference).

**Final year project.** 149 test functions pass, DLE=31mm, AUC=0.923.

---

## Table of Contents

1. [Quick Start (pre-built)](#quick-start-pre-built)
2. [Environment Setup](#environment-setup)
3. [Architecture Overview](#architecture-overview)
4. [Full Pipeline Runbook](#full-pipeline-runbook)
   - [Step 1: Build Source Space & Leadfield](#step-1-build-source-space--leadfield)
   - [Step 2: Generate Synthetic Training Data](#step-2-generate-synthetic-training-data)
   - [Step 3: Train the PhysDeepSIF Network](#step-3-train-the-physdeepsif-network)
   - [Step 4: Run Validation & Generate Figures](#step-4-run-validation--generate-figures)
   - [Step 5: Run the Web Application](#step-5-run-the-web-application)
   - [Step 6: Run CMA-ES Patient-Specific Inversion](#step-6-run-cma-es-patient-specific-inversion)
5. [Project Structure](#project-structure)
6. [Key Files & Their Roles](#key-files--their-roles)
7. [Running Tests](#running-tests)
8. [Config Reference](#config-reference)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start (pre-built)

If the pre-built model and data files are present (they ship with the repo), you can start the web app immediately:

```bash
# One command — checks deps, starts backend (port 8000) + frontend (port 3000)
./start.sh
```

Visit **http://localhost:3000** in a browser. Upload an EDF file or select a test sample from the dropdown.

### Start modes

```bash
./start.sh --check      # Run dependency & file checks only — exits
./start.sh --backend    # Start backend only (FastAPI on port 8000)
./start.sh --frontend   # Start frontend only (Next.js on port 3000)
./start.sh --kill       # Kill servers started by start.sh
```

### Health check

```bash
curl -sS http://127.0.0.1:8000/api/health
# → {"status":"ok","model_loaded":true,"device":"cuda","num_regions":76,...}
```

### Quick inference from test dataset

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "sample_idx=10" \
  -F "mode=biomarkers"
```

### Upload a patient EDF file

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@data/test_demo.edf" \
  -F "mode=source_localization"
```

### End-to-end test script

```bash
./scripts/test_e2e.sh
# Tests: health check → synthetic sample analyze → EDF upload (both modes)
```

---

## Environment Setup

### Option A: Conda (recommended)

```bash
conda create -n deepsif python=3.10
conda activate deepsif

# Install PyTorch with CUDA support (adjust cuda version to match your GPU)
conda install pytorch==2.1.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# Or let pip handle it:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install pipeline dependencies
pip install -r requirements.txt

# Install backend server dependencies
pip install -r backend/requirements.txt

# Install Node.js 20+ (for frontend)
# Use nvm, conda, or system package manager:
#   nvm install 20 && nvm use 20
#   OR: conda install -c conda-forge nodejs=20
```

### Option B: venv

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Verify environment

```bash
./start.sh --check
```

Expected output:
```
[  OK] Python: Python 3.9+
[  OK] Python packages: torch, fastapi, uvicorn, h5py, numpy, scipy
[  OK] GPU: CUDA available            # or "CPU only" — both work
[  OK] Found: checkpoint_best.pt
[  OK] Found: normalization_stats.json
[  OK] Found: leadfield_19x76.npy
[  OK] Found: connectivity_76.npy
[  OK] Found: region_labels_76.json
[  OK] Found: region_centers_76.npy
[  OK] Node.js: v20+
[  OK] SWC binary: OK
=== All Checks Passed ===
```

### Required files (must exist or be generated)

| File | Size | Source | Description |
|------|------|--------|-------------|
| `data/connectivity_76.npy` | 47 KB | Step 1 or shipped | 76×76 structural connectivity matrix |
| `data/region_centers_76.npy` | 2 KB | Step 1 or shipped | 76×3 MNI coordinates |
| `data/region_labels_76.json` | 2 KB | Step 1 or shipped | 76 region name strings |
| `data/tract_lengths_76.npy` | 47 KB | Step 1 or shipped | 76×76 tract length matrix |
| `data/leadfield_19x76.npy` | 11 KB | Step 1 or shipped | 19×76 BEM leadfield matrix |
| `data/synthetic3/train_dataset.h5` | ~10 GB | Step 2 | Training dataset (23k sims × 5 windows) |
| `data/synthetic3/val_dataset.h5` | ~1.3 GB | Step 2 | Validation dataset (2.9k sims × 5 windows) |
| `data/synthetic3/test_dataset.h5` | ~6.3 MB | Step 2 or shipped | Test dataset (2.9k sims × 5 windows) |
| `outputs/models/checkpoint_best.pt` | 4.8 MB | Step 3 or shipped | Trained model weights |
| `outputs/models/normalization_stats.json` | 297 B | Step 3 or shipped | Z-score normalization statistics |

**Total storage**: ~13 GB for full synthetic data, ~10 MB for model + source space files.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (Offline, One-Time)                              │
│                                                                     │
│  Step 1: Build source space + 19×76 leadfield matrix (MNE BEM)      │
│  Step 2: Generate 50k–100k synthetic EEG samples (TVB Epileptor)    │
│  Step 3: Train PhysDeepSIF (spatial MLP + BiLSTM + physics loss)    │
│  Step 4: Generate 6 validation figures (DLE, AUC, top-K, etc.)      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  INFERENCE PIPELINE (Per EEG Recording)                             │
│                                                                     │
│  Upload EDF (19ch, 200Hz)                                           │
│      │                                                              │
│      ▼                                                              │
│  PhysDeepSIF Inference                                              │
│      19ch EEG → 76-region source activity estimates                 │
│      │                                                              │
│      ├──► Phase A: Instantaneous Biomarker Detection (top-10)       │
│      │                                                              │
│      └──► Phase B: CMA-ES Biophysical Fitting                       │
│            TVB Epileptor model, optimise x₀ for each region         │
│            │                                                        │
│            ▼                                                        │
│      Sigmoid(x₀) → Biophysical Epileptogenicity Index (EI)         │
│            │                                                        │
│            ▼                                                        │
│      Concordance: Phase A ∩ Phase B                                 │
│      → HIGH (≥5/10) / MODERATE (2-4) / LOW (<2)                    │
│      → 3D brain heatmap + XAI occlusion importance                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Network**: PhysDeepSIF = Spatial Module (MLP 19→128→256→256→128→76, ReLU, BatchNorm, skip connections) + Temporal Module (2-layer BiLSTM, hidden=76, dropout=0.1).

**Loss**: Composite = α·L_source (MSE) + β·L_forward (β=0.0) + γ·L_physics (amplitude + temporal constraints) + δ·L_epi (BCE classification).

**Config**: β=0.0, λ_L=0.0, λ_T=0.3, λ_A=0.2 — forward loss disabled, spatial smoothness disabled.

---

## Full Pipeline Runbook

If you're reproducing from scratch (without the shipped pre-built files), follow these steps in order.

### Step 1: Build Source Space & Leadfield

**What**: Creates the foundational data files from TVB's Desikan-Killiany atlas and MNE's BEM forward model.

**Time**: ~5 minutes (one-time, CPU)

```bash
python scripts/01_build_leadfield.py
```

**What it does internally**:
1. Loads TVB's 76-region connectivity → `data/connectivity_76.npy`
2. Extracts region labels, MNI centers, tract lengths → `data/region_*.json|npy`
3. Constructs 3-layer BEM forward model on `fsaverage` template head
4. Maps 76 DK regions to 19 10-20 electrode positions
5. Applies linked-ear re-referencing (rank-1 nullspace)
6. Validates: shape (19,76), rank=18, no column norm >100× median

**Output**: `data/{connectivity_76.npy, region_centers_76.npy, region_labels_76.json, tract_lengths_76.npy, leadfield_19x76.npy}`

**Skip if**: Files already exist in `data/`.

---

### Step 2: Generate Synthetic Training Data

**What**: Runs TVB Epileptor simulations to produce HDF5 training datasets.

**Time**: ~5 hours on 1 CPU core, ~45 minutes on 16 cores (GPU not used)

```bash
# Generate all 3 splits (train + val + test) — full dataset
python scripts/02_generate_synthetic_data.py

# Or generate individual splits:
python scripts/02_generate_synthetic_data.py --split train
python scripts/02_generate_synthetic_data.py --split val
python scripts/02_generate_synthetic_data.py --split test

# Override simulation counts (useful for quick smoke test):
python scripts/02_generate_synthetic_data.py --n-sims 100 --split train

# Control parallelism:
python scripts/02_generate_synthetic_data.py --n-jobs 16
```

**What each simulation produces**:
1. Samples random Epileptor parameters: x₀ (healthy [-2.2,-2.05] or epi [-1.8,-1.2]), coupling strength [0.5,3.0], noise [1e-4,5e-3], time constants
2. Runs TVB simulation at dt=0.1ms → raw 20kHz → FIR anti-alias decimate ×100 → 200Hz
3. Projects through leadfield: clean EEG = L @ source_activity
4. Adds white Gaussian noise (SNR [5,30] dB) + colored 1/f^α noise (10-30% amplitude)
5. Applies skull attenuation (4th-order Butterworth LP @ 40Hz)
6. Applies spectral shaping (STFT-based: suppress delta, boost alpha, anteroposterior gradients)
7. Validates spatial-spectral properties (PDR, alpha/beta gradients); discards failures
8. Segments into 5 × 2-second windows (400 samples each)

**Default dataset sizes** (from `config.yaml` §`synthetic_data`):

| Split | Simulations | Windows (5/sim) | Approx. HDF5 Size | Purpose |
|-------|------------|-----------------|-------------------|---------|
| Train | 23,000 | 115,000 | ~10 GB | Network training |
| Val | 2,900 | 14,500 | ~1.3 GB | Early stopping, LR scheduling |
| Test | 2,900 | 14,500 | ~1.3 GB | Validation metrics (DLE, AUC, top-K) |

HDF5 structure per sample: `eeg` (19,400), `source_activity` (76,400), `epileptogenic_mask` (76,), `x0_vector` (76,), `snr_db`, `global_coupling`.

**Hardware estimate**: 16 GB RAM minimum. 32 GB recommended for parallel generation.

**Skip if**: `data/synthetic3/{train,val,test}_dataset.h5` already exist.

---

### Step 3: Train the PhysDeepSIF Network

**What**: Trains the spatial + temporal neural network on synthetic data.

**Time**: ~2-8 hours on GPU (CUDA), ~12-24 hours on CPU

```bash
# Standard training with config.yaml defaults
python scripts/03_train_network.py

# Override device explicitly
python scripts/03_train_network.py --device cuda
python scripts/03_train_network.py --device cpu

# Override hyperparameters
python scripts/03_train_network.py --epochs 300 --batch-size 128

# Use a custom data directory
python scripts/03_train_network.py --data-dir data/synthetic3/

# Use custom config
python scripts/03_train_network.py --config my_config.yaml
```

**Training hyperparameters** (from `config.yaml` §`training`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 64 | |
| Max epochs | 200 | Early stopping at 40 no-improvement |
| Optimizer | AdamW | lr=0.001, weight_decay=1e-4 |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1e-6 |
| Gradient clipping | 5.0 | Global norm |
| Loss weights | α=1.0, β=0.0, γ=0.01, δ=1.0 | β=0.0 = forward loss disabled |
| Physics weights | λ_A=0.2, λ_L=0.0, λ_T=0.3 | No spatial smoothness penalty |

**Output**:
- `outputs/models/checkpoint_best.pt` — best validation-loss checkpoint (4.8 MB)
- `outputs/models/checkpoint_latest.pt` — final epoch checkpoint
- `outputs/models/normalization_stats.json` — EEG/source z-score stats
- `outputs/models/training.log` — full training log (loss per epoch)

Additional normalization snapshots (`_beta_0`, `_lambdaL_01`, `_orig`, `_v1`) capture intermediate config variants.

**Skip if**: `outputs/models/checkpoint_best.pt` and `outputs/models/normalization_stats.json` already exist.

---

### Step 4: Run Validation & Generate Figures

**What**: Evaluates the trained model on the test set and generates 6 figures.

**Time**: ~10-30 minutes (CPU)

```bash
# Generate all 6 validation figures
python -m src.phase5_validation.generate_figures

# Or use the alternative script (generates 4 of 6 figures with different styling)
python scripts/12_generate_validation_figures.py --num-samples 1000 --device cpu
```

**The 6 validation figures** (saved to `outputs/figures/`):

| # | File | What It Shows |
|---|------|---------------|
| 1 | `dle_histogram.png` | Distance Localisation Error: PhysDeepSIF (31mm) vs eLORETA vs Oracle |
| 2 | `auc_vs_snr.png` | AUC vs SNR (5-30 dB) with error bars |
| 3 | `topk_recall.png` | Top-K recall (1-10) for epi region detection |
| 4 | `hemisphere_accuracy.png` | Left/right hemisphere correct-classification rate |
| 5 | `learning_curve.png` | Train/val loss across training epochs |
| 6 | `concordance_heatmap.png` | Phase A (biomarker) vs Phase B (CMA-ES) top-10 overlap distribution |

**Other validation scripts**:

```bash
# Comprehensive validation suite on fresh simulations
python scripts/10_final_validation.py

# Validate spectral/spatial properties of enhanced data
python scripts/09_validate_enhanced_data.py

# Generate + plot 3 fresh samples through full pipeline
python scripts/11_plot_fresh_samples.py

# Diagnostic gradient audit
python scripts/diag_gradient_audit.py
```

**Skip if**: All 6 figures already exist in `outputs/figures/`.

---

### Step 5: Run the Web Application

**What**: Start the FastAPI backend (port 8000) and Next.js frontend (port 3000).

```bash
# Start both servers
./start.sh

# Or start individually:
./start.sh --backend    # http://localhost:8000/docs → Swagger UI
./start.sh --frontend   # http://localhost:3000
```

**Frontend features** (7 pages + 6 API routes):
- `/` — Landing page with file upload and test sample selector
- `/analysis` — Main analysis page: upload EDF, select mode, view results
- `/biomarkers` — Instantaneous biomarker detection with 3D brain heatmap
- `/disease-detection` — Disease detection mode
- `/eeg-sliding-window` — Sliding-window EEG analysis
- `/eeg-source-localization` — Source localization results view
- `/optimized-localization` — CMA-ES optimized localization results
- Concordance badge (HIGH/MODERATE/LOW) with overlapping region names
- XAI occlusion analysis (channel importance bars + time heatmap)
- EEG waveform viewer with interactive Plotly charts

**API quick examples**:

```bash
# Health check
curl http://localhost:8000/api/health

# List available test samples
curl http://localhost:8000/api/test-samples

# Analyze a synthetic test sample (biomarker mode — fast)
curl -X POST http://localhost:8000/api/analyze \
  -F "sample_idx=10" \
  -F "mode=biomarkers"

# Analyze a synthetic test sample (source_localization mode — full pipeline)
curl -X POST http://localhost:8000/api/analyze \
  -F "sample_idx=10" \
  -F "mode=source_localization"

# Upload an EDF file (source_localization mode)
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@data/test_demo.edf" \
  -F "mode=source_localization"

# View results (job_id is returned in the analyze response)
curl http://localhost:8000/api/results/{job_id}/brain_heatmap.html
```

**Biomarkers mode** returns: region scores, detected epi regions, threshold, source activity metrics.  
**Source localization mode** returns: full heatmap, animated frames (for multi-window EDF), CMA-ES fitted x₀, biophysical EI, concordance tier.

---

### Step 6: Run CMA-ES Patient-Specific Inversion

**What**: Fits patient-specific x₀ excitability parameters using evolutionary optimization.

**Time**: ~5-15 minutes per patient (CPU, depends on max_generations)

```bash
# Quick test (30 generations, pop=14)
python scripts/08_run_cmaes.py --patient-idx 0 --quick-test

# Full CMA-ES inversion
python scripts/08_run_cmaes.py --patient-idx 0

# Override params
python scripts/08_run_cmaes.py --patient-idx 5 --max-generations 50 --device cuda
```

**See also**: `scripts/demo_biomarker_detection.py` for an end-to-end single-sample demo:

```bash
python scripts/demo_biomarker_detection.py --sample-idx 10
# Output: 3D brain heatmap HTML at outputs/brain_heatmap_sample_10.html
```

---

## Project Structure

```
fyp-2.0/
├── src/
│   ├── region_names.py             # Region name helper
│   ├── phase1_forward/            # TVB Epileptor simulation + synthetic dataset
│   │   ├── epileptor_simulator.py     # One TVB simulation (dt=0.1ms, Raw monitor, FIR decimation)
│   │   ├── synthetic_dataset.py       # Orchestrator: sims → EEG → noise → HDF5
│   │   ├── leadfield_builder.py       # MNE BEM forward model (19×76)
│   │   ├── parameter_sampler.py       # Random Epileptor parameter sampling
│   │   └── source_space.py            # Load/process TVB connectivity
│   ├── phase2_network/            # PhysDeepSIF network + training
│   │   ├── physdeepsif.py             # Spatial MLP + BiLSTM model
│   │   ├── loss_functions.py          # PhysicsInformedLoss (4-component composite)
│   │   ├── trainer.py                 # Training loop, early stopping, logging
│   │   └── metrics.py                 # DLE, AUC, spatial dispersion, temporal correlation
│   ├── phase4_inversion/          # CMA-ES patient-specific parameter fitting
│   │   ├── cmaes_optimizer.py         # fit_patient() — CMA-ES wrapper
│   │   ├── objective_function.py      # EEG PSD-based objective + L2 regularisation
│   │   ├── epileptogenicity_index.py  # Sigmoid(x₀) → biophysical EI
│   │   └── concordance.py             # Top-10 overlap → HIGH/MODERATE/LOW
│   ├── phase5_validation/         # Validation metrics + figure generation
│   │   └── generate_figures.py        # 6 publication-ready figures
│   └── xai/                        # XAI occlusion analysis
│       └── eeg_occlusion.py           # Channel + time occlusion importance
│
├── backend/                        # FastAPI web server (port 8000)
│   ├── server.py                     # 2545 lines: all endpoints, inference, WebSocket
│   └── requirements.txt             # Server dependencies (20 packages)
│
├── frontend/                       # Next.js web dashboard (port 3000)
│   ├── app/
│   │   ├── page.tsx                   # Landing page
│   │   ├── mainpage.tsx               # Main page layout
│   │   ├── analysis/page.tsx          # Core analysis page
│   │   ├── biomarkers/page.tsx        # Biomarker detection view
│   │   ├── disease-detection/page.tsx  # Disease detection mode
│   │   ├── eeg-sliding-window/page.tsx # Sliding-window analysis
│   │   ├── eeg-source-localization/page.tsx # Source localization results
│   │   ├── optimized-localization/page.tsx  # CMA-ES optimized view
│   │   └── api/
│   │       ├── analyze-eeg/route.ts   # EEG analysis proxy
│   │       ├── analyze-mat/route.ts   # MAT file analysis proxy
│   │       ├── job-status/route.ts    # Async job status polling
│   │       ├── physdeepsif/route.ts   # Direct PhysDeepSIF inference
│   │       ├── serve-result/route.ts  # Result file serving
│   │       └── test-samples/route.ts  # Test sample listing
│   ├── components/
│   │   ├── app-shell.tsx              # Application shell wrapper
│   │   ├── analysis-skeleton.tsx      # Loading skeleton for analysis
│   │   ├── brain-visualization.tsx    # Plotly brain heatmap
│   │   ├── concordance-badge.tsx      # HIGH/MODERATE/LOW tier display
│   │   ├── eeg-waveform-plot.tsx      # Interactive EEG waveform viewer
│   │   ├── error-alert.tsx            # Error alert component
│   │   ├── error-boundary.tsx         # React error boundary
│   │   ├── file-upload-section.tsx    # File upload UI
│   │   ├── processing-window.tsx      # Pipeline step checklist + progress bar
│   │   ├── results-summary.tsx        # Results summary panel
│   │   ├── step-indicator.tsx         # Processing step indicator
│   │   ├── theme-provider.tsx         # Theme context provider
│   │   ├── xai-panel.tsx              # Channel + time occlusion overlay
│   │   └── ui/                        # shadcn/ui components (~55 files)
│   ├── hooks/
│   │   └── use-websocket.ts           # WebSocket hook for live progress
│   ├── lib/
│   │   ├── colormaps.ts               # Colormap utilities
│   │   ├── job-store.ts               # Job state management
│   │   ├── npz-parser.ts              # NPZ file parser
│   │   └── utils.ts                   # General utilities
│   ├── package.json
│   ├── next.config.mjs
│   ├── tsconfig.json
│   ├── start-dev.sh                   # Frontend dev server launcher
│   └── public/                        # Static assets
│
├── scripts/                        # CLI entry points
│   ├── 01_build_leadfield.py         # Build source space + leadfield
│   ├── 02_generate_synthetic_data.py # Generate HDF5 datasets
│   ├── 03_train_network.py           # Train PhysDeepSIF
│   ├── 06_run_validation.py          # Run validation pipeline
│   ├── 08_apply_spatial_gradients.py # Apply spatial gradients to data
│   ├── 08_run_cmaes.py               # CMA-ES parameter inversion
│   ├── 09_validate_enhanced_data.py  # Validate spectral/spatial properties
│   ├── 10_final_validation.py        # Full validation suite
│   ├── 11_plot_fresh_samples.py      # Visual inspection of samples
│   ├── 12_generate_validation_figures.py # Alternative figure generator
│   ├── demo_biomarker_detection.py   # End-to-end single-sample demo
│   ├── diag_gradient_audit.py        # Gradient diagnostic audit
│   ├── overfit_test.py               # Overfit/sanity test
│   ├── test_analyze_eeg.py           # EEG analysis test
│   ├── test_spectral_shaping.py      # Spectral shaping test
│   ├── test_e2e.sh                   # E2E integration test
│   └── smoke_test.sh                 # Quick smoke test
│
├── deploy/                         # Docker deployment
│   ├── docker-compose.yml            # 2-service orchestration
│   ├── Dockerfile.backend            # Backend container
│   ├── Dockerfile.frontend           # Frontend container
│   └── .dockerignore
│
├── data/
│   ├── leadfield_19x76.npy          # 19×76 BEM leadfield
│   ├── connectivity_76.npy          # 76×76 structural connectivity
│   ├── region_labels_76.json        # Region names
│   ├── region_centers_76.npy        # Region XYZ coordinates
│   ├── tract_lengths_76.npy         # 76×76 tract lengths
│   ├── synthetic3/
│   │   ├── test_dataset.h5           # Test dataset (~6.3 MB)
│   │   └── test_batch_check.h5       # Batch validation check (~4 MB)
│   ├── test_demo.edf                # Small demo EDF (43 KB)
│   └── samples/
│       ├── 0001082.edf               # Patient EDF sample
│       └── 1082.csv                  # Patient channel labels
│
├── outputs/
│   ├── models/
│   │   ├── checkpoint_best.pt        # Best-validation-loss checkpoint (4.8 MB)
│   │   ├── checkpoint_latest.pt      # Final-epoch checkpoint (4.8 MB)
│   │   ├── normalization_stats.json  # Current normalization stats
│   │   ├── normalization_stats_beta_0.json     # β=0 config snapshot
│   │   ├── normalization_stats_lambdaL_01.json # λ_L=0.1 config snapshot
│   │   ├── normalization_stats_orig.json       # Original config snapshot
│   │   └── normalization_stats_v1.json         # v1 config snapshot
│   ├── figures/                     # 6 validation figures + diagnostic plots
│   ├── frontend_results/            # Per-job runtime artifacts (~170 jobs)
│   ├── patient_heatmaps/            # Patient heatmap output directory
│   └── results/                     # Result output directory
│
├── tests/                          # 149 test functions (43 files, 6 suites)
│   ├── conftest.py                   # Shared fixtures
│   ├── test_api.py                   # API endpoint integration tests
│   ├── test_api_errors.py            # Error handling (NaN, oversized, traversal)
│   ├── test_inference.py             # EI computation, source activity tests
│   ├── test_model.py                 # Model loading/shape/finiteness tests
│   ├── test_xai.py                   # XAI occlusion tests
│   ├── mock_data/                    # Test fixtures (5 .npy files)
│   ├── unit/                         # 18 files: model components, metrics, losses
│   ├── functional/                   # 4 files: pipeline stages, data flow
│   ├── integration/                  # 3 files: API endpoints, model loading
│   ├── regression/                   # 3 files: output stability, format compat
│   └── system/                       # 4 files: determinism, memory, throughput
│
├── config.yaml                      # Master configuration (164 lines)
├── requirements.txt                 # Pipeline Python dependencies (39 lines)
├── start.sh                         # Orchestrator script (598 lines)
├── pytest.ini                       # Test configuration + markers
└── README.md                        # This file
```

---

## Running Tests

```bash
# Activate environment first (if using conda)
conda activate deepsif

# All 149 tests
pytest tests/ -v

# Exclude slow tests (~2 minutes)
pytest tests/ -m "not slow" -v

# Just unit tests (~2 seconds)
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_dle_metric.py -v

# Integration tests (requires model checkpoint)
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Pipe test output to a file
pytest tests/ -v 2>&1 | tee test_results.txt
```

**Test markers** (from `pytest.ini`):

| Marker | Description | Speed |
|--------|-------------|-------|
| `unit` | Fast, isolated, no I/O | <1s each |
| `functional` | Component integration | 1-5s each |
| `system` | Full pipeline, needs checkpoints | 5-60s each |
| `integration` | End-to-end, needs backend | 5-30s each |
| `slow` | >5s each (excluded with `-m "not slow"`) | Variable |
| `regression` | Bug-triggered regression tests | 1-5s each |

---

## Key Files & Their Roles

### Data files

| File | Shape | Role |
|------|-------|------|
| `data/leadfield_19x76.npy` | (19, 76) | Forward model: source → EEG projection |
| `data/connectivity_76.npy` | (76, 76) | Structural connectivity for TVB simulations |
| `data/region_centers_76.npy` | (76, 3) | MNI coordinates for DLE centroid calculation |
| `data/region_labels_76.json` | [str × 76] | Human-readable region names |
| `data/tract_lengths_76.npy` | (76, 76) | Tract lengths for distance-weighted analysis |
| `data/synthetic3/train_dataset.h5` | ~80k × (19,400) | Training EEG windows |
| `data/synthetic3/val_dataset.h5` | ~10k × (19,400) | Validation EEG windows |
| `data/synthetic3/test_dataset.h5` | ~10k × (19,400) | Test EEG windows |

### Model files

| File | Size | Role |
|------|------|------|
| `outputs/models/checkpoint_best.pt` | 4.8 MB | Best-validation-loss model weights |
| `outputs/models/checkpoint_latest.pt` | 4.8 MB | Final epoch model weights |
| `outputs/models/normalization_stats.json` | 297 B | `eeg_mean`, `eeg_std`, `src_mean`, `src_std` |

### Scripts

| Script | Phase | Input | Output |
|--------|-------|-------|--------|
| `01_build_leadfield.py` | 1 | TVB default connectivity | 5 data/ files |
| `02_generate_synthetic_data.py` | 1 | leadfield, connectivity | 3 HDF5 files |
| `03_train_network.py` | 2 | HDF5 files, leadfield | checkpoint + norm stats |
| `06_run_validation.py` | 5 | checkpoint, test data | Validation metrics |
| `08_run_cmaes.py` | 4 | checkpoint, connectivity | Fitted x₀ vector + EI |
| `08_apply_spatial_gradients.py` | Util | HDF5 data | Gradient-augmented data |
| `09_validate_enhanced_data.py` | Util | HDF5 data | Spectral/spatial validation |
| `10_final_validation.py` | 5 | checkpoint, test data | Comprehensive validation |
| `11_plot_fresh_samples.py` | 5 | checkpoint, data | Sample visualizations |
| `12_generate_validation_figures.py` | 5 | checkpoint, test data | 4-6 validation figures |
| `demo_biomarker_detection.py` | Demo | checkpoint, test data | 3D brain heatmap |
| `diag_gradient_audit.py` | Util | HDF5 data | Gradient diagnostics |

### Config file (`config.yaml`)

The 164-line YAML configuration controls every hyperparameter. Key sections:

| Section | What it controls |
|---------|-----------------|
| `forward_model` | 10-20 channel names, montage, reference type |
| `network.spatial_module` | MLP hidden dims [128,256,256,128], ReLU, BatchNorm, skip connections |
| `network.temporal_module` | BiLSTM: 2 layers, hidden=76, dropout=0.1 |
| `neural_mass_model` | Epileptor dt=0.1ms, x₀ ranges, coupling, noise |
| `synthetic_data` | 23k/2.9k/2.9k sim splits, 200Hz, 2s windows, 5 windows/sim |
| `training` | batch=64, epochs=200, lr=0.001, AdamW, early_stop=40 |
| `training.loss_weights` | α=1.0, β=0.0, γ=0.01, δ=1.0 |
| `training.physics_sub_weights` | λ_amplitude=0.2, λ_laplacian=0.0, λ_temporal=0.3 |
| `parameter_inversion` | CMA-ES: pop=14, max_gen=30, bounds=[-2.4,-1.0] |

---

## Config Reference

The final optimal configuration (`config.yaml`) uses:

```
Loss weights:
  α_source = 1.0       # Source reconstruction MSE
  β_forward = 0.0      # Forward consistency (DISABLED — leadfield provides spatial prior)
  γ_physics = 0.01     # Physics regularisation weight
  δ_epi = 1.0         # Epileptogenicity classification BCE

Physics sub-weights:
  λ_amplitude = 0.2    # Amplitude regularisation
  λ_laplacian = 0.0    # Spatial smoothness (DISABLED — DC offset is the spatial prior)
  λ_temporal = 0.3     # Temporal smoothness
```

---

## Deployment

### Docker Compose (recommended for production)

```bash
cd deploy

# Copy and edit environment
cp .env.example .env
# Edit .env — set backend URL, port mapping, etc.

# Build and start both services
docker compose up --build

# Services:
#   Backend:  http://localhost:8000  (healthcheck: 30s interval, 10s timeout, 3 retries)
#   Frontend: http://localhost:3000  (depends_on backend with condition: healthy)

# Stop
docker compose down
```

### Manual production

```bash
# Backend (use a production ASGI server)
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend (build and serve)
cd frontend && npm run build && npm start
```

### Deploy on a single VPS behind nginx

```bash
# 1. Docker compose up as above
# 2. Install nginx, add config:
#    - Proxy /api/* and /ws/* to localhost:8000
#    - Proxy everything else to localhost:3000
# 3. certbot --nginx for HTTPS
```

---

## Troubleshooting

### `start.sh --check` reports missing files

Each missing file has a specific fix:
- **Source space files**: Run `python scripts/01_build_leadfield.py`
- **Synthetic data**: Run `python scripts/02_generate_synthetic_data.py`
- **Model checkpoint**: Run `python scripts/03_train_network.py`
- **Node modules**: Run `npm install --legacy-peer-deps` in `frontend/`

### Backend crashes on startup with "module not found"

```bash
conda activate deepsif
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### CUDA out of memory during training

```bash
# Reduce batch size
python scripts/03_train_network.py --batch-size 32 --device cuda

# Or train on CPU (slower but works)
python scripts/03_train_network.py --device cpu
```

### SWC binary error when starting frontend

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

### Port 8000 or 3000 already in use

```bash
# Kill previous instances
./start.sh --kill

# Or manually:
pkill -f "server.py"   # backend
pkill -f "next dev"    # frontend
```

### "CUDA not available" but you have a GPU

```bash
# Reinstall PyTorch with CUDA matching your driver
pip uninstall torch torchvision
conda install pytorch==2.1.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# Check: python -c "import torch; print(torch.cuda.is_available())"
```

### MNE leadfield build fails

The BEM model download requires internet access. If behind a proxy:
```bash
export MNE_DATA=/path/to/mne_data
python -c "import mne; mne.datasets.fetch_fsaverage()"   # one-time download
```

### EDF upload returns 413

The backend limits uploads to 100 MB. Reduce window count or use a smaller file.

---

## Documentation

Full documentation is available in the `docs/` directory (rebuild from source if empty). Project overview and technical specifications are generated from the codebase's docstrings and configuration files.

**Key scientific references**: Sun et al. (2022, PNAS) — DeepSIF architecture. Jirsa et al. (2014) — Epileptor model. Desikan et al. (2006) — DK parcellation atlas.

---

## License

This is an academic project. See individual file headers for authorship.
