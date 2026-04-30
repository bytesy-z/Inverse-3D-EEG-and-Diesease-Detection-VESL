# PhysDeepSIF — Physics-Informed Deep Source Imaging Framework

Physics-constrained deep learning for EEG source localization and patient-specific epileptogenicity mapping. Uses a 76-region brain parcellation (Desikan-Killiany atlas) and the Epileptor neural mass model in The Virtual Brain (TVB) to produce biophysically grounded epileptogenicity heatmaps from 19-channel scalp EEG (10-20 montage, linked-ear reference).

**Final year project submission.** Tag: `v2.0-submission`.

---

## Quick Start

```bash
# One command — runs dependency checks, starts both backend (port 8000) and frontend (port 3000)
./start.sh
```

### Start modes

```bash
./start.sh --check      # Run dependency & file checks only
./start.sh --backend    # Start backend only (port 8000)
./start.sh --frontend   # Start frontend only (port 3000)
./start.sh --kill       # Kill servers started by start.sh
```

### Health check

```bash
curl -sS http://127.0.0.1:8000/api/health
# {"status":"ok","model_loaded":true,...}
```

### Quick inference from test dataset

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "sample_idx=10" \
  -F "mode=biomarkers"
```

### Upload an EDF patient file

```bash
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@data/test_demo.edf" \
  -F "mode=source_localization"
```

---

## Running Tests

```bash
# All 149 tests
pytest tests/ -v

# Exclude slow tests
pytest tests/ -m "not slow" -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run the full end-to-end test (requires both servers running)
./scripts/test_e2e.sh
```

Tests are organized by type:
- `tests/unit/` — 18 files (model components, metrics, data loading, TVB simulation)
- `tests/functional/` — 4 files (pipeline stages, data flow)
- `tests/integration/` — 3 files (API endpoints, model loading)
- `tests/regression/` — 3 files (output stability)
- `tests/system/` — 4 files (end-to-end pipelines)
- `tests/error/` — 1 file (error handling)

---

## Project Structure

```
fyp-2.0/
├── src/
│   ├── phase1_forward/         # TVB Epileptor simulation + synthetic dataset generation
│   ├── phase2_network/         # PhysDeepSIF neural network + training
│   ├── phase3_inference/       # Inference pipeline for EEG → source estimates
│   ├── phase4_inversion/       # CMA-ES patient-specific parameter optimization
│   ├── phase5_validation/      # Validation metrics, DLE computation, figures
│   └── xai/                    # EEG occlusion XAI (channel importance + time heatmap)
├── backend/
│   ├── server.py               # FastAPI backend (port 8000)
│   └── requirements.txt        # Python dependencies
├── frontend/
│   └── app/                    # Next.js dashboard (port 3000)
├── deploy/
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Dockerfile.backend      # Backend container
│   └── Dockerfile.frontend     # Frontend container
├── scripts/
│   └── test_e2e.sh             # End-to-end integration test
├── docs/                       # Full documentation (see docs/ for index)
├── data/
│   ├── leadfield_19x76.npy     # 19×76 leadfield matrix (BEM forward model)
│   ├── connectivity_76.npy     # 76×76 structural connectivity (DTI tractography)
│   ├── region_labels_76.json   # Region names by index
│   ├── region_centers_76.npy   # Region XYZ coordinates
│   └── samples/                # Demo EEG files
├── outputs/
│   ├── models/                 # Trained model checkpoint + normalization stats
│   └── figures/                # Validation figures (6 required)
├── tests/                      # 149 tests (unit, functional, integration, system)
├── start.sh                    # Orchestrator script
└── README.md
```

---

## Architecture

```
EDF upload (19ch, 200Hz)
    │
    ▼
┌─────────────────────────────┐
│  PhysDeepSIF Inference       │  Phase A: inverse solver (NN)
│  19ch EEG → 76-region source │  Instantaneous biomarker detection
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  CMA-ES Biophysical Fitting  │  Phase B: TVB model inversion
│  Optimize x₀ for each region │  Patient-specific EI heatmap
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Concordance Analysis        │  Overlap Phase A ∩ Phase B
│  HIGH / MODERATE / LOW       │  → clinical confidence tier
└─────────────────────────────┘
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/01_PLAIN_LANGUAGE_DESCRIPTION.md` | Plain-language project overview for evaluators |
| `docs/02_TECHNICAL_SPECIFICATIONS.md` | Full technical specification (14 sections) |
| `docs/03_EXPERIMENTATION_LOGS.md` | Training experiment log (12 sections, SVD math appendix) |
| `docs/30thaprplan.md` | Final model configuration, execution plan |
| `docs/SYSTEM_ARCHITECTURE.md` | Codebase map with file-level detail |
| `docs/FINAL_WORK_PLAN_v2.md` | Wave 4 completion tracker |
| `AGENTS.md` | Agent/developer runbook for automated tooling |

---

## Requirements

- **Python 3.9+** with PyTorch (CUDA recommended)
- **Node.js 20+** with npm
- 6 required data files (checked at startup):
  - `outputs/models/checkpoint_best.pt`
  - `outputs/models/normalization_stats.json`
  - `data/leadfield_19x76.npy`
  - `data/connectivity_76.npy`
  - `data/region_labels_76.json`
  - `data/region_centers_76.npy`

---

## Key Scientific Results

- **DLE (Distance Localisation Error)**: 31.06 mm (centroid DLE, asymmetric mask)
- **AUC**: 0.923 (epileptogenic vs healthy region classification)
- **Top-5 Recall**: 0.822
- **Run inference**: `./start.sh` then visit http://localhost:3000

---

## Deployment

```bash
# Docker Compose (both services + healthcheck)
cd deploy && docker compose up --build
```

Deployment details in `deploy/docker-compose.yml` and `docs/02_TECHNICAL_SPECIFICATIONS.md` §11.

---

## License

This is an academic project. See individual file headers for authorship.
