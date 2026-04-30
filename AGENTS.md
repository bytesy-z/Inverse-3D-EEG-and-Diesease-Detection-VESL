AGENTS — Repo-specific agent runbook

Purpose
- Short, high-signal instructions an automated agent or new developer must not miss.

Quick start (recommended)
1. Ensure required data + model files exist (see "Required files").
2. From repo root run the orchestrator which performs checks and starts both servers:
   ./start.sh

Useful start modes
- ./start.sh                Start backend + frontend (runs checks first)
- ./start.sh --backend      Start backend only
- ./start.sh --frontend     Start frontend only
- ./start.sh --check        Run dependency & file checks only
- ./start.sh --kill         Kill servers started by the script

Ports
- Backend: 8000 (FastAPI)
- Frontend: 3000 (Next.js)

Health check
- Backend: curl -sS http://127.0.0.1:8000/api/health
  Expect JSON: {"status":"ok","model_loaded":true,...}

One-off backend start (if you prefer manual)
- Use the start.sh wrapper whenever possible. If you must run manually, use the python used by start.sh (see start.sh for the exact path on the author machine) or your own venv:
  PYTHON=python3   # or the conda env python you use
  $PYTHON backend/server.py

Required files (exact paths start.sh and backend expect)
- outputs/models/checkpoint_best.pt
- outputs/models/normalization_stats.json
- data/leadfield_19x76.npy
- data/connectivity_76.npy
- data/region_labels_76.json
- data/region_centers_76.npy
- data/synthetic3/test_dataset.h5

If any of these are missing the backend will fail at startup. start.sh --check reports missing files.

Frontend / Node / SWC gotchas
- README in frontend/ recommends pnpm, but start.sh uses npm and will run npm install --legacy-peer-deps when node_modules are missing.
- Next.js uses a native SWC binary at node_modules/@next/swc-*/swc. If that binary is truncated (common when copying node_modules), start.sh contains a fix_swc_binary helper and will attempt to replace it. If frontend build errors reference swc or an unexpected ELF file error, delete node_modules and run npm install --legacy-peer-deps (or run the start.sh orchestrator which will attempt fixes).

Start order and validation
1. Start backend first (./start.sh --backend or ./start.sh). Validate via /api/health.
2. Start frontend (./start.sh --frontend or ./start.sh). The frontend expects the backend to be reachable; the UI will show a message if the backend is not running.

API quick examples (call backend directly)
- Health: curl -sS http://127.0.0.1:8000/api/health
- Run inference on a test sample index (uses test_dataset.h5):
  curl -sS -X POST "http://127.0.0.1:8000/api/analyze" -F "sample_idx=10" -F "mode=biomarkers"
- Upload an EDF/NPY/MAT file (example):
  curl -sS -X POST "http://127.0.0.1:8000/api/analyze" -F "file=@/path/to/sample.edf" -F "mode=source_localization"

Where results land
- The backend writes job artifacts into outputs/frontend_results/{job_id}/. The GET /api/results/{path} endpoint serves them.

Inference behaviour agents must not change
- Preprocessing applied before model inference: global z-score normalization using raw (DC+AC) training statistics from normalization_stats.json. NO per-channel de-meaning — matches training pipeline (EEG retains DC spatial prior during training).
- When returning predictions the backend denormalizes outputs to original scale.
- Sliding-window segmentation for EDF uploads: window_length=400 samples (2s at 200 Hz) with 50% overlap (step=200). The backend builds Plotly animation HTML for multi-window outputs and embeds only the body + inline Plotly script when returning to the frontend.

Common failure modes & remedies
- Backend crashes on startup: missing checkpoint/normalization file or other required data. Run ./start.sh --check to see missing items.
- Frontend build fails with SWC/native errors: delete node_modules and run npm install --legacy-peer-deps, or run ./start.sh which will attempt to fix SWC.
- Ports already in use: start.sh tries to kill prior processes it started; if ports are still blocked, find processes with ss or pgrep and kill them manually.

Agent / contributor constraints (important)
- Follow repository agent rules in .github/copilot-instructions.md. Notable items to preserve:
  - Do not auto-generate or overwrite docs unless explicitly requested.
  - Preserve coding style and commenting conventions found in the repo.
  - When making changes that affect endpoint behavior, update this AGENTS.md and the backend tests/README accordingly.

If you will run this on a different machine
- start.sh contains machine-specific paths (example: a hard-coded conda python path and NVM_DIR in the author's environment). Update the PYTHON path and NVM_DIR variables in start.sh or run the backend/frontend manually with your environment's python/node. Always run ./start.sh --check after adjustments.

If you want me to:
1) Commit this file for you (create a git commit) — say "commit".
2) Add example curl scripts in scripts/ for quick local testing — say "add scripts".
3) Draft a short production deploy checklist (Docker, ports, env vars) — say "deploy checklist".
