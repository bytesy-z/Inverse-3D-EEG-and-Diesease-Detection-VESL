# PhysDeepSIF Recovery Plan

## Goal

Revert to commit `e820166` (clean pre-mega-commit state) and selectively reapply good features from the mega-commit `491e6fa` ("yes.") while avoiding broken animation/brain-viz changes introduced in subsequent failed fix commits.

**Key principle**: Start from clean `e820166`, add only what's needed. Do NOT re-apply broken layout changes to brain-viz or the sticky sidebar layout in the biomarkers view.

---

## Repository Layout

```
fyp-2.0/
├── backend/
│   └── server.py              # FastAPI backend (1984→2600 lines)
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   └── analysis/
│   │       └── page.tsx        # Analysis dashboard (392→607 lines)
│   ├── components/
│   │   ├── brain-visualization.tsx   # 3D brain renderer (394→~500 lines)
│   │   ├── xai-panel.tsx             # NEW — XAI explainability panel
│   │   ├── concordance-badge.tsx     # NEW — CMA-ES concordance badge
│   │   ├── error-boundary.tsx        # NEW — React error boundary
│   │   ├── analysis-skeleton.tsx     # NEW — Loading skeleton
│   │   └── ...                       # other components unchanged
│   ├── hooks/
│   │   └── use-websocket.ts          # NEW — WebSocket hook for async jobs
│   ├── lib/
│   │   └── job-store.ts              # Type definitions (91→120 lines)
│   └── package.json
├── src/
│   ├── phase4_inversion/             # NEW — CMA-ES biophysical inversion
│   │   ├── __init__.py
│   │   ├── cmaes_optimizer.py
│   │   ├── objective_function.py
│   │   ├── epileptogenicity_index.py
│   │   └── concordance.py
│   ├── phase5_validation/            # NEW — Validation figure generation
│   │   └── generate_figures.py
│   ├── xai/
│   │   └── eeg_occlusion.py          # UPDATED — XAI occlusion
│   └── phase2_network/
│       ├── metrics.py                # UPDATED
│       ├── loss_functions.py         # UPDATED
│       └── 03_train_network.py       # UPDATED
├── scripts/
│   ├── 08_run_cmaes.py               # NEW
│   └── 12_generate_validation_figures.py  # NEW
├── config.yaml                       # UPDATED
├── outputs/models/
│   ├── checkpoint_best.pt            # UPDATED
│   └── normalization_stats.json     # UPDATED (v2 dual raw+AC format)
└── data/
    └── tract_lengths_76.npy          # Must already exist

Recovery kit reference files: /home/zik/fyp-recovery-kit/
```

---

## PHASE 1: Backup & Reset (USER runs these)

Run these commands from the repo root: `/home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0`

### Step 1.1 — Kill all running servers

```bash
# Kill backend
pkill -f "uvicorn" 2>/dev/null
pkill -f "python.*server.py" 2>/dev/null || true

# Kill frontend dev server
pkill -f "next dev" 2>/dev/null
pkill -f "node.*next" 2>/dev/null || true

# Verify ports are free
ss -tlnp | grep -E ':(8000|3000)\s' && echo "WARNING: ports still in use" || echo "Ports clear"
```

### Step 1.2 — Create backup branch

```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0

# Create backup of current state (the failed fix attempts)
git branch backup-failed-fixes HEAD

# Verify
git branch | grep backup
```

### Step 1.3 — Hard reset to e820166

```bash
# WARNING: This discards ALL changes since e820166
git reset --hard e820166
```

### Step 1.4 — Verify clean state

```bash
# Should show 1984 lines
wc -l backend/server.py

# Should show 392 lines
wc -l frontend/app/analysis/page.tsx

# Should show 394 lines
wc -l frontend/components/brain-visualization.tsx

# Should NOT exist
test -d src/phase4_inversion && echo "ERROR: phase4_inversion still exists" || echo "OK: phase4_inversion gone"
test -f frontend/components/xai-panel.tsx && echo "ERROR: xai-panel still exists" || echo "OK: xai-panel gone"
test -f frontend/hooks/use-websocket.ts && echo "ERROR: use-websocket still exists" || echo "OK: use-websocket gone"

# Should show clean working tree
git status
```

---

## PHASE 2: Drop-in Files (USER runs these)

These files need NO editing — just copy them from the recovery kit to the repo.

All source files are at: `/home/zik/fyp-recovery-kit/`

### Step 2.1 — Phase 4 CMA-ES module (entire new directory)

```bash
REPO=/home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0
KIT=/home/zik/fyp-recovery-kit

mkdir -p $REPO/src/phase4_inversion
cp $KIT/phase4_inversion/__init__.py $REPO/src/phase4_inversion/
cp $KIT/phase4_inversion/cmaes_optimizer.py $REPO/src/phase4_inversion/
cp $KIT/phase4_inversion/objective_function.py $REPO/src/phase4_inversion/
cp $KIT/phase4_inversion/epileptogenicity_index.py $REPO/src/phase4_inversion/
cp $KIT/phase4_inversion/concordance.py $REPO/src/phase4_inversion/
```

### Step 2.2 — Phase 5 Validation module

```bash
mkdir -p $REPO/src/phase5_validation
cp $KIT/phase5_validation/generate_figures.py $REPO/src/phase5_validation/
```

### Step 2.3 — Frontend NEW components

```bash
cp $KIT/frontend-components/xai-panel.tsx $REPO/frontend/components/xai-panel.tsx
cp $KIT/frontend-components/concordance-badge.tsx $REPO/frontend/components/concordance-badge.tsx
cp $KIT/frontend-components/error-boundary.tsx $REPO/frontend/components/error-boundary.tsx
cp $KIT/frontend-components/analysis-skeleton.tsx $REPO/frontend/components/analysis-skeleton.tsx
```

### Step 2.4 — Frontend NEW hook

```bash
mkdir -p $REPO/frontend/hooks
cp $KIT/frontend-hooks/use-websocket.ts $REPO/frontend/hooks/use-websocket.ts
```

### Step 2.5 — Scripts

```bash
cp $KIT/scripts/08_run_cmaes.py $REPO/scripts/08_run_cmaes.py
cp $KIT/scripts/12_generate_validation_figures.py $REPO/scripts/12_generate_validation_figures.py
```

### Step 2.6 — Training files (drop-in replacements)

These files had training improvements. Copy them from `$KIT/diffs/` into the repo:

```bash
# Training scripts — overwrite e820166 versions
cp $KIT/diffs/03_train_network.py $REPO/src/phase2_network/03_train_network.py
cp $KIT/diffs/metrics.py $REPO/src/phase2_network/metrics.py
cp $KIT/diffs/loss_functions.py $REPO/src/phase2_network/loss_functions.py
cp $KIT/diffs/eeg_occlusion.py $REPO/src/xai/eeg_occlusion.py
```

### Step 2.7 — Model files

```bash
cp $KIT/diffs/normalization_stats.json $REPO/outputs/models/normalization_stats.json
cp $KIT/diffs/checkpoint_best.pt $REPO/outputs/models/checkpoint_best.pt
```

### Step 2.8 — Config

```bash
cp $KIT/config.yaml $REPO/config.yaml
```

### Step 2.9 — Backend requirements

```bash
cp $KIT/backend/requirements.txt $REPO/backend/requirements.txt
```

### Step 2.10 — Verify drop-in files exist

```bash
test -f $REPO/src/phase4_inversion/__init__.py && echo "OK" || echo "MISSING"
test -f $REPO/frontend/components/xai-panel.tsx && echo "OK" || echo "MISSING"
test -f $REPO/frontend/hooks/use-websocket.ts && echo "OK" || echo "MISSING"
test -f $REPO/scripts/08_run_cmaes.py && echo "OK" || echo "MISSING"
```

---

## PHASE 3: Agent-Led Rebuild (AGENT runs these)

The agent now rebuilds the files that need MANUAL merging. All editing happens in `$REPO` = `/home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0`.

### Section 3A — Rebuild `backend/server.py`

**Starting point**: The e820166 server.py (1984 lines).  
**Reference for desired state**: `/home/zik/fyp-recovery-kit/backend/server.py` (2600 lines).  
**Strategy**: Apply a sequence of 16 targeted edits to the e820166 server.py to add all good features. The edits are in ORDER from top to bottom of the file.

**Before starting**, verify the base file:
```bash
wc -l backend/server.py   # should print: 1984
```

#### Edit 1: Replace imports block (lines 20–66 in e820166)

Replace the entire imports section with the expanded imports from the current server.py. This adds:
- `from logging.handlers import RotatingFileHandler`
- `import shutil`
- `import threading`
- `from concurrent.futures import ThreadPoolExecutor, Future`
- `from collections import defaultdict`
- `Request` added to FastAPI imports

**Replace** lines 20–66 (from `import json` through `from src.region_names import ...`):

```python
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO
import base64

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

# Suppress warnings before importing heavy libs
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# FastAPI imports
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import asyncio

# Ensure project root is on the path so local imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import torch
import torch.nn as nn
from scipy import linalg as la

# Optional: for waveform rendering (MNE/matplotlib)
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

# Local imports — PhysDeepSIF model builder
from src.phase2_network.physdeepsif import build_physdeepsif
from src.region_names import get_region_name, format_region_for_display

# CMA-ES biophysical inversion for concordance validation
from src.phase4_inversion import (
    fit_patient, ProgressCallback,
    compute_biophysical_ei, compute_concordance,
)
```

#### Edit 2: Replace logging config (lines 71–76 in e820166)

Replace the `logging.basicConfig(...)` block with:

```python
# ========================================================================
# Logging (JsonFormatter class must be defined before use)
# ========================================================================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        return json.dumps(log_entry)


logger = logging.getLogger("physdeepsif_api")
logger.handlers.clear()
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

# File handler with rotation (10MB, 5 backups)
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(
    str(log_dir / "backend.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setFormatter(JsonFormatter())
logger.addHandler(file_handler)
```

#### Edit 3: Extend constants section (lines 79–106 in e820166)

After `RESULTS_DIR` definition on line 105, INSERT the new constants:

```python
MIN_FREE_BYTES = 100 * 1024 * 1024

TRACT_LENGTHS_PATH = PROJECT_ROOT / "data" / "tract_lengths_76.npy"

# CMA-ES configuration (matches config.yaml)
CMAES_POPULATION_SIZE = 2  # reduced for debugging (was 14)
CMAES_MAX_GENERATIONS = 30
CMAES_INITIAL_X0 = -2.1
CMAES_INITIAL_SIGMA = 0.3
CMAES_BOUNDS = (-2.4, -1.0)
CMAES_SEED = 42

START_TIME = time.time()
```

NOTE: Also modify the `MAX_EDF_UPLOAD_BYTES` line to keep it but ensure `MIN_FREE_BYTES` and `TRACT_LENGTHS_PATH` come after `RESULTS_DIR`.

Also, on the `logging.basicConfig` removal line, keep `CHANNEL_NAMES` and `CHECKPOINT_PATH` through `RESULTS_DIR` exactly as they are, then INSERT the block above between `RESULTS_DIR` and the `class WaveformRequest`.

#### Edit 4: Add `_render_waveform_mne_html` function

Insert AFTER the `_render_waveform_png` function (after its closing check at line 161):

```python
def _render_waveform_mne_html(eeg_data: NDArray[np.float32], sfreq: float, uV_per_div: int = 50) -> str:
    import mpld3
    import mne
    from mne.io import RawArray
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_ch = eeg_data.shape[0]
    ch_names = CHANNEL_NAMES[:n_ch] if len(CHANNEL_NAMES) >= n_ch else [f"Ch{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=['eeg'] * n_ch)
    raw = RawArray(eeg_data.astype(np.float32), info)

    scalings = {'eeg': float(uV_per_div) * 10}

    plot_result = raw.plot(
        n_channels=n_ch,
        duration=eeg_data.shape[1] / float(sfreq),
        scalings=scalings,
        show=False,
        block=False,
    )

    if isinstance(plot_result, tuple):
        fig = plot_result[0]
    else:
        fig = plot_result

    fig_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return fig_html
```

#### Edit 5: Extend global state (lines 164–172 in e820166)

Replace the global state section:

```python
# ========================================================================
# Global model state (loaded once on startup)
# ========================================================================
model: Optional[nn.Module] = None
norm_stats: Optional[Dict[str, float]] = None
region_labels: Optional[List[str]] = None
leadfield_matrix: Optional[NDArray[np.float32]] = None
device: Optional[torch.device] = None
connectivity_weights: Optional[NDArray[np.float32]] = None
region_centers: Optional[NDArray[np.float32]] = None
tract_lengths: Optional[NDArray[np.float32]] = None

# In-memory job tracker
jobs: Dict[str, dict] = {}

# Thread lock for concurrent access to active_jobs dict
active_jobs_lock = threading.Lock()

# Background executor for CMA-ES (avoids blocking HTTP responses)
_cmaes_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cmaes")
```

NOTE: The `active_jobs` dict is declared later in the WebSocket section of e820166 (around line 1773). When making this edit, REMOVE the later `active_jobs: Dict[str, Dict] = {}` declaration at line 1773 since it's now here. We'll handle duplicates when we get there.

#### Edit 6: Add RateLimiter class and HTTP middleware (BEFORE the FastAPI app creation)

Insert AFTER the global state section and BEFORE `# ========================================================================` and `app = FastAPI(...)`:

```python

# ========================================================================
# Rate Limiter
# ========================================================================

class RateLimiter:
    def __init__(self, requests_per_minute=100):
        self.rate = requests_per_minute
        self.buckets: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self.buckets[key]
            cutoff = now - 60
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= self.rate:
                return False
            bucket.append(now)
            return True

rate_limiter = RateLimiter()
```

#### Edit 7: Update CORS middleware and add rate-limit middleware

**Replace** the CORS middleware block (around lines 185–198 in e820166):

```python
# CORS — allow the frontend dev servers to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:3010",
        "http://127.0.0.1:3010",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})
    return await call_next(request)
```

#### Edit 8: Add `startup_check` function and update the startup event

**Insert** the `startup_check` function BEFORE `@app.on_event("startup")`:

```python

def startup_check():
    required_files = {
        CHECKPOINT_PATH: "checkpoint",
        NORM_STATS_PATH: "normalization stats",
        LEADFIELD_PATH: "leadfield matrix",
        CONNECTIVITY_PATH: "connectivity",
        REGION_LABELS_PATH: "region labels",
        REGION_CENTERS_PATH: "region centers",
        TRACT_LENGTHS_PATH: "tract lengths",
    }
    all_ok = True
    for path, name in required_files.items():
        exists = path.exists()
        logger.info(f"  {name}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_ok = False
    if all_ok:
        logger.info("All required files present.")
    else:
        logger.warning("Some required files are missing.")
    return all_ok
```

Then **replace** the `startup_load_model` function (lines 204–280 in e820166) with this updated version:

```python
@app.on_event("startup")
async def startup_load_model():
    """Load the PhysDeepSIF model and all supporting data files on startup."""
    global model, norm_stats, region_labels, leadfield_matrix, device, connectivity_weights, region_centers, tract_lengths

    logger.info("=" * 60)
    logger.info("PhysDeepSIF API — Loading model and data...")
    logger.info("=" * 60)

    # Select device: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model checkpoint
    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Build the PhysDeepSIF model (creates architecture + registers buffers)
    # hidden_size=76 matches the final trained checkpoint (β=0.0, 410k params)
    model = build_physdeepsif(
        leadfield_path=str(LEADFIELD_PATH),
        connectivity_path=str(CONNECTIVITY_PATH),
        lstm_hidden_size=76,
        lstm_dropout=0.0,  # No dropout at inference time
    )

    # Load trained weights from checkpoint
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    logger.info(f"Model loaded: epoch={epoch}, val_loss={val_loss}")

    # Load normalization statistics (computed during training)
    if NORM_STATS_PATH.exists():
        with open(NORM_STATS_PATH, 'r') as f:
            norm_stats = json.load(f)
        logger.info(
            f"Normalization stats: EEG μ={norm_stats['eeg_mean']:.4f} "
            f"σ={norm_stats['eeg_std']:.4f}, "
            f"Src μ={norm_stats['src_mean']:.6f} "
            f"σ={norm_stats['src_std']:.6f}"
        )
    else:
        logger.error(f"Normalization stats not found at {NORM_STATS_PATH}")
        raise RuntimeError(f"Normalization stats not found: {NORM_STATS_PATH}")

    # Load region labels (76 DK region abbreviations)
    with open(REGION_LABELS_PATH, 'r') as f:
        region_labels = json.load(f)
    logger.info(f"Loaded {len(region_labels)} region labels")

    # Load leadfield matrix (19×76) for forward consistency checks
    leadfield_matrix = np.load(str(LEADFIELD_PATH)).astype(np.float32)
    logger.info(f"Loaded leadfield: shape {leadfield_matrix.shape}")

    # Load data files for CMA-ES biophysical inversion
    connectivity_weights = np.load(str(CONNECTIVITY_PATH)).astype(np.float32)
    region_centers = np.load(str(REGION_CENTERS_PATH)).astype(np.float32)
    tract_lengths = np.load(str(TRACT_LENGTHS_PATH)).astype(np.float32)
    logger.info(
        f"Loaded CMA-ES data: connectivity {connectivity_weights.shape}, "
        f"centers {region_centers.shape}, tracts {tract_lengths.shape}"
    )

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run startup validation checks (abort if any required files are missing)
    if not startup_check():
        raise RuntimeError("Startup check failed: required files are missing. Run ./start.sh --check for details.")

    logger.info("=" * 60)
    logger.info("PhysDeepSIF API — Ready to serve requests")
    logger.info("=" * 60)

    # Start periodic cleanup for WebSocket job tracker and old result files
    asyncio.create_task(_cleanup_stale_jobs())
    asyncio.create_task(_cleanup_old_results())
```

#### Edit 9: Replace `run_inference` (lines 287–336 in e820166)

Replace with the updated version that uses raw (DC+AC) stats, NO per-channel de-meaning:

```python
def run_inference(eeg: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Run PhysDeepSIF inference on EEG data.

    Applies: global z-score normalization using raw (DC+AC) training statistics.
    NO per-channel de-meaning — matches training pipeline (EEG retains DC spatial
    prior during training).

    Args:
        eeg: Raw EEG data, shape (19, 400) or (batch, 19, 400).

    Returns:
        Source activity estimate, shape (76, 400) or (batch, 76, 400).

    Raises:
        HTTPException(503): If model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not np.isfinite(eeg).all():
        raise HTTPException(status_code=400, detail="Input contains NaN or Inf values")
    # Guarantee batch dimension
    single_sample = eeg.ndim == 2
    if single_sample:
        eeg = eeg[np.newaxis, ...]  # (1, 19, 400)

    eeg_tensor = torch.from_numpy(eeg.astype(np.float32)).to(device)
    eps = 1e-7

    # Global z-score normalization using training statistics (RAW DC+AC).
    # NO per-channel de-meaning — matches training pipeline (EEG retains DC).
    # Training: EEG → z-score with raw (DC+AC) stats, sources → de-mean → z-score.
    # Inference: same EEG normalization → model → denormalize as AC-only sources.
    eeg_tensor = (eeg_tensor - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + eps)

    # Forward pass (inference mode — no gradient computation)
    with torch.no_grad():
        source_pred = model(eeg_tensor)  # (batch, 76, 400)

    # Denormalize: model outputs (de_meaned_src - src_mean) / src_std
    # Reverse: pred * src_std + src_mean → AC-only source activity
    source_pred = source_pred * (norm_stats['src_std'] + eps) + norm_stats['src_mean']

    source_np = source_pred.cpu().numpy()
    if single_sample:
        source_np = source_np[0]  # (76, 400)

    return source_np
```

#### Edit 10: Update `compute_epileptogenicity_index` (lines 339–462 in e820166)

Two minor changes:
1. Add `'scores_array_raw': ei_raw.tolist()` to the result dict (needed for XAI occlusion)
2. Add confidence floor: `threshold = max(threshold, 0.6)` after the `np.clip` line

**Find** this line (around line 574 in e820166):
```python
    threshold = np.clip(threshold, 0.0, 1.0)
```

**Insert** after it:
```python
    threshold = max(threshold, 0.6)  # Absolute confidence floor
```

**Find** the result dict, look for `'scores_array'` line, and **insert** after it:
```python
        'scores_array_raw': ei_raw.tolist(), # Pre-rescale sigmoid scores for XAI occlusion
```

So the two lines become:
```python
        'scores_array': ei_scores.tolist(),  # JSON-safe list, min-max rescaled to [0,1]
        'scores_array_raw': ei_raw.tolist(), # Pre-rescale sigmoid scores for XAI occlusion
```

#### Edit 11: Update `_process_edf_raw` signature (line 486 in e820166)

Change the function signature from:
```python
def _process_edf_raw(raw) -> dict:
```
to:
```python
def _process_edf_raw(raw, max_windows=None) -> dict:
```

And update the docstring to include the `max_windows` param. Then **find** the window truncation logic block and replace from `if total_windows > MAX_EDF_WINDOWS:` to the end of starts/timestamps, with:

```python
    if max_windows is not None and max_windows > 0:
        if total_windows > max_windows:
            starts = starts[:max_windows]
            truncated = True
        else:
            truncated = False
    elif total_windows > MAX_EDF_WINDOWS:
        keep = np.linspace(0, total_windows - 1, num=MAX_EDF_WINDOWS, dtype=int)
        starts = [starts[i] for i in keep]
        truncated = True
    else:
        truncated = False
```

#### Edit 12: Update `_set_job_status` with thread lock (lines 537–540 in e820166)

Replace:
```python
def _set_job_status(job_id: str, status: str, progress: int, message: str, **extra) -> None:
    """Set job status with timestamp for automatic cleanup."""
    active_jobs[job_id] = {"status": status, "progress": progress, "message": message, "_ts": time.time(), **extra}
```

with:
```python
def _set_job_status(job_id: str, status: str, progress: int, message: str, **extra) -> None:
    """Set job status with timestamp for automatic cleanup."""
    with active_jobs_lock:
        active_jobs[job_id] = {"status": status, "progress": progress, "message": message, "_ts": time.time(), **extra}
```

Also update `_cleanup_stale_jobs` to use the lock:

```python
async def _cleanup_stale_jobs():
    """Periodically purge stale jobs — terminal ones older than 5 min, any older than 30 min."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        with active_jobs_lock:
            stale = [
                jid for jid, job in list(active_jobs.items())
                if (
                    job.get("status") in ("completed", "failed")
                    and now - job.get("_ts", now) > 300
                ) or (
                    now - job.get("_ts", now) > 1800
                )
            ]
            for jid in stale:
                active_jobs.pop(jid, None)
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale WebSocket job(s)")
```

Insert after `_cleanup_stale_jobs`:

```python

async def _cleanup_old_results():
    while True:
        await asyncio.sleep(3600)
        now = time.time()
        for d in Path(RESULTS_DIR).iterdir():
            if d.is_dir():
                mtime = d.stat().st_mtime
                if now - mtime > 86400:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info(f"Cleaned up old result: {d.name}")
```

#### Edit 13: Add mesh cache (lines 564 in e820166)

Add a module-level cache before `_load_fsaverage5_mesh`:

```python
_mesh_cache = None
```

Wrap the function body:
```python
def _load_fsaverage5_mesh():
    global _mesh_cache
    if _mesh_cache is not None:
        return _mesh_cache
    # ... existing body ...
    _mesh_cache = (all_coords, all_faces, n_verts_lh, lh_coords, rh_coords)
    return _mesh_cache
```

#### Edit 14: Update `compute_source_activity_metrics` (lines 765–853 in e820166)

Update the function signature to accept global min/max:

```python
def compute_source_activity_metrics(
    source_activity: NDArray[np.float32],
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> dict:
```

Add `_debug_label: str = ""` parameter too.

In the normalization section, replace the existing normalization with:

```python
    if global_min is not None and global_max is not None:
        metric_min = global_min
        metric_max = global_max
    else:
        metric_min = primary_metric.min()
        metric_max = primary_metric.max()
```

#### Edit 15: CRITICAL — Update `generate_source_activity_heatmap_html` (lines 856–1083 in e820166)

This is the animation fix. Replace the function with the version that has `global_cmin`/`global_cmax` params.

**Key changes from e820166 version:**

a) Add `global_cmin` and `global_cmax` params to the function signature:

```python
def generate_source_activity_heatmap_html(
    activity_scores: NDArray[np.float32],
    title: str = "EEG Source Imaging — Estimated Brain Activity",
    animated_frames: Optional[List[NDArray[np.float32]]] = None,
    frame_timestamps: Optional[List[float]] = None,
    global_cmin: Optional[float] = None,
    global_cmax: Optional[float] = None,
) -> str:
```

b) In the `_scores_to_vertex_colors` inner function, log per-frame score ranges when a `_debug_tag` is provided.

c) Replace the hardcoded `cmin=0.0, cmax=1.0` in the Mesh3d trace with dynamic values:

```python
    use_global_range = global_cmin is not None and global_cmax is not None
    intensity_cmin = global_cmin if use_global_range else 0.0
    intensity_cmax = global_cmax if use_global_range else 1.0
```

And use `intensity_cmin`/`intensity_cmax` in the trace.

d) In the animation frames loop, when building each frame's Mesh3d, also set cmin/cmax on individual frames when `use_global_range` is True:

```python
            frame_mesh_kw: dict = dict(intensity=v_colors_frame)
            if use_global_range:
                frame_mesh_kw["cmin"] = intensity_cmin
                frame_mesh_kw["cmax"] = intensity_cmax
            frames.append(go.Frame(
                data=[go.Mesh3d(**frame_mesh_kw)],
                ...
```

e) Update the colorbar tickvals to show actual range values when global range is used.

**The full replacement function is in `/home/zik/fyp-recovery-kit/backend/server.py` lines 1046–1331.** The agent should copy that entire function (from `def generate_source_activity_heatmap_html` through the end of the function at the `return html_output` line) and replace the e820166 version.

#### Edit 16: Update `/api/health` (line 1090–1098 in e820166)

Replace the simple health check with:

```python
@app.get("/api/health")
async def health_check():
    """Health check — verify model is loaded and report system info."""
    results_path = Path("outputs/frontend_results")
    results_path.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(results_path)
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "checkpoint": str(CHECKPOINT_PATH),
        "disk_usage_bytes": du.used,
        "results_count": len(list(results_path.iterdir())) if results_path.exists() else 0,
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "degraded": model is None,
    }
```

#### Edit 17: Add helper functions before `/api/analyze`

INSERT the following helper functions AFTER the `generate_source_activity_heatmap_html` function and BEFORE `@app.get("/api/health")`:

These are `_build_eeg_payload`, `_extract_plotly_body`, `_run_xai`, `_make_cmaes_callback`, and `_run_cmaes_inversion`. Copy all five functions from `/home/zik/fyp-recovery-kit/backend/server.py` lines 1400–1552. These are the exact same functions.

#### Edit 18: Replace the `/api/analyze` endpoint (lines 1143–1677 in e820166)

This is the BIGGEST edit. The e820166 version has separate endpoints `/api/analyze`, `/api/analyze-eeg`, and `/api/physdeepsif`. The current version has a single unified `/api/analyze` with mode parameter and all the CMA-ES/XAI features.

The agent must replace the ENTIRE `/api/analyze` function (everything from `@app.post("/api/analyze")` through the end of the function at the final `raise HTTPException(status_code=500, detail=str(e))`) with the current version from `/home/zik/fyp-recovery-kit/backend/server.py` lines 1555–2121.

**Critical details to copy correctly:**
- The function signature adds: `max_windows`, `cmaes_generations`, `debug` params
- Inference runs via `asyncio.wait_for` and `run_in_executor` (not synchronous `run_inference`)
- Uses `_extract_plotly_body` for plot HTML extraction
- Uses `_build_eeg_payload` for EEG payload construction
- Source localization mode: Computes `all_win_sources` per-window RAW RMS, calculates `global_metric_min`/`global_metric_max`, passes to `generate_source_activity_heatmap_html` with `global_cmin`/`global_cmax`
- Biomarkers mode: Adds `_run_xai` call, CMA-ES launch (`_wants_cmaes`, `_cmaes_future` submission to `_cmaes_executor`), debug mode dummy concordance
- Uses structured logging with `extra={"job_id": job_id}`

#### Edit 19: Keep `/api/biomarkers` endpoint (around line 1679 in e820166)

The `/api/biomarkers` convenience endpoint delegates to `analyze_eeg` with `file=None`. Keep it the same — the e820166 version already delegates correctly.

#### Edit 20: Keep `/api/test-samples` (lines 1699–1744 in e820166)

No changes needed.

#### Edit 21: Add `/api/job/{job_id}/cmaes` polling endpoint

INSERT after the test-samples endpoint and BEFORE `/api/results/{path:path}`:

```python
@app.get("/api/job/{job_id}/cmaes")
async def poll_cmaes_results(job_id: str):
    """Poll CMA-ES concordance results for a running job.

    Returns:
        {status: "running"|"completed"|"failed"|"not_found",
         ...concordance/cmaes fields if completed...}
    """
    with active_jobs_lock:
        entry = active_jobs.get(job_id)
    if entry is None:
        return JSONResponse({"status": "not_found", "error": "Job not found"})

    future: Optional[Future] = entry.get("_cmaes_future")
    if future is None:
        # Check if we already have a cached result
        cached = entry.get("_cmaes_result")
        if cached is not None:
            return JSONResponse(cached)
        return JSONResponse({"status": "not_found", "error": "No CMA-ES running for this job"})

    if not future.done():
        # Still running — return progress from active_jobs
        return JSONResponse({
            "status": "running",
            "generation": entry.get("cmaes_generation", 0),
            "max_generations": entry.get("cmaes_max_generations", 30),
            "best_score": entry.get("cmaes_best_score"),
        })

    # Future is done — collect result
    try:
        result = future.result()
        response_data = {
            "status": "completed",
            "concordance": result["concordance"],
            "cmaes": {
                "status": "completed",
                "best_score": result["best_score"],
                "generations": result["generations"],
                "biophysical_ei": result["biophysical_ei"],
            },
        }
    except Exception as exc:
        logger.warning(f"CMA-ES failed for job {job_id}: {exc}")
        response_data = {
            "status": "failed",
            "concordance": None,
            "cmaes": {"status": "failed", "error": str(exc)},
        }

    # Cache result so repeated polls don't re-raise
    with active_jobs_lock:
        if job_id in active_jobs:
            active_jobs[job_id]["_cmaes_result"] = response_data

    return JSONResponse(response_data)
```

#### Edit 22: Add MNE waveform endpoint

INSERT after the CMA-ES polling endpoint and BEFORE the generic `/api/results/{path:path}`:

```python
@app.get("/api/results/{job_id}/mne-waveform/{window_idx}")
async def serve_mne_waveform(job_id: str, window_idx: int, scale: int = 50):
    """Render an EEG window as interactive MNE/mpld3 HTML."""
    job_dir = RESULTS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    edf_windows_path = job_dir / "edf_all_windows.npy"
    eeg_data_path = job_dir / "eeg_data.npy"

    if edf_windows_path.exists():
        all_windows = np.load(str(edf_windows_path))
        if window_idx < 0 or window_idx >= all_windows.shape[0]:
            raise HTTPException(status_code=400, detail=f"window_idx out of range [0, {all_windows.shape[0]})")
        window_data = all_windows[window_idx]
    elif eeg_data_path.exists():
        if window_idx != 0:
            raise HTTPException(status_code=400, detail="Only window 0 available (single-window data)")
        window_data = np.load(str(eeg_data_path))
    else:
        raise HTTPException(status_code=404, detail="No EEG data found for this job")

    html = _render_waveform_mne_html(window_data, sfreq=SAMPLING_RATE, uV_per_div=scale)
    return HTMLResponse(content=html)
```

#### Edit 23: Update WebSocket `_process_analysis_async` (lines 1776–1970 in e820166)

Replace the entire `_process_analysis_async` function and `websocket_endpoint` function.

The agent must copy from `/home/zik/fyp-recovery-kit/backend/server.py`:
- `_process_analysis_async` (lines 2306–2570) — the version with CMA-ES support, proper animation, `max_windows`/`cmaes_generations`/`debug` params
- `@app.websocket("/ws/{job_id}")` (lines 2573–2585) — the updated version with `active_jobs_lock`

**IMPORTANT**: Remove the duplicate `active_jobs: Dict[str, Dict] = {}` line at line 2303 (or wherever it ended up in current). Since we declared `active_jobs` earlier in the global state section, we don't want a duplicate.

The WebSocket section should start with comment line `# ========================================================================` and `# WebSocket for real-time job status`.

#### Edit 24: Update entry point (lines 1976–1984 in e820166)

Add `timeout_keep_alive`:

```python
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=30,
    )
```

#### Edit 25: Remove old `/api/analyze-eeg` and `/api/physdeepsif` route files

The old e820166 frontend API routes must be removed since we now use a single `/api/analyze`:

```bash
rm -f frontend/app/api/analyze/route.ts   # old dedicated /api/analyze route
# The e820166 frontend used: /api/analyze-eeg and /api/physdeepsif
# These are Next.js API route handlers that proxy to the backend.
# They exist at:
#   frontend/app/api/analyze-eeg/route.ts
#   frontend/app/api/physdeepsif/route.ts
```

Actually, check if these files exist. The frontend page.tsx references them directly. The new frontend will call the Python backend at `http://localhost:8000/api/analyze` directly (not through Next.js proxy routes).

---

### Section 3B — Rebuild `frontend/app/analysis/page.tsx`

**Starting point**: The e820166 page.tsx (392 lines).  
**Reference**: `/home/zik/fyp-recovery-kit/frontend-pages/analysis-page.tsx` (607 lines).  
**Goal**: Add CMA-ES/XAI features while KEEPING the simple layout from e820166 (NOT the sticky sidebar grid).

#### Edit B1: Update imports (lines 1–14 in e820166)

Replace the import block with extended imports that include new components:

```typescript
"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { RotateCcw, Brain, Activity, Zap, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
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
```

#### Edit B2: Update `ESIResult` interface (around line 28 in e820166)

Add `jobId` field:

```typescript
interface ESIResult {
  jobId: string
  fileName: string
  // ... rest same as e820166
}
```

#### Edit B3: Add new state variables (after line 54 in e820166)

After `const [viewMode, setViewMode] = useState<ViewMode>("source")`, insert:

```typescript
  const [maxWindows, setMaxWindows] = useState<number>(5)
  const [cmaesGens, setCmaesGens] = useState<number>(20)
  const [debugMode, setDebugMode] = useState<boolean>(false)
```

#### Edit B4: Replace `handleAnalyze` (lines 73–135 in e820166)

Replace the entire `handleAnalyze` callback with the version that:
- Uses parallel fetch to `/api/analyze` (not sequential `/api/analyze-eeg` + `/api/physdeepsif`)
- Sends `max_windows`, `cmaes_generations`, `debug` params
- Depends on `[selectedFile, maxWindows, cmaesGens, debugMode]`

```typescript
  /* ---- Run analysis ---- */
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return
    setStep("analyze")
    setError(null)

    try {
      const fdSource = new FormData()
      fdSource.append("file", selectedFile)
      fdSource.append("include_eeg", "true")
      fdSource.append("max_windows", String(maxWindows))
      fdSource.append("cmaes_generations", String(cmaesGens))
      fdSource.append("debug", debugMode ? "true" : "false")
      fdSource.append("ws", "false")
      fdSource.append("mode", "source_localization")

      const fdBio = new FormData()
      fdBio.append("file", selectedFile)
      fdBio.append("include_eeg", "true")
      fdBio.append("max_windows", String(maxWindows))
      fdBio.append("cmaes_generations", String(cmaesGens))
      fdBio.append("debug", debugMode ? "true" : "false")
      fdBio.append("ws", "false")
      fdBio.append("mode", "biomarkers")

      const backendUrl = process.env.NEXT_PUBLIC_PHYSDEEPSIF_BACKEND || "http://localhost:8000"

      const [sourceRes, bioRes] = await Promise.all([
        fetch(`${backendUrl}/api/analyze`, { method: "POST", body: fdSource }),
        fetch(`${backendUrl}/api/analyze`, { method: "POST", body: fdBio }),
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

      setEsiResult(sourceData)
      setBioResult(bioData)
      setStep("results")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during analysis")
      setStep("upload")
    }
  }, [selectedFile, maxWindows, cmaesGens, debugMode])
```

#### Edit B5: Add CMA-ES polling useEffect (BEFORE `setClampedWindow` or after handleReset)

Insert after `handleReset`:

```typescript
  /* ---- CMA-ES polling — when backend launches background CMA-ES, poll for results ---- */
  const cmaesPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  useEffect(() => {
    if (!bioResult?.cmaes || bioResult.cmaes.status !== "running" || !bioResult.jobId) {
      return
    }

    const jobId = bioResult.jobId
    const backendUrl = process.env.NEXT_PUBLIC_PHYSDEEPSIF_BACKEND || "http://localhost:8000"
    const poll = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/job/${jobId}/cmaes`)
        if (!res.ok) return
        const data = await res.json()
        if (data.status === "completed") {
          setBioResult((prev) => {
            if (!prev) return prev
            return {
              ...prev,
              concordance: data.concordance ?? prev.concordance,
              cmaes: data.cmaes ?? prev.cmaes,
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
```

#### Edit B6: Add `setClampedWindow` helper (after CMA-ES polling)

Insert:

```typescript
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
```

Update `handleBrainFrameChange` to use `setClampedWindow`:

```typescript
  const handleBrainFrameChange = useCallback((frameIndex: number) => {
    const total = esiResult?.eegData?.windows?.length ?? 0
    if (total <= 0) return
    const normalizedIndex = ((frameIndex % total) + total) % total
    setClampedWindow(normalizedIndex)
  }, [esiResult?.eegData?.windows?.length, setClampedWindow])
```

#### Edit B7: Update the Upload section (around line 209 in e820166)

After the `FileUploadSection` component and BEFORE the `Analyze EEG` button, add CMA-ES controls:

```tsx
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
```

#### Edit B8: Update the Processing step (around line 229 in e820166)

Replace `<ProcessingWindow />` with:

```tsx
            <>
              <ProcessingWindow
                elapsedTime={0}
                progress={0}
                status={"Running inference..."}
              />
              <AnalysisSkeleton />
            </>
```

#### Edit B9: Update the Biomarkers View (lines 334–383 in e820166)

This is the MOST IMPORTANT frontend edit. Replace the entire biomarkers view block with a version that:
- KEEPS the simple vertical layout (not sticky sidebar grid)
- Adds CMA-ES loading spinner when `bioResult?.cmaes?.status === "running"`
- Adds ConcordanceBadge when concordance is available
- Adds XaiPanel when xai is available
- Wraps things in Card components

Replace lines 334–383 (entire `{viewMode === "biomarkers" ...` block) with:

```tsx
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

                  <DetectedRegions
                    regions={detectedRegions}
                    variant="clinical"
                  />

                  {/* CMA-ES loading spinner */}
                  {bioResult?.cmaes?.status === "running" && (
                    <Card>
                      <div className="p-10 flex flex-col items-center justify-center gap-4 text-center">
                        <Loader2 className="h-10 w-10 animate-spin text-primary" />
                        <div>
                          <p className="text-base font-semibold">Running CMA-ES Biophysical Inversion</p>
                          <p className="text-sm text-muted-foreground mt-1.5 max-w-xs">
                            Validating concordance between heuristic and biophysical epileptogenicity markers
                          </p>
                        </div>
                      </div>
                    </Card>
                  )}

                  {/* Concordance Card */}
                  {bioResult?.concordance && (
                    <Card>
                      <div className="px-4 py-2 border-b">
                        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Biophysical Validation</span>
                      </div>
                      <div className="p-4 space-y-3">
                        <ConcordanceBadge
                          tier={bioResult.concordance.tier}
                          overlap={bioResult.concordance.overlap}
                          description={bioResult.concordance.tier_description}
                          sharedRegions={bioResult.concordance.shared_regions}
                        />
                      </div>
                    </Card>
                  )}

                  {/* XAI Panel */}
                  {bioResult?.xai && (
                    <Card>
                      <div className="px-4 py-2 border-b">
                        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Explainability (XAI)</span>
                      </div>
                      <div className="p-4">
                        <XaiPanel
                          channelImportance={bioResult.xai.channel_importance ?? []}
                          timeImportance={bioResult.xai.time_importance ?? []}
                          channelNames={esiResult?.eegData?.channels ?? DEFAULT_CHANNEL_NAMES}
                          topSegments={bioResult.xai.top_segments ?? []}
                          eegData={esiResult?.eegData ?? undefined}
                          selectedWindow={selectedWindow}
                        />
                      </div>
                    </Card>
                  )}
                </div>
              )}
```

#### Edit B10: Wrap the JSX in ErrorBoundary

Wrap the main content inside `<AppContainer>` with `<ErrorBoundary>`:

In the current e820166 code, the structure is:
```tsx
          <AppContainer>
            {/* ... all content ... */}
          </AppContainer>
```

Change to:
```tsx
          <ErrorBoundary>
          <AppContainer>
            {/* ... all content ... */}
          </AppContainer>
          </ErrorBoundary>
```

#### Edit B11: Update source localization view to handle missing esiResult gracefully

In the source localization view (around lines 307-331), add fallback to `bioResult` fields when `esiResult` is partially available. The current e820166 code accesses `esiResult?.eegData` and `esiResult?.plotHtml` — add fallback checks:

The grid row (line 308) should keep `viewMode === "source"` but accept both `esiResult` and `bioResult` as data sources:

```tsx
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
                    return (
                      <BrainVisualization
                        plotHtml={html}
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
```

#### Edit B12: Verify the page compiles

```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0/frontend
npm run build 2>&1 | tail -20
```

Fix any TypeScript/import errors (e.g., missing `@/components/ui/input`, `@/components/ui/checkbox`, `@/components/ui/card` — these should already exist as they're common shadcn/ui components).

---

### Section 3C — Update `frontend/lib/job-store.ts`

Replace the e820166 version (91 lines) with the recovery kit version (`/home/zik/fyp-recovery-kit/frontend-lib/job-store.ts`, 120 lines). The key additions are:

- `heuristic_ei_scores?: number[]`
- `concordance?: { tier: string; overlap: number; shared_regions?: string[]; tier_description?: string } | null`
- `cmaes?: { status: string; best_score?: number; generations?: number; biophysical_ei?: number[]; error?: string } | null`
- `xai?: { channel_importance: number[]; time_importance: number[]; ... } | null`

**Command**: Copy the file:
```bash
cp /home/zik/fyp-recovery-kit/frontend-lib/job-store.ts /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0/frontend/lib/job-store.ts
```

---

### Section 3D — Update `frontend/package.json`

The current e820166 `package.json` may be missing `plotly.js-dist-min`. Compare with the recovery kit version at `/home/zik/fyp-recovery-kit/frontend-pages/package.json`.

**Action**: Copy the recovery kit package.json:
```bash
cp /home/zik/fyp-recovery-kit/frontend-pages/package.json /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0/frontend/package.json
```

Then run:
```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0/frontend
npm install --legacy-peer-deps
```

---

### Section 3E — Update `frontend/app/page.tsx` (landing page)

The current e820166 mainpage (shown at `/home/zik/fyp-recovery-kit/diffs/mainpage_e820166.tsx`) actually has the SAME content as the recovery kit version (`/home/zik/fyp-recovery-kit/frontend-pages/mainpage.tsx`). No changes needed — keep e820166 version.

---

### Section 3F — `frontend/components/brain-visualization.tsx` — KEEP AS-IS

**DO NOT MODIFY** the e820166 brain-visualization.tsx. It already has frame sync and works correctly. The diff to current (516-line patch) is almost entirely debug logging/`console.log` statements with only 2 minor behavioral changes:

1. Wrapping `Plotly.animate`/`relayout` to catch promise rejections
2. Avoiding `relayout` mid-animation with `isAnimatingRef`

These are defensive bug fixes for edge cases, not essential. KEEP the clean e820166 version.

---

### Section 3G — Reinstall Python dependencies

```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0
pip install -r backend/requirements.txt
# Or use the conda python from start.sh if applicable
```

---

## PHASE 4: Verification (USER runs)

### Step 4.1 — Start backend

```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0
# Use the start.sh wrapper
./start.sh --backend
```

Wait for backend to start. Verify health:
```bash
curl -sS http://127.0.0.1:8000/api/health | python3 -m json.tool
```

Expected output: `{"status": "ok", "model_loaded": true, ...}` with `disk_usage_bytes`, `uptime_seconds`, and other new fields.

### Step 4.2 — Start frontend

```bash
./start.sh --frontend
```

### Step 4.3 — Test source localization with animation

1. Open `http://localhost:3000/analysis` in browser
2. Upload an EDF file
3. Set "Max windows:" to 5 (lower = faster for testing)
4. Click "Analyze EEG"
5. Verify:
   - **Animation**: Source Localization tab shows animated brain with visibly DIFFERENT colors per frame (not all identical). Click Play and confirm the brain animation shows changing intensity patterns.
   - **Frame sync**: Clicking EEG waveform segments changes the brain view to the corresponding time window.
   - **Playback speed**: Speed buttons (0.5x, 1x, 2x, 4x) change animation speed.

### Step 4.4 — Test biomarkers with CMA-ES

1. Switch to "Biomarker Detection" tab
2. Verify:
   - **Epileptogenicity heatmap** renders (red/gray brain with top-5 regions highlighted)
   - **Detected Regions** card appears with region names
   - **CMA-ES loading spinner** appears briefly (or "Running CMA-ES Biophysical Inversion" card)
   - After CMA-ES completes, **Biophysical Validation** card appears with concordance tier (HIGH/MODERATE/LOW) and overlap count
   - **Explainability (XAI)** card appears with channel importance bars

### Step 4.5 — Test debug mode

1. Click "New Analysis"
2. Check "Debug mode (skip CMA-ES, dummy concordance)"
3. Run analysis again
4. Verify: Concordance shows "HIGH CONCORDANCE" immediately (dummy), no polling spinner.

### Step 4.6 — Verify no broken layout

1. On Biomarker Detection tab, confirm the layout is **vertical** (brain, then regions, then CMA-ES, then XAI) — NOT a sticky sidebar. The brain should scroll naturally with the page.
2. The page should NOT crash or show ErrorBoundary fallback.
3. Resize the window — the layout should adapt responsively but stay simple.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Backend crashes on startup: `tract_lengths_76.npy` missing | File not in `data/` | Ensure `data/tract_lengths_76.npy` exists |
| Frontend build error: `Cannot find module '@/components/ui/input'` | shadcn/ui component not generated | Run `npx shadcn@latest add input checkbox card` in `frontend/` |
| Animation all frames look identical | `global_cmin`/`global_cmax` not passed to heatmap function | Check Edit 15 was applied correctly; the `all_win_sources` loop must compute RAW RMS (not normalized scores) |
| CMA-ES polling never completes | `_cmaes_executor` not created or `_cmaes_future` not stored in `active_jobs` | Check Edit 5 (global state) and Edit 18 (/api/analyze biomarkers path) |
| `RateLimiter` missing | Edit 6 not applied | Add the RateLimiter class before FastAPI app creation |
| Old Next.js API routes causing conflicts | e820166 had proxy routes at `/api/analyze-eeg/*` and `/api/physdeepsif/*` | Delete `frontend/app/api/analyze-eeg/` and `frontend/app/api/physdeepsif/` directories if they exist |

---

## Git Commit (after verification passes)
```bash
cd /home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0
git add -A
git commit -m "Recovery: reapply CMA-ES, XAI, concordance, animation fix from mega-commit without broken layout changes"
```
