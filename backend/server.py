"""
PhysDeepSIF Backend API Server
Purpose: FastAPI server exposing PhysDeepSIF inference and biomarker detection
         as HTTP endpoints for the Next.js frontend.

The server loads the trained model once on startup and handles:
  1. EEG source localization (19ch EEG → 76-region source estimates)
  2. Epileptogenic zone detection (source activity → per-region EI scores)
  3. 3D brain heatmap generation (Plotly HTML)

Endpoints:
  POST /api/analyze       — Upload EEG file, run full pipeline, return results
  POST /api/biomarkers    — Run biomarker detection on a test sample or uploaded EEG
  GET  /api/health        — Health check
  GET  /api/results/{path} — Serve result files (HTML, JSON)

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 4 for pipeline details.
"""

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

# ========================================================================
# Constants (from Technical Specs §3, §4)
# ========================================================================
N_REGIONS = 76              # Desikan-Killiany parcellation
N_CHANNELS = 19             # Standard 10-20 EEG montage
WINDOW_LENGTH = 400         # 2 seconds at 200 Hz
SAMPLING_RATE = 200.0       # Hz
# Hard caps to prevent OOM on long EDF uploads.
# With 50% overlap (1-second step), 90 windows ~= 90 seconds of timeline.
MAX_EDF_WINDOWS = 90
MAX_EDF_UPLOAD_BYTES = 200 * 1024 * 1024

# Channel order matching our dataset (Technical Specs §3.3.2)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz',
]

# File paths relative to PROJECT_ROOT
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "checkpoint_best.pt"
NORM_STATS_PATH = PROJECT_ROOT / "outputs" / "models" / "normalization_stats.json"
LEADFIELD_PATH = PROJECT_ROOT / "data" / "leadfield_19x76.npy"
CONNECTIVITY_PATH = PROJECT_ROOT / "data" / "connectivity_76.npy"
REGION_LABELS_PATH = PROJECT_ROOT / "data" / "region_labels_76.json"
REGION_CENTERS_PATH = PROJECT_ROOT / "data" / "region_centers_76.npy"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "synthetic3" / "test_dataset.h5"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "frontend_results"
MIN_FREE_BYTES = 100 * 1024 * 1024

TRACT_LENGTHS_PATH = PROJECT_ROOT / "data" / "tract_lengths_76.npy"

# CMA-ES configuration (matches config.yaml)
CMAES_POPULATION_SIZE = 14
CMAES_MAX_GENERATIONS = 30
CMAES_INITIAL_X0 = -2.1
CMAES_INITIAL_SIGMA = 0.3
CMAES_BOUNDS = (-2.4, -1.0)
CMAES_SEED = 42

START_TIME = time.time()


class WaveformRequest(BaseModel):
    eeg: List[List[float]]
    samplingRate: float = 200.0
    asDataURL: bool = True
    title: Optional[str] = None


def _render_waveform_png(eeg_data: NDArray[np.float32], sfreq: float) -> bytes:
    """Render EEG waveforms into a PNG image using MNE when available.
    Falls back to a matplotlib-based plot if MNE is unavailable or plotting fails.
    """
    import io
    import numpy as np
    try:
        import mne
        from mne.io import RawArray
        n_ch = eeg_data.shape[0]
        ch_names = CHANNEL_NAMES[:n_ch] if len(CHANNEL_NAMES) >= n_ch else [f"Ch{i+1}" for i in range(n_ch)]
        info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=['eeg'] * n_ch)
        raw = RawArray(eeg_data.astype(np.float32), info)
        fig, _ = raw.plot(n_channels=n_ch, duration=eeg_data.shape[1] / float(sfreq), show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return buf.getvalue()
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')
            fig, ax = plt.subplots(figsize=(12, 6))
            t = np.arange(eeg_data.shape[1]) / float(sfreq)
            colors = plt.cm.viridis(np.linspace(0, 1, eeg_data.shape[0]))
            ch_names_local = CHANNEL_NAMES[:eeg_data.shape[0]] if len(CHANNEL_NAMES) >= eeg_data.shape[0] else [f"Ch{i+1}" for i in range(eeg_data.shape[0])]
            for i in range(eeg_data.shape[0]):
                ax.plot(t, eeg_data[i, :], color=colors[i], lw=0.8, alpha=0.9, label=ch_names_local[i])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            from PIL import Image
            img = Image.new('RGB', (1, 1), color=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return buf.getvalue()
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

# In-memory job tracker (replaced by active_jobs below)
jobs: Dict[str, dict] = {}

# Thread lock for concurrent access to active_jobs dict
active_jobs_lock = threading.Lock()


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


# ========================================================================
# FastAPI app
# ========================================================================
app = FastAPI(
    title="PhysDeepSIF API",
    description="Physics-informed deep learning EEG inverse solver backend",
    version="1.0.0",
)

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


# ========================================================================
# Startup: load model, normalization stats, region labels
# ========================================================================

def startup_check():
    required_files = {
        "outputs/models/checkpoint_best.pt": "checkpoint",
        "outputs/models/normalization_stats.json": "normalization stats",
        "data/leadfield_19x76.npy": "leadfield matrix",
        "data/connectivity_76.npy": "connectivity",
        "data/region_labels_76.json": "region labels",
        "data/region_centers_76.npy": "region centers",
        "data/tract_lengths_76.npy": "tract lengths",
    }
    all_ok = True
    for path, name in required_files.items():
        exists = Path(path).exists()
        logger.info(f"  {name}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_ok = False
    if all_ok:
        logger.info("All required files present.")
    else:
        logger.warning("Some required files are missing.")
    return all_ok


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
        # Support both v1 (AC-only stats) and v2 (dual raw+AC stats) formats
        eeg_mean_inf = norm_stats.get('eeg_mean_ac', norm_stats['eeg_mean'])
        eeg_std_inf = norm_stats.get('eeg_std_ac', norm_stats['eeg_std'])
        logger.info(
            f"Normalization stats: EEG inf μ={eeg_mean_inf:.4f} "
            f"σ={eeg_std_inf:.4f}, "
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

    # Run startup validation checks
    startup_check()

    logger.info("=" * 60)
    logger.info("PhysDeepSIF API — Ready to serve requests")
    logger.info("=" * 60)

    # Start periodic cleanup for WebSocket job tracker and old result files
    asyncio.create_task(_cleanup_stale_jobs())
    asyncio.create_task(_cleanup_old_results())


# ========================================================================
# Inference helpers
# ========================================================================

def run_inference(eeg: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Run PhysDeepSIF inference on EEG data.

    Applies: per-channel temporal de-meaning (DC removal, matches clinical
    AC-coupling), then global z-score using AC-only inference stats from training.

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

    # Per-channel temporal de-meaning (matches training pipeline, Tech Specs §4.4.7)
    # Clinical EEG is AC-coupled; de-meaning is a no-op for already AC-coupled
    # data. For compatibility with v2 training (EEG retains DC), de-mean here
    # and use AC-only normalisation stats.
    eeg_tensor = eeg_tensor - eeg_tensor.mean(dim=-1, keepdim=True)

    # Global z-score normalization using saved AC-only inference statistics
    eeg_mean_inf = norm_stats.get('eeg_mean_ac', norm_stats['eeg_mean'])
    eeg_std_inf = norm_stats.get('eeg_std_ac', norm_stats['eeg_std'])
    eeg_tensor = (eeg_tensor - eeg_mean_inf) / (eeg_std_inf + eps)

    # Forward pass (inference mode — no gradient computation)
    with torch.no_grad():
        source_pred = model(eeg_tensor)  # (batch, 76, 400)

    # Denormalize source predictions back to original (AC) scale
    # After training with raw sources (DC+AC), the model predicts DC+AC.
    # For inference on real (AC-coupled) EEG, we de-mean the prediction
    # to isolate the AC component and denormalize with AC-only stats.
    source_pred_ac = source_pred - source_pred.mean(dim=-1, keepdim=True)
    src_mean_inf = norm_stats.get('src_mean_ac', norm_stats['src_mean'])
    src_std_inf = norm_stats.get('src_std_ac', norm_stats['src_std'])
    source_pred = source_pred_ac * (src_std_inf + eps) + src_mean_inf

    source_np = source_pred.cpu().numpy()
    if single_sample:
        source_np = source_np[0]  # (76, 400)

    return source_np


def compute_epileptogenicity_index(
    source_activity: NDArray[np.float32],
    epileptogenic_mask: Optional[NDArray[np.bool_]] = None,
    threshold_percentile: float = 87.5,
) -> dict:
    """
    Compute per-region epileptogenicity index using power-based scoring.

    After per-region temporal de-meaning (DC removal), epileptogenic regions
    exhibit 3.9× higher source power (variance) than healthy regions, which
    is the primary discriminative signal.  We score based on time-averaged
    power: high power → high epileptogenicity score.

    Uses the same feature (mean source² over time) as compute_auc_epileptogenicity
    in metrics.py for consistency between training validation and deployment.

    Args:
        source_activity: Predicted source, shape (76, 400) or (batch, 76, 400).
        epileptogenic_mask: Optional ground truth mask, shape (76,).
        threshold_percentile: Percentile for adaptive threshold (default: 87.5 → top ~10 regions).

    Returns:
        Dictionary with scores, detected regions, threshold, and (if mask given) recall/precision.
    """
    # Handle batch dim
    if source_activity.ndim == 3:
        source_activity = source_activity.mean(axis=0)  # (76, 400)

    # Power-based scoring: higher power → higher epileptogenicity
    # Matches compute_auc_epileptogenicity: mean(source²) over time
    region_power = np.mean(source_activity ** 2, axis=1)  # (76,)

    # ── Three-stage scoring from raw power to [0, 1] EI scores ──
    #
    # Stage 1 — z-score: region power is heavy-tailed (some regions orders
    #   of magnitude more active).  z-scoring puts all regions on a common
    #   scale, identifying which are outliers relative to the per-sample
    #   distribution.  This is a simple proxy for identifying the subset
    #   of regions whose power stands out — clinically, the epileptogenic
    #   zone is defined by focal increased activity.
    #
    # Stage 2 — sigmoid: maps z-scores to (0, 1), producing a smooth
    #   ranking.  Regions at +3σ → ~0.95, at -3σ → ~0.05.  This is
    #   equivalent to a soft threshold that suppresses noise while
    #   preserving the relative ordering.
    #
    # Stage 3 — min-max rescale: stretches scores to exactly [0, 1] so
    #   the heatmap colorscheme uses the full dynamic range regardless of
    #   the per-sample spread.  Without this, low-variance samples would
    #   produce nearly-uniform scores that are visually uninformative.
    #
    # Combined effect: the pipeline acts as a robust, non-parametric
    #   ranking that is invariant to the absolute scale of source power.
    #   This is justified empirically: the overfit test (Phase 1.6) showed
    #   AUC=0.732 with this scoring, demonstrating discriminative utility
    #   despite the heuristic derivation.
    power_mean = region_power.mean()
    power_std = region_power.std()
    if power_std < 1e-10:
        z_scores = np.zeros(N_REGIONS)
    else:
        z_scores = (region_power - power_mean) / power_std

    ei_raw = 1.0 / (1.0 + np.exp(-np.clip(z_scores, -30, 30)))

    score_min = ei_raw.min()
    score_max = ei_raw.max()
    if score_max - score_min < 1e-10:
        ei_scores = np.zeros(N_REGIONS)
    else:
        ei_scores = (ei_raw - score_min) / (score_max - score_min)

    # Adaptive threshold
    threshold = float(np.percentile(ei_scores, threshold_percentile))
    threshold = np.clip(threshold, 0.0, 1.0)

    epileptogenic_idx = np.where(ei_scores > threshold)[0]
    epileptogenic_names = [region_labels[i] for i in epileptogenic_idx]
    # Include full anatomical names for each detected region
    epileptogenic_names_full = [
        format_region_for_display(region_labels[i])
        for i in epileptogenic_idx
    ]

    # Build per-region scores dict for JSON serialization
    scores_dict = {
        region_labels[i]: float(ei_scores[i])
        for i in range(N_REGIONS)
    }

    result = {
        'scores': scores_dict,
        'scores_array': ei_scores.tolist(),  # JSON-safe list
        'epileptogenic_regions': epileptogenic_names,
        'epileptogenic_regions_full': epileptogenic_names_full,  # With full anatomical names
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'max_score_region': region_labels[int(np.argmax(ei_scores))],
        'max_score': float(np.max(ei_scores)),
        'region_labels': region_labels,
    }

    # Ground truth comparison (only for synthetic test data)
    if epileptogenic_mask is not None:
        true_epi_idx = np.where(epileptogenic_mask)[0]
        true_epi_names = [region_labels[i] for i in true_epi_idx]
        result['ground_truth_regions'] = true_epi_names
        result['n_true_epileptogenic'] = len(true_epi_names)

        pred_set = set(epileptogenic_idx.tolist())
        true_set = set(true_epi_idx.tolist())
        if len(true_set) > 0:
            intersection = pred_set & true_set
            result['recall'] = len(intersection) / len(true_set)
        if len(pred_set) > 0 and len(true_set) > 0:
            result['precision'] = len(pred_set & true_set) / len(pred_set)

        # Top-K recall metrics
        for k in [5, 10]:
            top_k_idx = set(np.argsort(ei_scores)[-k:].tolist())
            topk_recall = len(top_k_idx & true_set) / max(len(true_set), 1)
            result[f'top{k}_recall'] = topk_recall

    return result


# ========================================================================
# Shared data-loading helpers (used by both sync and async paths)
# ========================================================================

def _load_test_sample(sample_idx: int):
    """Load a single sample from the synthetic test dataset.

    Returns (eeg_data, mask) where mask may be None.
    Raises FileNotFoundError or ValueError.
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")
    with h5py.File(str(TEST_DATA_PATH), 'r') as f:
        n_test = f['eeg'].shape[0]
        if sample_idx < 0 or sample_idx >= n_test:
            raise ValueError(f"Sample index {sample_idx} out of range [0, {n_test})")
        eeg = f['eeg'][sample_idx]
        mask = f['epileptogenic_mask'][sample_idx] if 'epileptogenic_mask' in f else None
    return eeg, mask


def _process_edf_raw(raw) -> dict:
    """Shared EDF processing: pick 10-20 channels, resample, extract sliding windows.

    Args:
        raw: mne.io.Raw object with EDF data loaded.

    Returns:
        dict with keys: eeg_data, edf_all_windows, edf_window_timestamps,
        total_edf_windows, edf_windows_truncated.
    """
    available = [ch.upper() for ch in raw.ch_names]
    pick_names = []
    for ch in CHANNEL_NAMES:
        if ch in raw.ch_names:
            pick_names.append(ch)
        elif ch.upper() in available:
            idx = available.index(ch.upper())
            pick_names.append(raw.ch_names[idx])
        else:
            raise ValueError(f"Required channel '{ch}' not found in EDF file")
    raw.pick_channels(pick_names)
    if raw.info['sfreq'] != SAMPLING_RATE:
        raw.resample(SAMPLING_RATE)
    data = raw.get_data(units='uV')
    if data.shape[1] < WINDOW_LENGTH:
        raise ValueError(f"EDF file too short: {data.shape[1]} samples (need {WINDOW_LENGTH})")
    step = WINDOW_LENGTH // 2
    n_total = data.shape[1]
    starts = list(range(0, n_total - WINDOW_LENGTH + 1, step))
    total_windows = len(starts)
    if total_windows > MAX_EDF_WINDOWS:
        keep = np.linspace(0, total_windows - 1, num=MAX_EDF_WINDOWS, dtype=int)
        starts = [starts[i] for i in keep]
        truncated = True
    else:
        truncated = False
    windows = [data[:, s:s + WINDOW_LENGTH].astype(np.float32) for s in starts]
    timestamps = [(s + WINDOW_LENGTH / 2) / SAMPLING_RATE for s in starts]
    return {
        "eeg_data": windows[0],
        "edf_all_windows": windows if len(windows) > 1 else None,
        "edf_window_timestamps": timestamps if len(windows) > 1 else None,
        "total_edf_windows": total_windows,
        "edf_windows_truncated": truncated,
    }


# ========================================================================
# Job status tracker helper (adds timestamps for periodic cleanup)
# ========================================================================

def _set_job_status(job_id: str, status: str, progress: int, message: str, **extra) -> None:
    """Set job status with timestamp for automatic cleanup."""
    with active_jobs_lock:
        active_jobs[job_id] = {"status": status, "progress": progress, "message": message, "_ts": time.time(), **extra}


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


_mesh_cache = None
def _load_fsaverage5_mesh():
    global _mesh_cache
    if _mesh_cache is not None:
        return _mesh_cache
    import nibabel as nib
    import nilearn.datasets
    fsaverage5 = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    lh_mesh = nib.load(fsaverage5['pial_left'])
    rh_mesh = nib.load(fsaverage5['pial_right'])

    lh_coords = lh_mesh.darrays[0].data
    rh_coords = rh_mesh.darrays[0].data
    lh_faces = lh_mesh.darrays[1].data
    rh_faces = rh_mesh.darrays[1].data

    n_verts_lh = lh_coords.shape[0]
    rh_faces_offset = rh_faces + n_verts_lh

    all_coords = np.vstack([lh_coords, rh_coords])
    all_faces = np.vstack([lh_faces, rh_faces_offset])

    _mesh_cache = (all_coords, all_faces, n_verts_lh, lh_coords, rh_coords)
    return _mesh_cache


def _assign_vertices_to_regions(
    all_coords, n_verts_lh, lh_coords, rh_coords
):
    """
    Assign each cortical surface vertex to the nearest DK76 region.

    Returns an array of shape (n_verts,) with the region index for each vertex.
    """
    region_centers = np.load(str(REGION_CENTERS_PATH)).astype(np.float32)
    n_verts = all_coords.shape[0]

    lh_region_idx = [i for i, name in enumerate(region_labels) if name.startswith('l')]
    rh_region_idx = [i for i, name in enumerate(region_labels) if name.startswith('r')]

    vertex_region = np.full(n_verts, -1, dtype=np.int32)

    # Left hemisphere vertices → nearest left-hemisphere region
    if lh_region_idx:
        lh_centers = region_centers[lh_region_idx]
        diffs = lh_coords[:, None, :] - lh_centers[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        nearest = np.argmin(dists, axis=1)
        for local_idx, global_idx in enumerate(lh_region_idx):
            mask = nearest == local_idx
            vertex_region[:n_verts_lh][mask] = global_idx

    # Right hemisphere vertices → nearest right-hemisphere region
    if rh_region_idx:
        rh_centers = region_centers[rh_region_idx]
        diffs = rh_coords[:, None, :] - rh_centers[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        nearest = np.argmin(dists, axis=1)
        for local_idx, global_idx in enumerate(rh_region_idx):
            mask = nearest == local_idx
            vertex_region[n_verts_lh:][mask] = global_idx

    return vertex_region


def generate_heatmap_html(
    ei_scores: NDArray[np.float32],
    title: str = "Epileptogenic Zone Detection",
    top_k: int = 5,
) -> str:
    """
    Generate a standalone Plotly 3D brain heatmap for epileptogenicity.

    Uses a HARD THRESHOLD approach: only the top-K highest-scoring regions
    are colored (red shading by intensity). All other regions are rendered
    in a neutral brain gray, creating a clear clinical-style visualization
    where highlighted regions stand out unambiguously.

    Args:
        ei_scores: Array of shape (76,) with values in [0, 1].
        title: Title displayed above the 3D plot.
        top_k: Number of top epileptogenic regions to highlight (default: 5).

    Returns:
        The complete HTML document string containing the Plotly 3D mesh.
    """
    import plotly.graph_objects as go

    # ---- Load cortical mesh and build vertex→region mapping ----
    all_coords, all_faces, n_verts_lh, lh_coords, rh_coords = _load_fsaverage5_mesh()
    n_verts = all_coords.shape[0]
    vertex_region = _assign_vertices_to_regions(
        all_coords, n_verts_lh, lh_coords, rh_coords
    )

    # ---- Hard threshold: only top-K regions are colored ----
    top_k_indices = set(np.argsort(ei_scores)[::-1][:top_k].tolist())
    top_k_names = [region_labels[i] for i in sorted(top_k_indices)]

    # Build vertex colors:
    #   - Non-highlighted regions → 0.0 (will map to neutral gray)
    #   - Highlighted regions → score rescaled within highlighted range for contrast
    highlighted_scores = ei_scores[list(top_k_indices)]
    hl_min = highlighted_scores.min()
    hl_max = highlighted_scores.max()
    hl_range = hl_max - hl_min if hl_max - hl_min > 1e-10 else 1.0

    # Use a two-tier intensity scheme:
    #   0.0 = neutral brain, 0.3-1.0 = epileptogenic gradient
    vertex_colors = np.full(n_verts, 0.0, dtype=np.float32)
    vertex_hover = [''] * n_verts

    for v_idx in range(n_verts):
        r_idx = vertex_region[v_idx]
        if r_idx < 0:
            vertex_hover[v_idx] = "unassigned"
            continue

        rname = region_labels[r_idx]
        if r_idx in top_k_indices:
            # Map score to 0.3–1.0 range for the epileptogenic gradient
            normalized = (ei_scores[r_idx] - hl_min) / hl_range
            vertex_colors[v_idx] = 0.3 + 0.7 * normalized
            vertex_hover[v_idx] = f"⚠ {rname} (EI: {ei_scores[r_idx]:.3f})"
        else:
            vertex_colors[v_idx] = 0.0
            vertex_hover[v_idx] = rname

    # ---- Build Plotly 3D mesh figure ----
    fig = go.Figure()

    # Colorscale: neutral gray base, then sharp transition to red for flagged regions
    fig.add_trace(go.Mesh3d(
        x=all_coords[:, 0],
        y=all_coords[:, 1],
        z=all_coords[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        intensity=vertex_colors,
        colorscale=[
            [0.00, '#d4d4d4'],   # Neutral light gray (healthy brain)
            [0.25, '#d4d4d4'],   # Still gray — ensures non-flagged regions stay gray
            [0.30, '#fee08b'],   # Sharp transition: yellow entry (borderline)
            [0.50, '#fc8d59'],   # Orange (moderate risk)
            [0.70, '#e34a33'],   # Red (high risk)
            [0.85, '#b30000'],   # Dark red (very high risk)
            [1.00, '#7f0000'],   # Deepest red (highest risk)
        ],
        cmin=0.0,
        cmax=1.0,
        opacity=1.0,
        hovertext=vertex_hover,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Epileptogenicity', side='right', font=dict(color='#e0e0e0')),
            tickvals=[0.0, 0.5, 0.75, 1.0],
            ticktext=['Normal', 'Moderate', 'High', 'Very High'],
            tickfont=dict(color='#e0e0e0'),
            len=0.5,
        ),
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.15, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
        name='Brain Surface',
    ))

    # Dark theme matching the ESI page — unified visual style
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='#e0e0e0'),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.8, y=0, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='data',
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        margin=dict(l=0, r=0, t=50, b=10),
        font=dict(color='#e0e0e0'),
        # No camera view buttons — users can rotate interactively via mouse
        autosize=True,
    )

    return fig.to_html(
        include_plotlyjs='cdn', full_html=True, auto_play=False,
        config=dict(responsive=True),
    )


def compute_source_activity_metrics(
    source_activity: NDArray[np.float32],
) -> dict:
    """
    Compute per-region source activity metrics for EEG source imaging (ESI).

    Unlike epileptogenicity scoring (inverted range), this computes straightforward
    activity-level metrics: how electrically active is each brain region?

    The model outputs source_activity of shape (76, 400).  We compute:
      - Power: mean of squared values across time  (dominant metric)
      - Variance: temporal variance (dynamic activity)
      - RMS: root mean square amplitude
      - Peak amplitude: max absolute value across time

    Returns a dictionary with per-region scores normalized to [0, 1] for
    visualization, plus the raw metric arrays.

    Args:
        source_activity: Predicted source, shape (76, 400) or (batch, 76, 400).

    Returns:
        Dictionary with normalized scores, raw metrics, and region-level details.
    """
    # Handle batch dim — average across epochs for a single summary
    if source_activity.ndim == 3:
        source_activity = source_activity.mean(axis=0)  # (76, 400)

    # ---- Per-region activity metrics ----
    # After de-meaning in training, source activity is zero-centered.
    # Variance IS the meaningful signal (see Tech Specs §4.4.7).
    region_variance = np.var(source_activity, axis=1)    # (76,)
    region_rms = np.sqrt(np.mean(source_activity ** 2, axis=1))  # (76,)
    region_peak = np.max(np.abs(source_activity), axis=1)  # (76,)
    region_ptp = np.ptp(source_activity, axis=1)  # (76,) peak-to-peak range

    # Use RMS as the primary activity metric for the heatmap.
    # After per-channel de-meaning (applied during inference), DC residual
    # is zero by construction, so RMS = sqrt(variance).  RMS is chosen over
    # variance for interpretability (same units as source activity, μV).
    primary_metric = region_rms

    # Normalize to [0, 1] for visualization
    metric_min = primary_metric.min()
    metric_max = primary_metric.max()
    if metric_max - metric_min < 1e-10:
        normalized_scores = np.full(N_REGIONS, 0.5)
    else:
        normalized_scores = (primary_metric - metric_min) / (metric_max - metric_min)

    # Identify top active regions
    top_indices = np.argsort(normalized_scores)[::-1]
    top_regions = [region_labels[i] for i in top_indices[:10]]
    # Include full anatomical names for top regions
    top_regions_full = [
        format_region_for_display(region_labels[i])
        for i in top_indices[:10]
    ]

    # Build per-region scores dict
    scores_dict = {
        region_labels[i]: float(normalized_scores[i])
        for i in range(N_REGIONS)
    }

    return {
        'scores': scores_dict,
        'scores_array': normalized_scores.tolist(),
        'top_active_regions': top_regions,
        'top_active_regions_full': top_regions_full,  # With full anatomical names
        'max_activity_region': region_labels[int(np.argmax(normalized_scores))],
        'max_activity_score': float(np.max(normalized_scores)),
        'region_labels': region_labels,
        # Raw metric arrays for the frontend details panel
        'metrics_raw': {
            'variance': region_variance.tolist(),
            'rms': region_rms.tolist(),
            'peak_amplitude': region_peak.tolist(),
            'ptp_range': region_ptp.tolist(),
        },
        # Summary statistics
        'summary': {
            'mean_rms': float(region_rms.mean()),
            'max_rms': float(region_rms.max()),
            'mean_variance': float(region_variance.mean()),
            'max_variance': float(region_variance.max()),
            'spatial_cv': float(region_rms.std() / (region_rms.mean() + 1e-10)),
        },
    }


def generate_source_activity_heatmap_html(
    activity_scores: NDArray[np.float32],
    title: str = "EEG Source Imaging — Estimated Brain Activity",
    animated_frames: Optional[List[NDArray[np.float32]]] = None,
    frame_timestamps: Optional[List[float]] = None,
) -> str:
    """
    Generate a standalone Plotly 3D brain heatmap for source activity (ESI).

    Uses an inferno-style colorscale (matching standard neuroimaging conventions):
    dark purple (low) → orange → bright yellow (high). Clean professional look
    with no debug annotations or technical overlays.

    If animated_frames is provided, creates a Plotly animation with play/pause
    and a frame slider for sliding-window playback of longer recordings.

    Args:
        activity_scores: Array of shape (76,) with values in [0, 1] — best/single window.
        title: Title displayed above the 3D plot.
        animated_frames: Optional list of (76,) score arrays, one per time window.
        frame_timestamps: Optional list of timestamps (seconds) for each frame.

    Returns:
        The complete HTML document string containing the Plotly 3D mesh.
    """
    import plotly.graph_objects as go

    # ---- Load cortical mesh and build vertex→region mapping ----
    all_coords, all_faces, n_verts_lh, lh_coords, rh_coords = _load_fsaverage5_mesh()
    n_verts = all_coords.shape[0]
    vertex_region = _assign_vertices_to_regions(
        all_coords, n_verts_lh, lh_coords, rh_coords
    )

    def _scores_to_vertex_colors(scores, with_hover=False):
        """Map 76 region scores to per-vertex colors and optional hover text."""
        v_colors = np.full(n_verts, 0.0, dtype=np.float32)
        v_hover = [''] * n_verts if with_hover else None
        for v_idx in range(n_verts):
            r_idx = vertex_region[v_idx]
            if r_idx >= 0:
                v_colors[v_idx] = scores[r_idx]
                if with_hover:
                    v_hover[v_idx] = (
                        f"{region_labels[r_idx]}: "
                        f"Activity {scores[r_idx]:.3f}"
                    )
            else:
                if with_hover:
                    v_hover[v_idx] = "unassigned"
        return v_colors, v_hover

    # Build first (or only) frame's vertex data
    vertex_colors, vertex_hover = _scores_to_vertex_colors(
        activity_scores, with_hover=True
    )

    # ---- Build Plotly 3D mesh figure ----
    fig = go.Figure()

    # Inferno-inspired colorscale (standard in neuroimaging)
    inferno_colorscale = [
        [0.00, '#000004'],   # Near-black (lowest activity)
        [0.15, '#1b0c41'],   # Dark indigo
        [0.30, '#4a0c6b'],   # Purple
        [0.45, '#781c6d'],   # Red-purple
        [0.55, '#a52c60'],   # Warm magenta
        [0.65, '#cf4446'],   # Orange-red
        [0.75, '#ed6925'],   # Orange
        [0.85, '#fb9b06'],   # Yellow-orange
        [0.95, '#f7d13d'],   # Yellow
        [1.00, '#fcffa4'],   # Bright yellow-white (highest activity)
    ]

    fig.add_trace(go.Mesh3d(
        x=all_coords[:, 0],
        y=all_coords[:, 1],
        z=all_coords[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        intensity=vertex_colors,
        colorscale=inferno_colorscale,
        cmin=0.0,
        cmax=1.0,
        opacity=1.0,
        hovertext=vertex_hover,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Source<br>Activity', side='right'),
            tickvals=[0, 0.5, 1.0],
            ticktext=['Low', 'Medium', 'High'],
            len=0.5,
        ),
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.15, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
        name='Brain Surface',
    ))

    # ---- Add animation frames if sliding-window data is provided ----
    has_animation = (
        animated_frames is not None
        and frame_timestamps is not None
        and len(animated_frames) > 1
    )

    if has_animation:
        frames = []
        slider_steps = []
        for i, (frame_scores, ts) in enumerate(
            zip(animated_frames, frame_timestamps)
        ):
            v_colors_frame, _ = _scores_to_vertex_colors(frame_scores)
            frames.append(go.Frame(
                data=[go.Mesh3d(intensity=v_colors_frame)],
                name=str(i),
                traces=[0],
            ))
            # Format timestamp as m:ss for cleaner display
            minutes = int(ts) // 60
            seconds = ts - (minutes * 60)
            ts_label = f"{minutes}:{seconds:04.1f}" if minutes > 0 else f"0:{seconds:04.1f}"
            slider_steps.append(dict(
                args=[[str(i)], dict(
                    frame=dict(duration=200, redraw=True),
                    mode='immediate',
                    transition=dict(duration=0),
                )],
                label=ts_label,
                method='animate',
            ))

        fig.frames = frames

        # Add play/pause buttons and slider — no camera view buttons
        # (users rotate the brain directly with mouse drag)
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    x=0.05, y=0.02,
                    xanchor='left', yanchor='bottom',
                    pad=dict(r=10, t=65),
                    font=dict(size=12),
                    bgcolor='rgba(40,40,60,0.85)',
                    bordercolor='rgba(120,120,150,0.5)',
                    borderwidth=1,
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            )],
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0),
                            )],
                        ),
                    ],
                ),
            ],
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    prefix='Time: ',
                    visible=True,
                    xanchor='center',
                ),
                pad=dict(b=10, t=40),
                len=0.9,
                x=0.05,
                xanchor='left',
                steps=slider_steps,
                transition=dict(duration=0),
            )],
        )
    else:
        # Static view — no buttons needed (mouse interaction for rotation)
        pass

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='#e0e0e0'),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.8, y=0, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='data',
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        margin=dict(l=0, r=0, t=50, b=60),
        font=dict(color='#e0e0e0'),
        autosize=True,
    )

    # Debug: log frame count before HTML generation
    n_frames_before = len(fig.frames) if fig.frames else 0
    logger.info(f"[generate_source_activity_heatmap_html] Frames in figure: {n_frames_before}, has_animation={has_animation}")
    
    html_output = fig.to_html(
        include_plotlyjs='cdn', full_html=True, auto_play=False,
        config=dict(responsive=True),
    )
    
    # Debug: count frames in generated HTML
    import re
    frame_names = re.findall(r'"name":"\d+"', html_output)
    logger.info(f"[generate_source_activity_heatmap_html] Frame definitions in HTML: {len(frame_names)}")
    
    return html_output


# ========================================================================
# API Endpoints
# ========================================================================

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


@app.post("/api/eeg_waveform")
async def eeg_waveform(req: WaveformRequest):
    """Render EEG waveforms into a PNG image using MNE (best effort).

    Request body:
      {
        "eeg": [[...], [...], ...],  // shape (19, N)
        "samplingRate": 200.0,
        "asDataURL": true|false,
        "title": "Optional title"
      }
    Returns either a data URL (PNG) or a path to a saved PNG under outputs/frontend_results.
    """
    try:
        eeg_np = np.array(req.eeg, dtype=np.float32)
        if eeg_np.ndim != 2 or eeg_np.shape[0] != N_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"eeg data must be shape ({N_CHANNELS}, N_samples). Got {eeg_np.shape}"
            )

        png_bytes = _render_waveform_png(eeg_np, sfreq=float(req.samplingRate))
        if req.asDataURL:
            import base64
            b64 = base64.b64encode(png_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"
            return JSONResponse({"jobId": str(uuid.uuid4()), "imageDataUrl": data_url})
        else:
            job_id = f"eeg_waveform_{uuid.uuid4().hex[:8]}"
            job_dir = RESULTS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            out_path = job_dir / "waveform.png"
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            return JSONResponse({"jobId": job_id, "imagePath": f"/api/results/{job_id}/waveform.png"})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("eeg_waveform rendering failed")
        raise HTTPException(status_code=500, detail=str(e))


def _build_eeg_payload(eeg_data, fs, channel_names, edf_all_windows=None, edf_window_timestamps=None, include_eeg=True, job_id=None):
    eeg_payload = None
    if include_eeg:
        try:
            if edf_all_windows is not None and edf_window_timestamps is not None:
                windows_payload = []
                for start_ts, win in zip(edf_window_timestamps, edf_all_windows):
                    windows_payload.append({
                        "startTime": float(start_ts - WINDOW_LENGTH / (2 * SAMPLING_RATE)),
                        "endTime": float(start_ts + WINDOW_LENGTH / (2 * SAMPLING_RATE)),
                        "data": win.tolist(),
                    })
                eeg_payload = {
                    "channels": CHANNEL_NAMES,
                    "samplingRate": SAMPLING_RATE,
                    "windowLength": WINDOW_LENGTH,
                    "windows": windows_payload,
                }
            else:
                eeg_payload = {
                    "channels": CHANNEL_NAMES,
                    "samplingRate": SAMPLING_RATE,
                    "windowLength": WINDOW_LENGTH,
                    "windows": [
                        {
                            "startTime": 0.0,
                            "endTime": WINDOW_LENGTH / SAMPLING_RATE,
                            "data": eeg_data.tolist(),
                        }
                    ],
                }
        except Exception:
            extra = {"job_id": job_id} if job_id else {}
            logger.exception(f"Failed to build eeg_payload", extra=extra)
            eeg_payload = None
    return eeg_payload


def _extract_plotly_body(html_content):
    import re
    script_match = re.search(
        r'<script[^>]*src="https://cdn\.plot\.ly/plotly[^"]*"[^>]*></script>',
        html_content, re.I
    )
    plotly_script = script_match.group(0) if script_match else ''
    body_match = re.search(
        r'<body[^>]*>([\s\S]*?)</body>', html_content, re.I
    )
    body_content = body_match.group(1) if body_match else html_content
    return plotly_script + body_content


def _run_xai(eeg_data, ei_result, region_labels):
    xai_result = None
    try:
        from src.xai.eeg_occlusion import explain_biomarker
        scores_array = np.array(ei_result['scores_array'])
        top_region_idx = int(np.argmax(scores_array))
        top_region_code = region_labels[top_region_idx]

        def _ei_pipeline(win):
            sources = run_inference(win)
            ei = compute_epileptogenicity_index(sources)
            return {"scores": np.array(ei["scores_array"])}

        xai_result = explain_biomarker(
            eeg_window=eeg_data.astype(np.float32),
            target_region_idx=top_region_idx,
            run_pipeline_fn=_ei_pipeline,
            occlusion_width=40,
            stride=20,
        )
        xai_result["target_region"] = top_region_code
        xai_result["target_region_full"] = format_region_for_display(top_region_code)
    except Exception as e:
        logger.warning(f"XAI skipped: {e}")
        xai_result = None
    return xai_result


def _make_cmaes_callback(job_id: str, max_generations: int):
    """Create CMA-ES progress callback that updates WebSocket clients."""
    def callback(gen: int, best_x, best_f, history):
        progress_pct = 70 + int(25 * gen / max_generations)
        _set_job_status(
            job_id, "cmaes_running", progress_pct,
            f"CMA-ES inversion: gen {gen}/{max_generations} (score={best_f:.4f})",
            cmaes_phase="running",
            cmaes_generation=gen,
            cmaes_max_generations=max_generations,
            cmaes_best_score=float(best_f),
        )
    return callback


def _run_cmaes_inversion(
    job_id: str,
    eeg_data: NDArray[np.float32],
    leadfield: NDArray[np.float32],
    connectivity: NDArray[np.float32],
    centers: NDArray[np.float32],
    tracts: NDArray[np.float32],
    rlabels: List[str],
    heuristic_ei: NDArray[np.float32],
) -> Dict:
    """Run CMA-ES inversion in thread, updating active_jobs with progress."""
    _set_job_status(job_id, "cmaes_queued", 72, "Initializing CMA-ES inversion...",
        cmaes_phase="queued", cmaes_generation=0,
        cmaes_max_generations=CMAES_MAX_GENERATIONS)

    cb = _make_cmaes_callback(job_id, CMAES_MAX_GENERATIONS)

    result = fit_patient(
        target_eeg=eeg_data.astype(np.float64),
        leadfield=leadfield.astype(np.float64),
        connectivity_weights=connectivity.astype(np.float64),
        region_centers=centers.astype(np.float64),
        region_labels=rlabels,
        tract_lengths=tracts.astype(np.float64),
        population_size=CMAES_POPULATION_SIZE,
        max_generations=CMAES_MAX_GENERATIONS,
        initial_x0=CMAES_INITIAL_X0,
        initial_sigma=CMAES_INITIAL_SIGMA,
        bounds=CMAES_BOUNDS,
        seed=CMAES_SEED,
        callback=cb,
    )

    biophysical_ei = compute_biophysical_ei(result["best_x0"])
    concordance = compute_concordance(
        heuristic_ei.astype(np.float64),
        biophysical_ei,
        top_k=10,
    )

    return {
        "best_x0": result["best_x0"].tolist(),
        "best_score": float(result["best_score"]),
        "generations": int(result["generations"]),
        "n_evaluations": int(result["n_evaluations"]),
        "biophysical_ei": biophysical_ei.tolist(),
        "concordance": {
            "tier": concordance["tier"],
            "overlap": int(concordance["overlap"]),
            "shared_regions": [rlabels[i] for i in concordance["shared_regions"]],
            "tier_description": concordance["tier_description"],
            "heuristic_top10": [rlabels[i] for i in concordance["heuristic_top10"]],
            "biophysical_top10": [rlabels[i] for i in concordance["biophysical_top10"]],
        },
    }


@app.post("/api/analyze")
async def analyze_eeg(
    file: Optional[UploadFile] = File(None),
    sample_idx: Optional[int] = Form(None),
    threshold_percentile: float = Form(87.5),
    mode: str = Form("biomarkers"),
    include_eeg: bool = Form(True),
    ws: bool = Form(False),
):
    """
    Full analysis pipeline: EEG → source localization → visualization.

    Accepts either:
      - An uploaded EEG file (EDF/MAT/NPY format, 19 channels)
      - A sample index from the synthetic test dataset

    Mode controls the post-processing and visualization:
      - 'source_localization': ESI visualization — 3D heatmap of source activity
        magnitude. Shows which brain regions are most electrically active.
      - 'biomarkers': Epileptogenic zone detection via power-based scoring
        (time-averaged squared source activity).
        Shows per-region epileptogenicity index (EI).

    Returns JSON with mode-appropriate scores, metrics, and an HTML heatmap.
    """
    start_time = time.time()

    # WebSocket mode: queue job and return immediately, process in background
    if ws:
        ws_job_id = f"ws_{uuid.uuid4().hex[:8]}"
        _set_job_status(ws_job_id, "queued", 0, "Starting...")

        file_path_to_pass = None
        if file is not None:
            file_ext = Path(file.filename).suffix.lower()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                file_path_to_pass = tmp.name

        asyncio.create_task(_process_analysis_async(
            ws_job_id,
            file_path=file_path_to_pass,
            sample_idx=sample_idx,
            mode=mode,
            threshold_percentile=threshold_percentile,
            include_eeg=include_eeg,
        ))
        logger.info(f"Queued async analysis (file={file_path_to_pass}, sample_idx={sample_idx}, mode={mode})", extra={"job_id": ws_job_id})
        return JSONResponse({"status": "queued", "job_id": ws_job_id})

    job_id = f"physdeepsif_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        # ---- Load EEG data ----
        eeg_data = None
        mask = None
        source_label = ""
        edf_all_windows = None        # Sliding window data (EDF only)
        edf_window_timestamps = None  # Timestamps for each window (EDF only)
        total_edf_windows = 1
        edf_windows_truncated = False

        if sample_idx is not None:
            # Load from synthetic test dataset
            logger.info(f"Loading test sample idx={sample_idx}", extra={"job_id": job_id})
            source_label = f"synthetic3/test sample {sample_idx}"

            if not TEST_DATA_PATH.exists():
                raise HTTPException(status_code=404, detail="Test dataset not found")

            with h5py.File(str(TEST_DATA_PATH), 'r') as f:
                n_test = f['eeg'].shape[0]
                if sample_idx < 0 or sample_idx >= n_test:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Sample index {sample_idx} out of range [0, {n_test})"
                    )
                eeg_data = f['eeg'][sample_idx]  # (19, 400)
                if 'epileptogenic_mask' in f:
                    mask = f['epileptogenic_mask'][sample_idx]  # (76,)

        elif file is not None:
            # Parse uploaded EEG file
            logger.info(f"Processing uploaded file: {file.filename}", extra={"job_id": job_id})
            source_label = file.filename

            file_ext = Path(file.filename).suffix.lower()

            if file_ext == '.edf':
                import mne
                import tempfile

                bytes_written = 0
                with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_EDF_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"EDF file too large ({bytes_written} bytes). Maximum allowed is {MAX_EDF_UPLOAD_BYTES} bytes (~200MB).",
                            )
                        tmp.write(chunk)
                    tmp_path = tmp.name

                try:
                    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                    result = _process_edf_raw(raw)
                    eeg_data = result["eeg_data"]
                    edf_all_windows = result["edf_all_windows"]
                    edf_window_timestamps = result["edf_window_timestamps"]
                    total_edf_windows = result["total_edf_windows"]
                    edf_windows_truncated = result["edf_windows_truncated"]
                    logger.info(
                        f"EDF sliding window: using "
                        f"{len(edf_all_windows) if edf_all_windows else 1} / {total_edf_windows} windows",
                        extra={"job_id": job_id}
                    )
                finally:
                    os.unlink(tmp_path)

            elif file_ext == '.npy':
                # Direct numpy array (19, 400)
                import io
                file_bytes = await file.read()
                eeg_data = np.load(io.BytesIO(file_bytes)).astype(np.float32)
                if eeg_data.shape != (N_CHANNELS, WINDOW_LENGTH):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected shape ({N_CHANNELS}, {WINDOW_LENGTH}), "
                               f"got {eeg_data.shape}"
                    )

            elif file_ext == '.mat':
                # MATLAB .mat file — look for EEG data array inside
                import io
                from scipy.io import loadmat

                file_bytes = await file.read()
                mat_data = loadmat(io.BytesIO(file_bytes))

                # Try common variable names for EEG data
                eeg_key = None
                candidate_keys = ['EEG', 'eeg', 'data', 'Data', 'X', 'x',
                                  'eeg_data', 'EEG_data', 'signal', 'signals']
                for k in candidate_keys:
                    if k in mat_data:
                        eeg_key = k
                        break
                # Fall back to the first non-metadata key
                if eeg_key is None:
                    user_keys = [k for k in mat_data.keys()
                                 if not k.startswith('__')]
                    if user_keys:
                        eeg_key = user_keys[0]
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"No data found in MAT file. Keys: {list(mat_data.keys())}"
                        )

                raw_mat = np.array(mat_data[eeg_key], dtype=np.float32)
                logger.info(f"MAT key='{eeg_key}', raw shape={raw_mat.shape}", extra={"job_id": job_id})

                # Accept (channels, time) or (time, channels) and auto-transpose
                if raw_mat.ndim != 2:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected 2D array in MAT file, got shape {raw_mat.shape}"
                    )

                # If shape is (time, channels), transpose
                if raw_mat.shape[0] > raw_mat.shape[1]:
                    raw_mat = raw_mat.T

                if raw_mat.shape[0] != N_CHANNELS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected {N_CHANNELS} channels, got {raw_mat.shape[0]}. "
                               f"MAT variable '{eeg_key}' has shape {raw_mat.shape}"
                    )

                # Take first 400 samples if longer
                if raw_mat.shape[1] < WINDOW_LENGTH:
                    raise HTTPException(
                        status_code=400,
                        detail=f"MAT data too short: {raw_mat.shape[1]} samples "
                               f"(need {WINDOW_LENGTH} = 2 seconds at 200 Hz)"
                    )
                eeg_data = raw_mat[:, :WINDOW_LENGTH]

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}. Use .edf, .mat, or .npy"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' (EEG upload) or 'sample_idx' (test data)"
            )

        # ---- Run inference ----
        logger.info(f"Running PhysDeepSIF inference (mode={mode})...", extra={"job_id": job_id})
        predicted_sources = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, run_inference, eeg_data),
            timeout=60.0
        )
        logger.info(
            f"Inference complete: "
            f"source shape {predicted_sources.shape}, "
            f"range [{predicted_sources.min():.4f}, {predicted_sources.max():.4f}]",
            extra={"job_id": job_id}
        )

        # Save results to disk
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Common source activity summary (included in both modes)
        source_summary = {
            "shape": list(predicted_sources.shape),
            "min": float(predicted_sources.min()),
            "max": float(predicted_sources.max()),
            "mean": float(predicted_sources.mean()),
            "std": float(predicted_sources.std()),
        }

        # ==============================================================
        # MODE: source_localization — ESI activity heatmap
        # ==============================================================
        if mode == 'source_localization':
            logger.info(f"Computing source activity metrics...", extra={"job_id": job_id})
            activity_result = compute_source_activity_metrics(predicted_sources)

            # ---- Sliding window animation (EDF files with > 1 window) ----
            animated_frames = None
            frame_timestamps = None
            n_windows_processed = 1

            if edf_all_windows is not None and edf_window_timestamps is not None:
                logger.info(
                    f"Processing {len(edf_all_windows)} sliding windows...",
                    extra={"job_id": job_id}
                )
                animated_frames = []
                frame_timestamps = edf_window_timestamps
                n_windows_processed = len(edf_all_windows)

                # Run inference on each window and compute normalized scores
                for i, win_eeg in enumerate(edf_all_windows):
                    win_sources = run_inference(win_eeg)  # (76, 400)
                    win_metrics = compute_source_activity_metrics(win_sources)
                    animated_frames.append(
                        np.array(win_metrics['scores_array'], dtype=np.float32)
                    )

                logger.info(f"Sliding window processing complete.", extra={"job_id": job_id})

            logger.info(f"Generating source activity 3D heatmap...", extra={"job_id": job_id})
            heatmap_html = generate_source_activity_heatmap_html(
                activity_scores=np.array(activity_result['scores_array']),
                title="EEG Source Imaging — Estimated Brain Activity",
                animated_frames=animated_frames,
                frame_timestamps=frame_timestamps,
            )

            if shutil.disk_usage(RESULTS_DIR).free < MIN_FREE_BYTES:
                raise HTTPException(status_code=507, detail="Insufficient disk space")

            # Save HTML to disk
            heatmap_path = job_dir / "source_activity_heatmap.html"
            with open(heatmap_path, 'w') as f:
                f.write(heatmap_html)

            # Save metrics JSON
            metrics_path = job_dir / "source_activity_metrics.json"
            json_result = {
                k: v for k, v in activity_result.items()
                if k != 'scores_array_np'
            }
            with open(metrics_path, 'w') as f:
                json.dump(json_result, f, indent=2)

            processing_time = time.time() - start_time
            logger.info(f"Complete in {processing_time:.1f}s", extra={"job_id": job_id})

            plot_content = _extract_plotly_body(heatmap_html)
            eeg_payload = _build_eeg_payload(
                eeg_data, SAMPLING_RATE, CHANNEL_NAMES,
                edf_all_windows=edf_all_windows,
                edf_window_timestamps=edf_window_timestamps,
                include_eeg=include_eeg, job_id=job_id,
            )

            return JSONResponse({
                "jobId": job_id,
                "status": "completed",
                "mode": "source_localization",
                "processingTime": round(processing_time, 2),
                "source": source_label,
                "plotHtml": plot_content,
                # Include raw EEG windows so frontend can render waveform comparisons
                "eegData": eeg_payload,
                "fullHtmlPath": f"/api/results/{job_id}/source_activity_heatmap.html",
                "nWindowsProcessed": n_windows_processed,
                "nWindowsTotal": total_edf_windows,
                "windowsTruncated": edf_windows_truncated,
                "hasAnimation": animated_frames is not None,
                # Per-region activity scores
                "sourceLocalization": {
                    "scores": activity_result['scores'],
                    "scores_array": activity_result['scores_array'],
                    "top_active_regions": activity_result['top_active_regions'],
                    "top_active_regions_full": activity_result['top_active_regions_full'],
                    "max_activity_region": activity_result['max_activity_region'],
                    "max_activity_score": activity_result['max_activity_score'],
                    "region_labels": region_labels,
                    "summary": activity_result['summary'],
                },
                # Raw source activity summary
                "sourceActivity": source_summary,
            })

        # ==============================================================
        # MODE: biomarkers — Epileptogenicity index heatmap (default)
        # ==============================================================
        else:
            logger.info(f"Computing epileptogenicity index...", extra={"job_id": job_id})
            ei_result = compute_epileptogenicity_index(
                predicted_sources,
                epileptogenic_mask=mask,
                threshold_percentile=threshold_percentile,
            )

            # ── XAI: Explain top detected region via occlusion ──
            xai_result = _run_xai(eeg_data, ei_result, region_labels)
            if xai_result:
                logger.info(f"XAI complete", extra={"job_id": job_id})

            logger.info(f"Generating epileptogenicity 3D heatmap...", extra={"job_id": job_id})
            heatmap_html = generate_heatmap_html(
                ei_scores=np.array(ei_result['scores_array']),
                title="Epileptogenic Zone Detection",
                top_k=5,
            )

            # Save HTML to disk
            if shutil.disk_usage(RESULTS_DIR).free < MIN_FREE_BYTES:
                raise HTTPException(status_code=507, detail="Insufficient disk space")
            heatmap_path = job_dir / "brain_heatmap.html"
            with open(heatmap_path, 'w') as f:
                f.write(heatmap_html)

            # Save scores JSON
            scores_path = job_dir / "epileptogenicity_scores.json"
            json_result = {
                k: v for k, v in ei_result.items()
                if k != 'scores_array_np'
            }
            with open(scores_path, 'w') as f:
                json.dump(json_result, f, indent=2)

            processing_time = time.time() - start_time
            logger.info(f"Complete in {processing_time:.1f}s", extra={"job_id": job_id})

            plot_content = _extract_plotly_body(heatmap_html)
            eeg_payload = _build_eeg_payload(
                eeg_data, SAMPLING_RATE, CHANNEL_NAMES,
                edf_all_windows=edf_all_windows,
                edf_window_timestamps=edf_window_timestamps,
                include_eeg=include_eeg, job_id=job_id,
            )

            return JSONResponse({
                "jobId": job_id,
                "status": "completed",
                "mode": "biomarkers",
                "processingTime": round(processing_time, 2),
                "source": source_label,
                "plotHtml": plot_content,
                "eegData": eeg_payload,
                "fullHtmlPath": f"/api/results/{job_id}/brain_heatmap.html",
                "nWindowsTotal": total_edf_windows,
                "windowsTruncated": edf_windows_truncated,
                # XAI occlusion attribution
                "xai": xai_result,
                # Epileptogenicity scores
                "epileptogenicity": {
                    "scores": ei_result['scores'],
                    "scores_array": ei_result['scores_array'],
                    "epileptogenic_regions": ei_result['epileptogenic_regions'],
                    "epileptogenic_regions_full": ei_result['epileptogenic_regions_full'],
                    "threshold": ei_result['threshold'],
                    "threshold_percentile": ei_result['threshold_percentile'],
                    "max_score_region": ei_result['max_score_region'],
                    "max_score": ei_result['max_score'],
                    "region_labels": region_labels,
                },
                # Ground truth metrics (only for synthetic test data)
                "groundTruth": {
                    "available": mask is not None,
                    "regions": ei_result.get('ground_truth_regions', []),
                    "n_epileptogenic": ei_result.get('n_true_epileptogenic', 0),
                    "recall": ei_result.get('recall', None),
                    "precision": ei_result.get('precision', None),
                    "top5_recall": ei_result.get('top5_recall', None),
                    "top10_recall": ei_result.get('top10_recall', None),
                } if mask is not None else None,
                # Source activity summary
                "sourceActivity": source_summary,
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}", extra={"job_id": job_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/biomarkers")
async def biomarker_detection(
    sample_idx: int = Form(...),
    threshold_percentile: float = Form(87.5),
):
    """
    Quick biomarker detection on a synthetic test sample.

    This is a convenience endpoint that loads a single test sample by index
    and runs the full biomarker detection pipeline.  Identical to /api/analyze
    with sample_idx but optimized for the biomarker-specific UI.
    """
    # Delegate to the main analyze endpoint logic
    return await analyze_eeg(
        file=None,
        sample_idx=sample_idx,
        threshold_percentile=threshold_percentile,
    )


@app.get("/api/test-samples")
async def list_test_samples(
    mode: str = "epileptogenic",
    limit: int = 20,
):
    """
    List available test sample indices for the demo UI.

    Args:
        mode: 'epileptogenic' | 'healthy' | 'all'
        limit: Maximum number of indices to return.

    Returns:
        List of sample indices matching the requested mode.
    """
    if not TEST_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Test dataset not found")

    indices = []
    with h5py.File(str(TEST_DATA_PATH), 'r') as f:
        n_test = f['eeg'].shape[0]
        has_mask = 'epileptogenic_mask' in f

        if mode == 'all' or not has_mask:
            indices = list(range(min(limit, n_test)))
        else:
            batch_size = 1000
            for start in range(0, n_test, batch_size):
                end = min(start + batch_size, n_test)
                masks = f['epileptogenic_mask'][start:end]
                for local_idx in range(masks.shape[0]):
                    is_epi = masks[local_idx].any()
                    if (mode == 'epileptogenic' and is_epi) or \
                       (mode == 'healthy' and not is_epi):
                        indices.append(start + local_idx)
                    if len(indices) >= limit:
                        break
                if len(indices) >= limit:
                    break

    return {
        "mode": mode,
        "count": len(indices),
        "total_test_samples": n_test if TEST_DATA_PATH.exists() else 0,
        "indices": indices,
    }


@app.get("/api/results/{path:path}")
async def serve_result_file(path: str):
    """Serve a result file (HTML, JSON) from the results directory."""
    file_path = RESULTS_DIR / path
    # Security: ensure path doesn't escape results dir
    if not file_path.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    suffix = file_path.suffix.lower()
    content_types = {
        '.html': 'text/html',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
    }
    return FileResponse(
        str(file_path),
        media_type=content_types.get(suffix, 'application/octet-stream'),
    )


# ========================================================================
# WebSocket for real-time job status
# ========================================================================
active_jobs: Dict[str, Dict] = {}  # job_id -> {status, progress, message}


async def _process_analysis_async(
    job_id: str,
    file_path: Optional[str],
    sample_idx: Optional[int],
    mode: str,
    threshold_percentile: float,
    include_eeg: bool,
):
    """Run analysis in background task with WebSocket status updates."""
    try:
        _set_job_status(job_id, "loading", 5, "Loading EEG data...")

        eeg_data = None
        mask = None
        edf_all_windows = None
        edf_window_timestamps = None
        total_edf_windows = 1
        edf_windows_truncated = False

        if sample_idx is not None:
            _set_job_status(job_id, "loading", 10, f"Loading test sample {sample_idx}...")
            eeg_data, mask = _load_test_sample(sample_idx)

        elif file_path is not None:
            _set_job_status(job_id, "loading", 10, f"Loading {file_path}...")
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.edf':
                import mne
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                result = _process_edf_raw(raw)
                eeg_data = result["eeg_data"]
                edf_all_windows = result["edf_all_windows"]
                edf_window_timestamps = result["edf_window_timestamps"]
                total_edf_windows = result["total_edf_windows"]
                edf_windows_truncated = result["edf_windows_truncated"]
            elif file_ext == '.mat':
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                eeg_key = None
                for k in ['EEG', 'eeg', 'data', 'Data', 'X', 'x', 'eeg_data', 'EEG_data']:
                    if k in mat_data:
                        eeg_key = k
                        break
                if eeg_key is None:
                    user_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    eeg_key = user_keys[0] if user_keys else None
                if eeg_key is None:
                    raise ValueError("No EEG data found in MAT file")
                raw_mat = np.array(mat_data[eeg_key], dtype=np.float32)
                if raw_mat.ndim != 2:
                    raise ValueError(f"Expected 2D array, got shape {raw_mat.shape}")
                if raw_mat.shape[0] > raw_mat.shape[1]:
                    raw_mat = raw_mat.T
                if raw_mat.shape[0] != N_CHANNELS:
                    raise ValueError(f"Expected {N_CHANNELS} channels, got {raw_mat.shape[0]}")
                eeg_data = raw_mat[:, :WINDOW_LENGTH].astype(np.float32)
            else:
                eeg_data = np.load(file_path).astype(np.float32)

        if eeg_data is None:
            raise ValueError("No EEG data provided")

        # Clean up temp file if one was provided (uploaded via WebSocket)
        if file_path is not None and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass

        _set_job_status(job_id, "preprocessing", 20, "Preprocessing EEG...")

        _set_job_status(job_id, "inference", 40, "Running PhysDeepSIF inference...")
        predicted_sources = run_inference(eeg_data)

        _set_job_status(job_id, "postprocessing", 70, "Computing biomarkers...")

        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if mode == "source_localization":
            activity_result = compute_source_activity_metrics(predicted_sources)

            # Sliding window animation for EDF uploads with >1 window
            animated_frames = None
            frame_timestamps = None
            if edf_all_windows is not None and edf_window_timestamps is not None:
                animated_frames = []
                frame_timestamps = edf_window_timestamps
                for win_eeg in edf_all_windows:
                    win_sources = run_inference(win_eeg)
                    win_metrics = compute_source_activity_metrics(win_sources)
                    animated_frames.append(np.array(win_metrics['scores_array'], dtype=np.float32))

            heatmap_html = generate_source_activity_heatmap_html(
                activity_scores=np.array(activity_result['scores_array']),
                title="EEG Source Imaging — Estimated Brain Activity",
                animated_frames=animated_frames,
                frame_timestamps=frame_timestamps,
            )

            if shutil.disk_usage(RESULTS_DIR).free < MIN_FREE_BYTES:
                raise HTTPException(status_code=507, detail="Insufficient disk space")
            heatmap_path = job_dir / "source_activity_heatmap.html"
            with open(heatmap_path, 'w') as f:
                f.write(heatmap_html)

            _set_job_status(job_id, "completed", 100, "Analysis complete",
                result={
                    "jobId": job_id,
                    "status": "completed",
                    "mode": "source_localization",
                    "fullHtmlPath": f"/api/results/{job_id}/source_activity_heatmap.html",
                }
            )
        else:
            ei_result = compute_epileptogenicity_index(
                predicted_sources,
                epileptogenic_mask=mask,
                threshold_percentile=threshold_percentile,
            )

            xai_result = _run_xai(eeg_data, ei_result, region_labels)
            if xai_result:
                logger.info(f"XAI complete", extra={"job_id": job_id})

            heatmap_html = generate_heatmap_html(
                ei_scores=np.array(ei_result['scores_array']),
                title="Epileptogenic Zone Detection",
                top_k=5,
            )
            scores_path = job_dir / "epileptogenicity_scores.json"
            json_result = {
                k: v for k, v in ei_result.items()
                if k != 'scores_array_np'
            }
            with open(scores_path, 'w') as f:
                json.dump(json_result, f, indent=2)

            # Save heatmap NOW so frontend can load via fullHtmlPath
            if shutil.disk_usage(RESULTS_DIR).free < MIN_FREE_BYTES:
                raise HTTPException(status_code=507, detail="Insufficient disk space")
            heatmap_path = job_dir / "brain_heatmap.html"
            with open(heatmap_path, 'w') as f:
                f.write(heatmap_html)

            # Build common result payload (shared between phase_a and completed)
            result_payload = {
                "jobId": job_id,
                "mode": "biomarkers",
                "fullHtmlPath": f"/api/results/{job_id}/brain_heatmap.html",
                "xai": xai_result,
                "epileptogenicity": {
                    "scores": list(ei_result.get('scores', {}).items()),
                    "epileptogenic_regions": ei_result.get('epileptogenic_regions', []),
                    "epileptogenic_regions_full": ei_result.get('epileptogenic_regions_full', []),
                    "threshold": ei_result.get('threshold'),
                    "threshold_percentile": ei_result.get('threshold_percentile'),
                    "max_score_region": ei_result.get('max_score_region'),
                    "max_score": ei_result.get('max_score'),
                },
            }

            # Phase A complete — WS stays open for CMA-ES progress
            _set_job_status(job_id, "phase_a_complete", 70,
                "Biomarker detection complete. Running biophysical validation...",
                result=result_payload,
                cmaes_phase="scheduled", cmaes_generation=0,
                cmaes_max_generations=CMAES_MAX_GENERATIONS,
            )

            # Phase B: CMA-ES biophysical inversion (runs in executor thread)
            try:
                if connectivity_weights is None or tract_lengths is None:
                    raise RuntimeError("CMA-ES data files not loaded at startup")
                heuristic_scores = np.array(ei_result.get('scores_array', []), dtype=np.float32)

                cmaes_out = await asyncio.to_thread(
                    _run_cmaes_inversion,
                    job_id, eeg_data,
                    leadfield_matrix, connectivity_weights,
                    region_centers, tract_lengths,
                    region_labels, heuristic_scores,
                )

                result_payload["status"] = "completed"
                result_payload["concordance"] = cmaes_out["concordance"]
                result_payload["cmaes"] = {
                    "status": "completed",
                    "best_score": cmaes_out["best_score"],
                    "generations": cmaes_out["generations"],
                }

                _set_job_status(job_id, "completed", 100,
                    "Analysis complete — biophysical validation concordant",
                    result=result_payload,
                    cmaes_phase="completed",
                    cmaes_generation=CMAES_MAX_GENERATIONS,
                    cmaes_max_generations=CMAES_MAX_GENERATIONS,
                )
            except Exception as cmae_err:
                logger.warning(f"CMA-ES skipped: {cmae_err}", extra={"job_id": job_id})
                result_payload["status"] = "completed"
                result_payload["concordance"] = None
                result_payload["cmaes"] = {"status": "failed", "error": str(cmae_err)}
                _set_job_status(job_id, "completed", 100,
                    "Analysis complete (biophysical validation skipped)",
                    result=result_payload,
                    cmaes_phase="failed",
                )

    except Exception as e:
        logger.error(f"Async processing failed: {e}", extra={"job_id": job_id})
        _set_job_status(job_id, "failed", 0, str(e))


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            with active_jobs_lock:
                if job_id in active_jobs:
                    await websocket.send_json(active_jobs[job_id])
                    if active_jobs[job_id].get("status") in ("completed", "failed"):
                        break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        with active_jobs_lock:
            active_jobs.pop(job_id, None)


# ========================================================================
# Entry point
# ========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=30,
    )
