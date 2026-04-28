"""
Phase 3 — Inference engine

Provides a high-level `run_patient_inference` function that runs the
full patient inference pipeline: EDF preprocessing → PhysDeepSIF inference
→ window-level source estimates → simple per-region aggregations.

This file implements the Day 2 task from the work plan: a reusable
inference entrypoint for patient EDFs.
"""
from typing import Any, Dict, Optional
import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_numpy_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return x.cpu().numpy()
    return np.asarray(x)


def _normalize_eeg_windows(eeg_windows: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
    """
    Apply per-channel de-meaning and global z-score normalization.

    eeg_windows: shape (n_epochs, n_channels, n_time)
    norm_stats: dict with optional keys 'eeg_mean' and 'eeg_std' (shapes: n_channels)
    """
    # Per-channel temporal de-meaning (remove DC per epoch)
    eeg = eeg_windows.copy()
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)

    if norm_stats and 'eeg_mean' in norm_stats and 'eeg_std' in norm_stats:
        mean = np.asarray(norm_stats['eeg_mean']).reshape(1, -1, 1)
        std = np.asarray(norm_stats['eeg_std']).reshape(1, -1, 1)
        eeg = (eeg - mean) / (std + 1e-8)
    else:
        # fallback: z-score per-channel using aggregate across epochs/time
        ch_mean = eeg.mean(axis=(0, 2), keepdims=True).reshape(1, -1, 1)
        ch_std = eeg.std(axis=(0, 2), keepdims=True).reshape(1, -1, 1)
        eeg = (eeg - ch_mean) / (ch_std + 1e-8)

    return eeg.astype(np.float32)


def run_patient_inference(
    edf_path: str,
    model: torch.nn.Module,
    norm_stats: Optional[Dict] = None,
    device: str = 'cpu',
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Run full patient inference.

    Steps:
      1. Preprocess EDF via `src.phase3_inference.nmt_preprocessor.NMTPreprocessor`
      2. Produce windowed EEG: (n_epochs, n_channels, n_time)
      3. Ensure de-meaning + z-score normalization (training stats)
      4. Run `model` on windows in batches to get source estimates
      5. Aggregate per-region metrics across epochs and time

    Returns a dict with keys:
      - source_estimates: ndarray (n_epochs, n_regions, n_time)
      - mean_source_power: ndarray (n_regions,)
      - variance: ndarray (n_regions,)
      - peak_to_peak: ndarray (n_regions,)
      - activation_consistency: ndarray (n_regions,)
      - preprocessed_eeg: ndarray (n_epochs, n_channels, n_time)
      - n_epochs: int
      - epoch_times: list[float]
      - channel_names: list[str]
      - edf_path: str
    """
    # Lazy import of NMTPreprocessor to avoid hard dependency during early development
    try:
        from .nmt_preprocessor import NMTPreprocessor
    except Exception as e:
        logger.debug("Could not import NMTPreprocessor: %s", e)
        raise

    preproc = NMTPreprocessor()

    # Expect the preprocessor to expose a convenience method that returns a dict
    # with keys: 'preprocessed_eeg' or 'eeg_windows', 'channel_names', 'epoch_times'
    if hasattr(preproc, 'preprocess'):
        result = preproc.preprocess(edf_path)
    elif hasattr(preproc, 'run'):
        result = preproc.run(edf_path)
    else:
        # Fall back to explicit sequence: load_edf -> segment_and_normalize
        raw = preproc.load_edf(edf_path)
        result = preproc.segment_and_normalize(raw)

    # Extract EEG windows
    eeg_windows = None
    for key in ('preprocessed_eeg', 'eeg_windows', 'preprocessed', 'windows'):
        if key in result:
            eeg_windows = result[key]
            break

    if eeg_windows is None:
        raise RuntimeError('Preprocessor did not return EEG windows in expected keys')

    eeg_windows = _ensure_numpy_array(eeg_windows)
    if eeg_windows.ndim != 3:
        raise ValueError('Expected preprocessed EEG shape (n_epochs, n_channels, n_time)')

    channel_names = result.get('channel_names') or result.get('channels') or []
    epoch_times = result.get('epoch_times') or []

    # Normalization (ensure same preprocessing as training)
    eeg_windows = _normalize_eeg_windows(eeg_windows, norm_stats)

    n_epochs, _, _ = eeg_windows.shape

    # Prepare model
    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    # Batch inference
    all_preds = []
    with torch.no_grad():
        for start in range(0, n_epochs, batch_size):
            end = min(start + batch_size, n_epochs)
            batch = torch.from_numpy(eeg_windows[start:end]).float().to(model_device)

            # Model may accept (batch, channels, time) => outputs (batch, regions, time)
            preds = model(batch)

            # Support models that return dicts
            if isinstance(preds, dict):
                # common key 'source' or 'sources'
                if 'source' in preds:
                    preds = preds['source']
                elif 'sources' in preds:
                    preds = preds['sources']
                else:
                    # pick first tensor-like value
                    for v in preds.values():
                        preds = v
                        break

            preds_np = _ensure_numpy_array(preds)
            all_preds.append(preds_np)

    source_estimates = np.concatenate(all_preds, axis=0)

    # Aggregations across epochs and time
    # mean power: mean absolute amplitude across epochs and time
    mean_source_power = np.mean(np.abs(source_estimates), axis=(0, 2))
    variance = np.var(source_estimates, axis=(0, 2))
    peak_to_peak = np.max(source_estimates, axis=(0, 2)) - np.min(source_estimates, axis=(0, 2))

    # Activation consistency: fraction of epochs where region is in top-10% for that epoch
    n_regions = source_estimates.shape[1]
    top_k = max(1, math.ceil(0.10 * n_regions))
    epoch_region_scores = np.mean(np.abs(source_estimates), axis=2)  # (n_epochs, n_regions)
    is_topk = np.zeros_like(epoch_region_scores, dtype=bool)
    for ei in range(n_epochs):
        idx = np.argsort(epoch_region_scores[ei])[-top_k:]
        is_topk[ei, idx] = True
    activation_consistency = is_topk.mean(axis=0)

    out = {
        'source_estimates': source_estimates,
        'mean_source_power': mean_source_power,
        'variance': variance,
        'peak_to_peak': peak_to_peak,
        'activation_consistency': activation_consistency,
        'preprocessed_eeg': eeg_windows,
        'n_epochs': int(n_epochs),
        'epoch_times': epoch_times,
        'channel_names': channel_names,
        'edf_path': edf_path,
    }

    return out
