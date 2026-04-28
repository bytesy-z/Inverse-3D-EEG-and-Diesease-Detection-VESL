"""
Phase 3 — Source aggregator

Utilities to aggregate windowed/source estimates into per-region summary
metrics used by downstream validation and visualization components.

This module provides `aggregate_sources()` which accepts source estimates
with shape (n_epochs, n_regions, n_time) and returns a dictionary of
aggregated metrics (mean power, variance, peak-to-peak, mean timecourse,
temporal correlation, activation consistency) and region labels.
"""
from typing import Any, Dict, List, Optional
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _load_region_labels(path: str, n_regions: int) -> List[str]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        logger.warning("Could not read region labels from %s", path)
        return [f'region_{i}' for i in range(n_regions)]

    # Accept either a list of names or a dict mapping codes->names
    if isinstance(data, list) and len(data) == n_regions:
        return data
    if isinstance(data, dict):
        # Try to preserve a stable ordering: sort keys if they look like indices
        keys = list(data.keys())
        try:
            # If keys are numeric strings, sort by numeric value
            keys = sorted(keys, key=lambda k: int(k))
        except Exception:
            keys = sorted(keys)
        vals = [data[k] for k in keys]
        if len(vals) == n_regions:
            return vals

    # Fallback: create generic names
    logger.warning("Region labels file has unexpected format; falling back to generic labels")
    return [f'region_{i}' for i in range(n_regions)]


def aggregate_sources(
    source_estimates: np.ndarray,
    region_labels_path: str = 'data/region_labels_76.json',
    top_fraction: float = 0.10,
) -> Dict[str, Any]:
    """
    Aggregate windowed source estimates into per-region summary metrics.

    Args:
        source_estimates: ndarray, shape (n_epochs, n_regions, n_time)
        region_labels_path: path to JSON file with region labels (list or dict)
        top_fraction: fraction for activation consistency (default 0.10)

    Returns:
        dict containing:
          - region_names: list[str] (length n_regions)
          - mean_power: ndarray (n_regions,)
          - variance: ndarray (n_regions,)
          - peak_to_peak: ndarray (n_regions,)
          - mean_timecourse: ndarray (n_regions, n_time)
          - temporal_correlation: ndarray (n_regions,) average epoch correlation to mean
          - activation_consistency: ndarray (n_regions,) fraction epochs top-k
          - n_epochs, n_regions, n_time
    """
    if not isinstance(source_estimates, np.ndarray):
        source_estimates = np.asarray(source_estimates)

    if source_estimates.ndim != 3:
        raise ValueError('source_estimates must have shape (n_epochs, n_regions, n_time)')

    n_epochs, n_regions, n_time = source_estimates.shape

    # Load region labels (best-effort)
    region_names = _load_region_labels(region_labels_path, n_regions)

    # Mean absolute amplitude (power proxy)
    mean_power = np.mean(np.abs(source_estimates), axis=(0, 2))

    # Variance across epochs and time
    variance = np.var(source_estimates, axis=(0, 2))

    # Peak-to-peak across epochs and time
    peak_to_peak = np.max(source_estimates, axis=(0, 2)) - np.min(source_estimates, axis=(0, 2))

    # Mean timecourse per region (average across epochs)
    mean_timecourse = np.mean(source_estimates, axis=0)  # (n_regions, n_time)

    # Temporal correlation: for each region, average Pearson r between epoch timecourse and mean_timecourse
    temporal_correlation = np.zeros(n_regions, dtype=float)
    for r in range(n_regions):
        ref = mean_timecourse[r]
        # If reference is constant, correlation undefined -> set zeros
        ref_std = ref.std()
        if ref_std == 0:
            temporal_correlation[r] = 0.0
            continue
        cors = []
        for e in range(n_epochs):
            x = source_estimates[e, r]
            x_std = x.std()
            if x_std == 0:
                cors.append(0.0)
                continue
            c = np.corrcoef(ref, x)[0, 1]
            if np.isfinite(c):
                cors.append(c)
            else:
                cors.append(0.0)
        temporal_correlation[r] = float(np.mean(cors)) if cors else 0.0

    # Activation consistency: fraction of epochs where region is in top-k for that epoch
    top_k = max(1, int(np.ceil(top_fraction * n_regions)))
    epoch_region_scores = np.mean(np.abs(source_estimates), axis=2)  # (n_epochs, n_regions)
    is_topk = np.zeros_like(epoch_region_scores, dtype=bool)
    for ei in range(n_epochs):
        idx = np.argsort(epoch_region_scores[ei])[-top_k:]
        is_topk[ei, idx] = True
    activation_consistency = is_topk.mean(axis=0)

    out: Dict[str, Any] = {
        'region_names': region_names,
        'mean_power': mean_power,
        'variance': variance,
        'peak_to_peak': peak_to_peak,
        'mean_timecourse': mean_timecourse,
        'temporal_correlation': temporal_correlation,
        'activation_consistency': activation_consistency,
        'n_epochs': int(n_epochs),
        'n_regions': int(n_regions),
        'n_time': int(n_time),
    }

    return out
