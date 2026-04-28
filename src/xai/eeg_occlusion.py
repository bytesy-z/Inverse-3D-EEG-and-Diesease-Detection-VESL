"""
Module: eeg_occlusion.py
Purpose: Occlusion-based XAI for biomarker detection.

For a given EEG window and target epileptogenic region, masks successive
channel-time segments, re-runs the PhysDeepSIF + biomarker pipeline,
and measures the score drop.  Segments that cause the largest drop are
most influential for the detection.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)

# Default occlusion parameters
OCCLUSION_WIDTH_SAMPLES = 40   # 200 ms at 200 Hz
OCCLUSION_STRIDE_SAMPLES = 20  # 100 ms overlap
N_CHANNELS = 19
WINDOW_LENGTH = 400


def explain_biomarker(
    eeg_window: NDArray,                # (19, 400) single window, z-scored
    target_region_idx: int,             # 0-75 DK region index
    run_pipeline_fn: Callable,          # callable(eeg_window) -> dict with "scores"
    occlusion_width: int = OCCLUSION_WIDTH_SAMPLES,
    stride: int = OCCLUSION_STRIDE_SAMPLES,
) -> Dict:
    """
    Occlusion-based attribution for a biomarker detection.

    Args:
        eeg_window: Single EEG window (19, 400), z-scored.
        target_region_idx: Index of the top-1 detected region to explain.
        run_pipeline_fn: Function that takes (19,400) EEG and returns
                        dict with "scores" key -> (76,) array of EI scores.
        occlusion_width: Width of occlusion segment in samples (default 200 ms).
        stride: Step between occlusion segments (default 100 ms).

    Returns:
        dict with:
            channel_importance: (19,) mean attribution per channel
            time_importance: (n_segments,) attribution per time segment
            attribution_map: (19, n_segments) full channel-time attribution
            top_segments: list[dict] top-5 influential segments
            target_region_idx: int
            baseline_score: float  (unoccluded EI score)
    """
    # Baseline: score without occlusion
    baseline_result = run_pipeline_fn(eeg_window)
    baseline_score = float(baseline_result["scores"][target_region_idx])

    n_segments = (WINDOW_LENGTH - occlusion_width) // stride + 1
    attribution_map = np.zeros((N_CHANNELS, n_segments), dtype=np.float32)

    for ch in range(N_CHANNELS):
        for seg_idx in range(n_segments):
            t_start = seg_idx * stride
            t_end = t_start + occlusion_width

            # Create occluded EEG copy
            eeg_occ = eeg_window.copy()
            # Mask: replace segment with 0 (matches per-channel mean after de-meaning)
            eeg_occ[ch, t_start:t_end] = 0.0

            # Re-run pipeline
            occ_result = run_pipeline_fn(eeg_occ)
            occ_score = float(occ_result["scores"][target_region_idx])

            # Attribution = score drop (positive = segment supported detection)
            attribution_map[ch, seg_idx] = baseline_score - occ_score

    # Aggregate
    channel_importance = attribution_map.mean(axis=1)  # (19,)
    time_importance = attribution_map.mean(axis=0)      # (n_segments,)

    # Find top segments
    top_indices = np.argsort(attribution_map.ravel())[-5:][::-1]
    top_segments = []
    for flat_idx in top_indices:
        ch, seg = np.unravel_index(flat_idx, attribution_map.shape)
        t_center = seg * stride + occlusion_width // 2
        top_segments.append({
            "channel_idx": int(ch),
            "start_sample": int(seg * stride),
            "end_sample": int(seg * stride + occlusion_width),
            "start_time_sec": float(seg * stride / 200.0),
            "end_time_sec": float((seg * stride + occlusion_width) / 200.0),
            "importance": float(attribution_map[ch, seg]),
        })

    return {
        "channel_importance": channel_importance.tolist(),
        "time_importance": time_importance.tolist(),
        "attribution_map": attribution_map.tolist(),
        "top_segments": top_segments[:5],
        "target_region_idx": target_region_idx,
        "baseline_score": baseline_score,
    }
