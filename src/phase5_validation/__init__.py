"""
Package: src.phase5_validation
Phase: 5 — Validation and Benchmarking
Purpose: Validates PhysDeepSIF predictions against ground truth and compares
         against classical EEG source localization baselines (eLORETA, MNE,
         dSPM, LCMV).

Modules:
  synthetic_metrics.py     — DLE, SD, AUC, temporal correlation wrappers
  classical_baselines.py   — MNE-based classical inverse methods

See AGENTS.md Phase 5 section and docs/ for metrics target thresholds.
"""

from .synthetic_metrics import (
    compute_all_metrics,
    compute_auc,
    compute_dle,
    compute_sd,
    compute_temporal_correlation,
)
from .classical_baselines import (
    compute_all_baselines,
    compute_dspm,
    compute_eloreta,
    compute_lcmv,
    compute_mne,
)

__all__ = [
    "compute_dle",
    "compute_sd",
    "compute_auc",
    "compute_temporal_correlation",
    "compute_all_metrics",
    "compute_eloreta",
    "compute_mne",
    "compute_dspm",
    "compute_lcmv",
    "compute_all_baselines",
]
