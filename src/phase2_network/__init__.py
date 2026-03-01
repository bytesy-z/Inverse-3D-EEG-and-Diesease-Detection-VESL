"""
Phase 2 Network: PhysDeepSIF Architecture and Training

This module contains the complete PhysDeepSIF neural network architecture
designed for solving the EEG source imaging inverse problem with physics-informed
learning.

Components:
  - physdeepsif.py: Network architecture (SpatialModule + TemporalModule)
  - loss_functions.py: Physics-informed loss function
  - trainer.py: Training loop with data augmentation and monitoring
  - metrics.py: Validation metrics (DLE, SD, AUC, correlation)

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 4 for detailed specifications.
"""

from .loss_functions import PhysicsInformedLoss
from .metrics import (
    compute_auc_epileptogenicity,
    compute_dipole_localization_error,
    compute_spatial_dispersion,
    compute_temporal_correlation,
)
from .physdeepsif import PhysDeepSIF, SpatialModule, TemporalModule, build_physdeepsif
from .trainer import PhysDeepSIFTrainer

__all__ = [
    "PhysDeepSIF",
    "SpatialModule",
    "TemporalModule",
    "build_physdeepsif",
    "PhysicsInformedLoss",
    "PhysDeepSIFTrainer",
    "compute_dipole_localization_error",
    "compute_spatial_dispersion",
    "compute_auc_epileptogenicity",
    "compute_temporal_correlation",
]
