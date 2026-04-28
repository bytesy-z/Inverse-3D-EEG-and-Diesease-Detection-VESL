#!/usr/bin/env python3
"""
Script: day03_cmaes_test.py
Day: 3 (Tue Apr 29)
Purpose: CMA-ES integration testing on 1 synthetic test patient

Workflow:
  1. Load test dataset (synthetic3/test_batch_check.h5)
  2. Extract 1 patient's EEG + ground truth x0
  3. Load PhysDeepSIF checkpoint + normalization stats
  4. Compute patient EEG PSD
  5. Run CMA-ES with pop=14, gen=20 (quick test)
  6. Compare fitted x0 vs ground truth
  7. Verify: convergence history is monotonic & decreasing
  8. Compute epileptogenicity index
  9. Identify epileptogenic zones

Usage:
    python scripts/day03_cmaes_test.py --patient-idx 0 --test-run
    python scripts/day03_cmaes_test.py --patient-idx 0 --generations 20
    python scripts/day03_cmaes_test.py --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase4_inversion.cmaes_optimizer import fit_patient
from src.phase4_inversion.epileptogenicity_index import (
    compute_ei,
    identify_epileptogenic_zones,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_patient(hdf5_path: str, patient_idx: int = 0) -> dict:
    """Load a single patient's EEG and metadata from test HDF5."""
    logger.info(f"Loading test dataset from {hdf5_path}")
    
    with h5py.File(hdf5_path, "r") as f:
        # List available keys
        keys = list(f.keys())
        logger.info(f"Available datasets: {keys}")
        
        # Get EEG data
        if "eeg" in f:
            eeg_data = f["eeg"][()]  # (n_samples, 19, 400)
            logger.info(f"EEG shape: {eeg_data.shape}")
        else:
            raise ValueError("'eeg' dataset not found in HDF5")
        
        # Get source data (ground truth)
        if "source_activity" in f:
            source_data = f["source_activity"][()]  # (n_samples, 76, 400)
            logger.info(f"Source shape: {source_data.shape}")
        else:
            source_data = None
            logger.warning("'source_activity' dataset not found")
        
        # Get x0 parameters (ground truth excitability)
        if "x0_vector" in f:
            x0_data = f["x0_vector"][()]  # (n_samples, 76)
            logger.info(f"x0 shape: {x0_data.shape}")
        else:
            x0_data = None
            logger.warning("'x0_vector' dataset not found")
        
        # Get epileptogenic mask
        if "epileptogenic_mask" in f:
            epi_mask = f["epileptogenic_mask"][()]  # (n_samples, 76)
            logger.info(f"Epileptogenic mask shape: {epi_mask.shape}")
        else:
            epi_mask = None
        
        # Extract patient at index
        if patient_idx >= len(eeg_data):
            logger.error(f"Patient index {patient_idx} >= {len(eeg_data)}")
            raise ValueError(f"Invalid patient index")
        
        patient = {
            "eeg": eeg_data[patient_idx],  # (19, 400)
            "source": source_data[patient_idx] if source_data is not None else None,  # (76, 400)
            "x0_ground_truth": x0_data[patient_idx] if x0_data is not None else None,  # (76,)
            "epileptogenic_mask": epi_mask[patient_idx] if epi_mask is not None else None,  # (76,)
        }
    
    logger.info(f"✓ Loaded patient {patient_idx}")
    return patient


def compute_eeg_psd(eeg: np.ndarray, sfreq: int = 200, fmin: float = 0.5, fmax: float = 70.0) -> np.ndarray:
    """Compute average PSD across 19 EEG channels using Welch's method."""
    n_channels, n_samples = eeg.shape
    psds = []
    
    for ch_idx in range(n_channels):
        freqs, psd = welch(eeg[ch_idx], sfreq=sfreq, nperseg=min(256, n_samples))
        mask = (freqs >= fmin) & (freqs <= fmax)
        psds.append(psd[mask])
    
    avg_psd = np.mean(psds, axis=0)
    logger.info(f"Computed EEG PSD: shape {avg_psd.shape}, range [{avg_psd.min():.2e}, {avg_psd.max():.2e}]")
    return avg_psd


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load PhysDeepSIF model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    model = build_physdeepsif()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded: {model}")
    return model


def load_normalization_stats(stats_path: str) -> dict:
    """Load normalization statistics from JSON."""
    logger.info(f"Loading normalization stats from {stats_path}")
    
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    # Convert to numpy arrays
    stats = {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in stats.items()
    }
    
    logger.info(f"✓ Loaded stats: keys = {list(stats.keys())}")
    logger.info(f"  eeg_mean shape: {stats.get('eeg_mean', np.array([])).shape}")
    logger.info(f"  eeg_std shape: {stats.get('eeg_std', np.array([])).shape}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Day 3: CMA-ES Integration Test")
    parser.add_argument(
        "--patient-idx",
        type=int,
        default=0,
        help="Index of synthetic test patient (default: 0)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of CMA-ES generations (default: 20 for quick test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("DAY 3: CMA-ES INTEGRATION TEST")
    logger.info("=" * 70)
    logger.info(f"Patient index: {args.patient_idx}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)
    
    # Load configuration
    config = yaml.safe_load(args.config.read_text())
    logger.info("✓ Configuration loaded")
    
    # Load test patient
    test_h5_path = PROJECT_ROOT / "data/synthetic3/test_dataset.h5"
    if not test_h5_path.exists():
        logger.error(f"Test dataset not found at {test_h5_path}")
        logger.info("Available files:")
        for f in (PROJECT_ROOT / "data/synthetic3").glob("*.h5"):
            logger.info(f"  - {f.name}")
        sys.exit(1)
    patient = load_test_patient(str(test_h5_path), args.patient_idx)
    
    # Compute patient EEG PSD
    patient_eeg_psd = compute_eeg_psd(patient["eeg"], sfreq=200)
    
    # Load PhysDeepSIF model
    checkpoint_path = PROJECT_ROOT / "outputs/models/checkpoint_best.pt"
    model = load_checkpoint(str(checkpoint_path), device=args.device)
    
    # Load normalization statistics
    stats_path = PROJECT_ROOT / "outputs/models/normalization_stats.json"
    norm_stats = load_normalization_stats(str(stats_path))
    
    # Load brain connectivity data
    logger.info("Loading brain connectivity matrices...")
    leadfield = np.load(PROJECT_ROOT / "data/leadfield_19x76.npy")
    connectivity = np.load(PROJECT_ROOT / "data/connectivity_76.npy")
    region_centers = np.load(PROJECT_ROOT / "data/region_centers_76.npy")
    tract_lengths = np.load(PROJECT_ROOT / "data/tract_lengths_76.npy")
    
    with open(PROJECT_ROOT / "data/region_labels_76.json", "r") as f:
        region_labels = json.load(f)
    
    logger.info(f"✓ Leadfield shape: {leadfield.shape}")
    logger.info(f"✓ Connectivity shape: {connectivity.shape}")
    logger.info(f"✓ Region centers shape: {region_centers.shape}")
    logger.info(f"✓ Region labels: {len(region_labels)} regions")
    
    # Override max_generations in config
    config["parameter_inversion"]["max_generations"] = args.generations
    
    logger.info("=" * 70)
    logger.info("RUNNING CMA-ES OPTIMIZATION")
    logger.info("=" * 70)
    
    # Run CMA-ES
    try:
        best_x0, convergence_history, cma = fit_patient(
            patient_eeg_psd=patient_eeg_psd,
            leadfield=leadfield,
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            config=config,
            model=model,
            norm_stats=norm_stats,
            device=args.device,
            w_source=config["parameter_inversion"]["objective_weights"]["w_source"],
            w_eeg=config["parameter_inversion"]["objective_weights"]["w_eeg"],
            w_reg=config["parameter_inversion"]["objective_weights"]["w_reg"],
        )
        
        logger.info(f"✓ CMA-ES completed in {len(convergence_history)} generations")
        logger.info(f"  Final objective value: {convergence_history[-1]:.6f}")
        logger.info(f"  Initial objective value: {convergence_history[0]:.6f}")
        logger.info(f"  Improvement: {convergence_history[0] / convergence_history[-1]:.2f}x")
        
    except Exception as e:
        logger.error(f"CMA-ES failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute epileptogenicity index
    logger.info("=" * 70)
    logger.info("EPILEPTOGENICITY INDEX & ZONE IDENTIFICATION")
    logger.info("=" * 70)
    
    ei = compute_ei(best_x0, x0_baseline=-2.2, scale=0.15)
    zones = identify_epileptogenic_zones(ei, region_labels, threshold=0.5, top_k=5)
    
    logger.info(f"Mean EI: {zones['mean_ei']:.4f}")
    logger.info(f"Max EI: {zones['max_ei']:.4f}")
    logger.info(f"Epileptogenic fraction: {zones['epileptogenic_fraction']:.2%}")
    logger.info(f"\nTop-5 most epileptogenic regions:")
    for region, score in zip(zones["top_k_regions"], zones["top_k_scores"]):
        logger.info(f"  {region:20s}: EI = {score:.4f}")
    
    # Validation: compare fitted x0 vs ground truth
    if patient["x0_ground_truth"] is not None:
        logger.info("=" * 70)
        logger.info("VALIDATION: FITTED vs GROUND TRUTH")
        logger.info("=" * 70)
        
        x0_gt = patient["x0_ground_truth"]
        mse = np.mean((best_x0 - x0_gt) ** 2)
        mae = np.mean(np.abs(best_x0 - x0_gt))
        correlation = np.corrcoef(best_x0, x0_gt)[0, 1]
        
        logger.info(f"MSE (fitted vs ground truth): {mse:.6f}")
        logger.info(f"MAE (fitted vs ground truth): {mae:.6f}")
        logger.info(f"Correlation: {correlation:.4f}")
        
        # Show epileptogenic regions from ground truth
        if patient["epileptogenic_mask"] is not None:
            epi_mask = patient["epileptogenic_mask"].astype(bool)
            ei_gt = compute_ei(x0_gt, x0_baseline=-2.2, scale=0.15)
            
            ei_epi = ei[epi_mask]
            ei_healthy = ei[~epi_mask]
            
            logger.info(f"\nEI statistics for epileptogenic regions:")
            logger.info(f"  Mean EI (epileptogenic): {ei_epi.mean():.4f} ± {ei_epi.std():.4f}")
            logger.info(f"  Mean EI (healthy): {ei_healthy.mean():.4f} ± {ei_healthy.std():.4f}")
            logger.info(f"  Separation: {(ei_epi.mean() - ei_healthy.mean()):.4f}")
    
    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✓ Test patient EEG loaded: shape {patient['eeg'].shape}")
    logger.info(f"✓ PhysDeepSIF model loaded and inferred")
    logger.info(f"✓ CMA-ES optimization completed: {len(convergence_history)} generations")
    logger.info(f"✓ Epileptogenicity index computed: {ei.shape[0]} regions")
    logger.info(f"✓ Convergence history has {len(convergence_history)} points")
    
    # Check convergence quality
    diffs = np.diff(convergence_history)
    monotonic_fraction = np.mean(diffs <= 0)
    if monotonic_fraction > 0.8:
        logger.info(f"✓ Convergence quality GOOD: {monotonic_fraction:.1%} monotonic decreasing")
    else:
        logger.warning(f"⚠ Convergence quality POOR: only {monotonic_fraction:.1%} monotonic decreasing")
    
    logger.info("=" * 70)
    logger.info("✓ DAY 3 TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
