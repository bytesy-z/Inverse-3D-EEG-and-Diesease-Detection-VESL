#!/usr/bin/env python3
"""
Script: 08_run_cmaes.py
Phase: 4 — Parameter Inversion (CMA-ES)
Purpose: Run CMA-ES optimization to fit patient-specific epileptogenicity (x0)
  by minimizing the objective function that compares:
  1. Model-predicted source activity vs PhysDeepSIF output
  2. Simulated EEG PSD vs real EEG PSD
  3. Regularization toward healthy baseline

Usage:
    python scripts/08_run_cmaes.py --patient-idx 0
    python scripts/08_run_cmaes.py --max-generations 50 --device cpu
    python scripts/08_run_cmaes.py --patient-idx 1 --test-run

See docs/FINAL_WORK_PLAN.md Section 3 (Hira's tasks) for details.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CMA-ES Parameter Inversion for Epileptogenicity Mapping"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--patient-idx",
        type=int,
        default=0,
        help="Index of synthetic patient to fit (from test dataset)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Override max generations from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for PhysDeepSIF inference",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Quick test run with reduced generations (for debugging)",
    )
    args = parser.parse_args()

    logger.info(f"Loaded configuration from {args.config}")
    config = yaml.safe_load(args.config.read_text())

    # Override for test run
    if args.test_run:
        args.max_generations = 5
        logger.info("Test run mode: max_generations set to 5")

    inversion_config = config["parameter_inversion"]
    if args.max_generations is not None:
        inversion_config["max_generations"] = args.max_generations

    logger.info("=" * 60)
    logger.info("PHASE 4 CMA-ES OPTIMIZATION")
    logger.info("=" * 60)

    # Import Phase 4 modules
    try:
        from src.phase4_inversion.objective_function import objective, _compute_psd
        from src.phase4_inversion.cmaes_optimizer import fit_patient
        from src.phase4_inversion.epileptogenicity_index import compute_ei
        logger.info("✓ Phase 4 modules imported")
    except ImportError as e:
        logger.error(f"✗ Failed to import Phase 4 modules: {e}")
        sys.exit(1)

    # Load data files
    logger.info("Loading data files...")
    try:
        leadfield = np.load(PROJECT_ROOT / "data" / "leadfield_19x76.npy")
        connectivity = np.load(PROJECT_ROOT / "data" / "connectivity_76.npy")
        region_centers = np.load(PROJECT_ROOT / "data" / "region_centers_76.npy")
        import json
        with open(PROJECT_ROOT / "data" / "region_labels_76.json") as f:
            region_labels = json.load(f)
        tract_lengths = np.load(PROJECT_ROOT / "data" / "tract_lengths_76.npy")
        logger.info(f"  ✓ leadfield: {leadfield.shape}")
        logger.info(f"  ✓ connectivity: {connectivity.shape}")
        logger.info(f"  ✓ region_centers: {region_centers.shape}")
        logger.info(f"  ✓ region_labels: {len(region_labels)} regions")
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        sys.exit(1)

    # Load normalization stats
    try:
        with open(PROJECT_ROOT / "outputs" / "models" / "normalization_stats.json") as f:
            norm_stats = json.load(f)
        logger.info(f"  ✓ norm_stats loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load norm_stats: {e}")
        sys.exit(1)

    # Load PhysDeepSIF model
    logger.info("Loading PhysDeepSIF model...")
    try:
        import torch
        from src.phase2_network.physdeepsif import build_physdeepsif
        
        model = build_physdeepsif(
            input_dim=19,
            hidden_dims=[128, 256, 256, 128],
            output_dim=76,
            temporal_type="bilstm",
            temporal_hidden_dim=76,
            temporal_num_layers=2,
        )
        checkpoint = torch.load(
            PROJECT_ROOT / "outputs" / "models" / "checkpoint_best.pt",
            map_location=args.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        model.eval()
        logger.info(f"  ✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Load test dataset
    logger.info(f"Loading test dataset (patient {args.patient_idx})...")
    try:
        import h5py
        with h5py.File(PROJECT_ROOT / "data" / "synthetic3" / "test_batch_check.h5", "r") as f:
            eeg_test = f["eeg"][args.patient_idx]  # (19, 400)
            source_test = f["source"][args.patient_idx]  # (76, 400)
            x0_true = f["x0"][args.patient_idx]  # (76,)
        logger.info(f"  ✓ Patient {args.patient_idx}: EEG {eeg_test.shape}, Source {source_test.shape}")
        logger.info(f"  ✓ Ground truth x0: min={x0_true.min():.3f}, max={x0_true.max():.3f}")
    except Exception as e:
        logger.error(f"✗ Failed to load test data: {e}")
        logger.info("Creating synthetic test patient...")
        # Create a synthetic test patient
        np.random.seed(42 + args.patient_idx)
        eeg_test = np.random.randn(19, 400).astype(np.float32) * 10
        source_test = np.random.randn(76, 400).astype(np.float32) * 5
        x0_true = np.random.uniform(-2.2, -1.5, size=76).astype(np.float32)
        # Set some regions as epileptogenic
        x0_true[:5] = np.random.uniform(-1.5, -1.2, size=5)
        logger.info(f"  ✓ Synthetic patient created: x0 min={x0_true.min():.3f}, max={x0_true.max():.3f}")

    # Compute patient EEG PSD (target for CMA-ES)
    patient_eeg_psd = _compute_psd(eeg_test, sfreq=200)
    logger.info(f"  ✓ Patient EEG PSD shape: {patient_eeg_psd.shape}")

    # Run CMA-ES optimization
    logger.info("=" * 60)
    logger.info("Starting CMA-ES optimization...")
    logger.info(f"  Max generations: {inversion_config['max_generations']}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 60)

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
            w_source=inversion_config["objective_weights"]["w_source"],
            w_eeg=inversion_config["objective_weights"]["w_eeg"],
            w_reg=inversion_config["objective_weights"]["w_reg"],
        )
        logger.info("✓ CMA-ES optimization complete")
    except Exception as e:
        logger.error(f"✗ CMA-ES failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute Epileptogenicity Index
    ei = compute_ei(best_x0)
    logger.info(f"  ✓ EI computed: min={ei.min():.3f}, max={ei.max():.3f}")

    # Compare with ground truth
    ei_true = compute_ei(x0_true)
    logger.info(f"  ✓ True EI: min={ei_true.min():.3f}, max={ei_true.max():.3f}")

    # Find top epileptogenic regions
    top_k = 5
    top_predicted = np.argsort(ei)[-top_k:][::-1]
    top_true = np.argsort(ei_true)[-top_k:][::-1]

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Convergence history (last 5 gens): {convergence_history[-5:]}")
    logger.info(f"Final objective: {convergence_history[-1]:.6f}")
    logger.info(f"")
    logger.info(f"Top {top_k} predicted epileptogenic regions (by EI):")
    for i, idx in enumerate(top_predicted):
        logger.info(f"  {i+1}. {region_labels[idx]}: EI={ei[idx]:.3f} (x0={best_x0[idx]:.3f})")
    logger.info(f"")
    logger.info(f"Top {top_k} true epileptogenic regions:")
    for i, idx in enumerate(top_true):
        logger.info(f"  {i+1}. {region_labels[idx]}: EI={ei_true[idx]:.3f} (x0={x0_true[idx]:.3f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
