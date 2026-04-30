#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase4_inversion.cmaes_optimizer import fit_patient
from src.phase4_inversion.epileptogenicity_index import compute_biophysical_ei
from src.phase4_inversion.concordance import compute_concordance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
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
        help="Device for PhysDeepSIF inference (not used in Phase 4)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test: population=4, generations=2",
    )
    args = parser.parse_args()

    logger.info(f"Loaded configuration from {args.config}")
    config = yaml.safe_load(args.config.read_text())

    inversion_config = config["parameter_inversion"]
    if args.max_generations is not None:
        inversion_config["max_generations"] = args.max_generations

    pop_size = inversion_config["population_size"]
    max_gen = inversion_config["max_generations"]

    if args.quick_test:
        pop_size = 4
        max_gen = 2
        logger.info("Quick test mode: population=4, generations=2")

    logger.info("=" * 60)
    logger.info("PHASE 4 CMA-ES OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"  - Patient index: {args.patient_idx}")
    logger.info(f"  - Population size: {pop_size}")
    logger.info(f"  - Max generations: {max_gen}")
    logger.info(f"  - Initial x0: {inversion_config['initial_x0']}")
    logger.info(f"  - Initial sigma: {inversion_config['initial_sigma']}")
    logger.info(f"  - Bounds: {inversion_config['bounds']}")
    logger.info("=" * 60)

    logger.info("Loading data files...")
    leadfield = np.load(PROJECT_ROOT / "data/leadfield_19x76.npy")
    connectivity_weights = np.load(PROJECT_ROOT / "data/connectivity_76.npy")
    region_centers = np.load(PROJECT_ROOT / "data/region_centers_76.npy")
    tract_lengths = np.load(PROJECT_ROOT / "data/tract_lengths_76.npy")

    import json
    with open(PROJECT_ROOT / "data/region_labels_76.json") as f:
        region_labels = json.load(f)

    test_dataset_path = PROJECT_ROOT / "data/synthetic3/test_dataset.h5"
    logger.info(f"Loading test dataset from {test_dataset_path}")
    with h5py.File(test_dataset_path, "r") as f:
        target_eeg = f["eeg"][args.patient_idx]
        true_x0 = f["x0_vector"][args.patient_idx]
        true_mask = f["epileptogenic_mask"][args.patient_idx]

    logger.info(
        f"Patient {args.patient_idx}: "
        f"{int(true_mask.sum())} epileptogenic regions, "
        f"true x0 range [{true_x0.min():.3f}, {true_x0.max():.3f}]"
    )

    logger.info("Running CMA-ES optimization...")
    result = fit_patient(
        target_eeg=target_eeg,
        leadfield=leadfield,
        connectivity_weights=connectivity_weights,
        region_centers=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
        population_size=pop_size,
        max_generations=max_gen,
        initial_x0=inversion_config["initial_x0"],
        initial_sigma=inversion_config["initial_sigma"],
        bounds=inversion_config["bounds"],
        seed=42,
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Best score: {result['best_score']:.4e}")
    logger.info(f"  Generations: {result['generations']}")
    logger.info(f"  Evaluations: {result['n_evaluations']}")
    logger.info(
        f"  Fitted x0 range: [{result['best_x0'].min():.3f}, "
        f"{result['best_x0'].max():.3f}]"
    )

    biophysical_ei = compute_biophysical_ei(result["best_x0"])
    logger.info(
        f"  Biophysical EI range: [{biophysical_ei.min():.4f}, "
        f"{biophysical_ei.max():.4f}]"
    )

    n_true_epi = int(true_mask.sum())
    fitted_top10 = np.argsort(biophysical_ei)[-10:][::-1]
    true_epi_indices = np.where(true_mask)[0]
    overlap = len(set(fitted_top10.tolist()) & set(true_epi_indices.tolist()))
    logger.info(f"  True epileptogenic regions: {n_true_epi}")
    logger.info(f"  Top-10 overlap with ground truth: {overlap}/10")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
