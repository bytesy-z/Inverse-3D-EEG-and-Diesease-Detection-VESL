#!/usr/bin/env python3
"""
Script: 02_generate_synthetic_data.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Batch generation of synthetic EEG training, validation, and test
         datasets using TVB Epileptor simulations + leadfield projection.

This script orchestrates the synthetic dataset generation pipeline:
  1. Load previously built source space data and leadfield matrix
  2. For each dataset split (train/val/test):
     a. Run N_simulations TVB Epileptor simulations in parallel
     b. Project source activity through leadfield to get EEG
     c. Add white + colored noise
     d. Segment into 2-second windows
     e. Save to HDF5 with standardized schema

Dataset sizes from config.yaml (defaults from technical specs):
  - Training: 16,000 simulations × 5 windows = 80,000 samples
  - Validation: 2,000 simulations × 5 windows = 10,000 samples
  - Test: 2,000 simulations × 5 windows = 10,000 samples
  - Total: 100,000 samples

Computational cost: ~20,000 TVB simulations × ~1 second each ≈ 5.5 hours
on 1 core. With 8 cores (n_jobs=-1), expect ~45 minutes.

Usage:
    python scripts/02_generate_synthetic_data.py
    python scripts/02_generate_synthetic_data.py --config config.yaml
    python scripts/02_generate_synthetic_data.py --split train --n-sims 100

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 3.4 for details.
"""

# Standard library imports
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

# Third-party imports
import numpy as np
import yaml

# Suppress runtime warnings from TVB, numba, and scipy
# These are expected overflow/invalid value warnings during Epileptor simulation
# and don't affect the validity of discarded samples
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.history')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numba.np.ufunc.gufunc')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.coupling')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.models.epileptor')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.signal')

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from src.phase1_forward.source_space import load_source_space_data
from src.phase1_forward.leadfield_builder import load_leadfield
from src.phase1_forward.synthetic_dataset import generate_dataset, generate_all_splits

# Configure module-level logger
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging with timestamps for tracking long-running generation.

    Args:
        log_level: Logging level string. Default "INFO".
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=[console_handler])


def load_config(config_path: Path) -> dict:
    """
    Load the project configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        dict: Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def main() -> None:
    """
    Main entry point for synthetic dataset generation.

    Orchestrates loading pre-computed data, then generating datasets
    for one or all splits as specified by command-line arguments.
    """
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(
        description="Generate synthetic EEG datasets for PhysDeepSIF training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: project root)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which dataset split to generate (default: all)",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Override number of simulations (useful for testing)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override number of parallel jobs (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # --- Setup ---
    setup_logging(args.log_level)
    config = load_config(Path(args.config))

    # Apply command-line overrides to config
    if args.n_jobs is not None:
        config.setdefault("synthetic_data", {})["n_jobs"] = args.n_jobs

    # --- Load pre-computed data files ---
    # These were built by scripts/01_build_leadfield.py
    ss_config = config.get("source_space", {})
    data_dir = str(
        PROJECT_ROOT / Path(
            ss_config.get("connectivity_file", "data/connectivity_76.npy")
        ).parent
    )

    logger.info("Loading source space data...")
    connectivity, region_centers, region_labels, tract_lengths = (
        load_source_space_data(data_dir)
    )

    fm_config = config.get("forward_model", {})
    leadfield_path = str(
        PROJECT_ROOT / fm_config.get("leadfield_file", "data/leadfield_19x76.npy")
    )

    logger.info("Loading leadfield matrix...")
    leadfield = load_leadfield(leadfield_path)

    # --- Generate datasets ---
    start_time = time.time()

    if args.split == "all":
        # Override simulation counts if --n-sims was specified
        if args.n_sims is not None:
            config.setdefault("synthetic_data", {})
            config["synthetic_data"]["n_simulations_train"] = args.n_sims
            config["synthetic_data"]["n_simulations_val"] = max(1, args.n_sims // 8)
            config["synthetic_data"]["n_simulations_test"] = max(1, args.n_sims // 8)

        saved_paths = generate_all_splits(
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            config=config,
        )

        logger.info("Generated all splits:")
        for split_name, path in saved_paths.items():
            logger.info(f"  {split_name}: {path}")

    else:
        # Generate a single split
        syn_config = config.get("synthetic_data", {})
        output_dir = syn_config.get("output_dir", "data/synthetic/")

        # Determine simulation count
        if args.n_sims is not None:
            n_sims = args.n_sims
        else:
            sim_key = f"n_simulations_{args.split}"
            n_sims = syn_config.get(sim_key, 2000)

        # Determine base seed (different for each split to avoid overlap)
        seed_offsets = {"train": 0, "val": 100000, "test": 200000}
        base_seed = seed_offsets.get(args.split, 0)

        output_path = str(
            PROJECT_ROOT / Path(output_dir) / f"{args.split}_dataset.h5"
        )
        n_jobs = syn_config.get("n_jobs", -1)

        generate_dataset(
            n_simulations=n_sims,
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            output_path=output_path,
            config=config,
            n_jobs=n_jobs,
            base_seed=base_seed,
        )

    elapsed = time.time() - start_time
    logger.info(
        f"Dataset generation complete. "
        f"Total time: {elapsed / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
