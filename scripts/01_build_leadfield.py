#!/usr/bin/env python3
"""
Script: 01_build_leadfield.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: One-time script to build the source space data files and the
         19×76 leadfield matrix.

This script runs the first two subtasks of Phase 1:
  1. Build source space: Load TVB connectivity, preprocess, and save the
     connectivity matrix, region centers, region labels, and tract lengths.
  2. Build leadfield: Construct the BEM forward model on fsaverage using
     MNE-Python, average by DK parcellation, apply re-referencing, validate,
     and save the 19×76 leadfield matrix.

Both steps only need to be run once — the saved data files are then used
by all subsequent pipeline stages.

Usage:
    python scripts/01_build_leadfield.py
    python scripts/01_build_leadfield.py --config path/to/config.yaml

See docs/02_TECHNICAL_SPECIFICATIONS.md Sections 3.1 and 3.3 for details.
"""

# Standard library imports
import argparse
import logging
import sys
from pathlib import Path

# Third-party imports
import yaml

# Ensure the project root is on the Python path so we can import src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports — these are the modules we're orchestrating
from src.phase1_forward.source_space import build_source_space, load_source_space_data
from src.phase1_forward.leadfield_builder import build_leadfield

# Configure module-level logger
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the entire application.

    Sets up console logging with timestamps and module names so we can
    track the progress of the long-running BEM computation.

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
    Load and validate the project configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Expected config.yaml in project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def main() -> None:
    """
    Main entry point for source space and leadfield construction.

    Orchestrates the following steps:
      1. Parse command-line arguments
      2. Load configuration
      3. Build source space (TVB connectivity data)
      4. Build leadfield matrix (MNE-Python BEM forward model)
    """
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(
        description="Build source space and leadfield matrix for PhysDeepSIF."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: project root)",
    )
    parser.add_argument(
        "--skip-leadfield",
        action="store_true",
        help="Only build source space, skip leadfield (useful for testing)",
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

    # --- Step 1: Build source space ---
    # This loads TVB's default connectivity, preprocesses it, and saves
    # the four foundational data files to the data/ directory
    ss_config = config.get("source_space", {})
    data_dir = str(
        PROJECT_ROOT / Path(ss_config.get("connectivity_file", "data/connectivity_76.npy")).parent
    )

    logger.info("Step 1: Building source space...")
    build_source_space(output_dir=data_dir)

    # --- Step 2: Build leadfield matrix ---
    if not args.skip_leadfield:
        logger.info("Step 2: Building leadfield matrix...")

        # Load the region labels and centers we just saved (needed for alignment).
        # The region centers are used for spatial matching between TVB regions
        # and MNE DK labels, since TVB uses abbreviated anatomical codes rather
        # than standard DK atlas names.
        _, region_centers, region_labels, _ = load_source_space_data(data_dir)

        # Determine leadfield output path from config
        leadfield_file = str(
            PROJECT_ROOT / config.get("forward_model", {}).get(
                "leadfield_file", "data/leadfield_19x76.npy"
            )
        )

        build_leadfield(
            tvb_region_labels=region_labels,
            tvb_region_centers=region_centers,
            output_path=leadfield_file,
        )
    else:
        logger.info("Step 2: Skipped leadfield construction (--skip-leadfield)")

    logger.info("All done! Source space and leadfield files are ready.")


if __name__ == "__main__":
    main()
