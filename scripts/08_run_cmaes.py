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
    logger.info("PHASE 4 CMA-ES OPTIMIZATION — SKELETON")
    logger.info("=" * 60)
    logger.info("Configuration loaded:")
    logger.info(f"  - Patient index: {args.patient_idx}")
    logger.info(f"  - Initial x0: {inversion_config['initial_x0']}")
    logger.info(f"  - Initial sigma: {inversion_config['initial_sigma']}")
    logger.info(f"  - Bounds: {inversion_config['bounds']}")
    logger.info(f"  - Population size: {inversion_config['population_size']}")
    logger.info(f"  - Max generations: {inversion_config['max_generations']}")
    logger.info(f"  - Objective weights: {inversion_config['objective_weights']}")
    logger.info(f"  - Device: {args.device}")
    logger.info("=" * 60)

    # TODO: Integrate with Phase 4 modules once implemented:
    # from src.phase4_inversion.objective_function import objective
    # from src.phase4_inversion.cmaes_optimizer import fit_patient
    # from src.phase4_inversion.epileptogenicity_index import compute_ei
    
    # Example integration (to be implemented):
    # 1. Load test dataset and get patient EEG + ground truth x0
    # 2. Load PhysDeepSIF model and run inference to get source estimates
    # 3. Load leadfield, connectivity, region_centers
    # 4. Run CMA-ES optimization:
    #    from cmaes import CMA
    #    bounds = np.array([[args.bounds[0]] * 76, [args.bounds[1]] * 76]).T
    #    cma = CMA(mean=np.full(76, initial_x0), sigma=initial_sigma, bounds=bounds)
    #    for gen in range(max_generations):
    #        solutions = [(cma.ask(), objective(cma.ask(), ...)) for _ in range(pop_size)]
    #        cma.tell(solutions)
    # 5. Compute epileptogenicity index from fitted x0
    # 6. Compare with ground truth epileptogenic regions
    
    logger.info("Skeleton OK — awaiting Phase 4 module implementations")
    logger.info("Expected modules:")
    logger.info("  - src/phase4_inversion/objective_function.py")
    logger.info("  - src/phase4_inversion/cmaes_optimizer.py")
    logger.info("  - src/phase4_inversion/epileptogenicity_index.py")
    logger.info("=" * 60)
    
    # Verify CMA-ES is available
    try:
        from cmaes import CMA
        logger.info("✓ cmaes package available")
        
        # Quick API verification
        bounds = np.array([[-2.4, -1.0]] * 5)
        cma = CMA(mean=np.full(5, -2.1), sigma=0.3, bounds=bounds, seed=42)
        _ = [(cma.ask(), 1.0) for _ in range(cma.population_size)]
        logger.info(f"✓ CMA-ES API verified (pop_size={cma.population_size})")
    except ImportError:
        logger.error("✗ cmaes package not found. Install with: pip install cmaes")
        sys.exit(1)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
