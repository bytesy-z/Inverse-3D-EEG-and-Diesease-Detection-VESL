import logging
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from cmaes import CMA

from .objective_function import build_objective

logger = logging.getLogger(__name__)


class ProgressCallback:
    def __init__(self):
        self.history: Dict[str, list] = {"scores": [], "x0s": []}

    def __call__(
        self,
        gen: int,
        best_x: NDArray[np.float64],
        best_f: float,
        history: Dict[str, list],
    ) -> None:
        self.history["scores"].append(best_f)
        self.history["x0s"].append(best_x.copy())


def fit_patient(
    target_eeg: NDArray[np.float64],
    leadfield: NDArray[np.float64],
    connectivity_weights: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    population_size: int = 14,
    max_generations: int = 30,
    initial_x0: float = -2.1,
    initial_sigma: float = 0.3,
    bounds: tuple = (-2.4, -1.0),
    seed: int = 42,
    callback: Optional[Callable] = None,
) -> Dict:
    logger.info("Building objective function...")
    objective = build_objective(
        target_eeg=target_eeg,
        leadfield=leadfield,
        connectivity_weights=connectivity_weights,
        region_centers=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
    )

    n_regions = connectivity_weights.shape[0]
    bounds_array = np.array([[bounds[0], bounds[1]]] * n_regions)
    mean_init = np.full(n_regions, initial_x0)

    logger.info(
        f"Initializing CMA-ES: pop={population_size}, "
        f"max_gen={max_generations}, sigma={initial_sigma}, "
        f"seed={seed}, bounds={bounds}"
    )

    cma = CMA(
        mean=mean_init,
        sigma=initial_sigma,
        bounds=bounds_array,
        seed=seed,
        population_size=population_size,
    )

    best_x: Optional[NDArray] = None
    best_f: float = float("inf")
    history: Dict[str, list] = {"scores": [], "x0s": []}

    for gen in range(max_generations):
        if cma.should_stop():
            logger.info(f"CMA-ES converged at generation {gen + 1}")
            break

        solutions = []
        for _ in range(cma.population_size):
            x = cma.ask()
            try:
                f = objective(x)
                if np.isnan(f) or np.isinf(f):
                    f = 1e10
            except Exception as e:
                logger.debug(f"Evaluation failed: {e}")
                f = 1e10
            solutions.append((x, f))

        cma.tell(solutions)

        for x, f in solutions:
            if f < best_f:
                best_f = f
                best_x = x.copy()

        history["scores"].append(best_f)
        history["x0s"].append(best_x.copy())

        if callback is not None:
            callback(gen, best_x, best_f, history)

        logger.info(
            f"Gen {gen + 1:3d}/{max_generations} | "
            f"best={best_f:.4e} | "
            f"x0 range [{best_x.min():.3f}, {best_x.max():.3f}]"
        )

    n_evals = (gen + 1) * cma.population_size
    logger.info(
        f"CMA-ES completed: {gen + 1} generations, {n_evals} evaluations, "
        f"best score={best_f:.4e}"
    )

    return {
        "best_x0": best_x,
        "best_score": best_f,
        "history": history,
        "generations": gen + 1,
        "n_evaluations": n_evals,
    }
