from cmaes import CMA
import numpy as np
from numpy.typing import NDArray
from .objective_function import objective


def fit_patient(
    patient_eeg_psd: NDArray,
    leadfield: NDArray,
    connectivity: NDArray,
    region_centers: NDArray,
    region_labels: list,
    tract_lengths: NDArray,
    config: dict,
    model: object,
    norm_stats: dict,
    device: str = "cpu",
    w_source: float = 0.4,
    w_eeg: float = 0.4,
    w_reg: float = 0.2,
) -> tuple[NDArray, list[float], CMA]:
    """CMA-ES optimization for patient-specific epileptogenicity mapping.
    
    Returns:
        best_x0: 76-dim array of fitted x0 parameters
        convergence_history: list of min objective scores per generation
        cma: CMA optimizer instance for further inspection
    """
    param_cfg = config.get("parameter_inversion", {})
    x0_init = np.full(76, param_cfg.get("initial_x0", -2.1))
    sigma0 = param_cfg.get("initial_sigma", 0.3)
    bounds_lo = param_cfg.get("bounds", [-2.4, -1.0])[0]
    bounds_hi = param_cfg.get("bounds", [-2.4, -1.0])[1]
    bounds = (np.full(76, bounds_lo), np.full(76, bounds_hi))
    max_generations = param_cfg.get("max_generations", 50)
    seed = param_cfg.get("seed", 42)
    
    cma = CMA(mean=x0_init, sigma=sigma0, bounds=bounds, seed=seed)
    convergence_history = []
    best_x = None
    best_score = float('inf')
    
    for gen in range(max_generations):
        solutions = []
        while len(solutions) < cma.population_size:
            x = cma.ask()
            score = objective(
                x,
                patient_eeg_psd,
                leadfield,
                connectivity,
                region_centers,
                region_labels,
                tract_lengths,
                config,
                model,
                norm_stats,
                device,
                w_source,
                w_eeg,
                w_reg,
            )
            solutions.append((x, score))
            
            # Track best
            if score < best_score:
                best_score = score
                best_x = x.copy()
        
        cma.tell(solutions)
        convergence_history.append(min(score for _, score in solutions))
        
        if cma.should_stop():
            break
    
    return best_x, convergence_history, cma
