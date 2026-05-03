from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


def compute_concordance(
    heuristic_ei: NDArray[np.float64],
    biophysical_ei: NDArray[np.float64],
    top_k: int = 10,
) -> Dict:
    heuristic_ei = np.asarray(heuristic_ei, dtype=np.float64).ravel()
    biophysical_ei = np.asarray(biophysical_ei, dtype=np.float64).ravel()

    if heuristic_ei.size != biophysical_ei.size:
        raise ValueError(
            f"EI vectors must have same length, got "
            f"{heuristic_ei.size} vs {biophysical_ei.size}"
        )

    heuristic_top = np.argsort(heuristic_ei)[-top_k:][::-1].tolist()
    biophysical_top = np.argsort(biophysical_ei)[-top_k:][::-1].tolist()

    heuristic_set = set(heuristic_top)
    biophysical_set = set(biophysical_top)
    shared = heuristic_set & biophysical_set

    n_overlap = len(shared)

    if n_overlap >= 5:
        tier = "HIGH"
        description = "Both methods independently agree: strong evidence"
    elif 2 <= n_overlap <= 4:
        tier = "MODERATE"
        description = "Partial agreement: correlate with clinical findings"
    else:
        tier = "LOW"
        description = "Methods disagree: consider longer recording or stereo-EEG"

    return {
        "tier": tier,
        "overlap": n_overlap,
        "heuristic_top10": heuristic_top,
        "biophysical_top10": biophysical_top,
        "shared_regions": sorted(shared),
        "tier_description": description,
    }
