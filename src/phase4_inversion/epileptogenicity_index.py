import numpy as np
from numpy.typing import NDArray


def compute_biophysical_ei(
    x0: NDArray[np.float64],
    temperature: float = 0.15,
    healthy_baseline: float = -2.2,
) -> NDArray[np.float64]:
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    ei = 1.0 / (1.0 + np.exp(-(x0 - healthy_baseline) / temperature))
    return ei
