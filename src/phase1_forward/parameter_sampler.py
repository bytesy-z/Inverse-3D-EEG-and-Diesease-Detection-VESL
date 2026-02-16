"""
Module: parameter_sampler.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Generate randomized parameter sets for synthetic EEG dataset creation.

Each synthetic training sample requires a unique combination of Epileptor model
parameters. This module implements the sampling strategy described in the
technical specifications (Section 3.4.1), which controls:
  - How many brain regions are epileptogenic (1–8 regions)
  - Which specific regions are chosen (with optional spatial clustering)
  - The excitability level of each region (x0 parameter)
  - Global coupling strength, noise intensity, and conduction speed

The diversity of these parameter combinations is critical for the network's
ability to generalize — the trained PhysDeepSIF must handle any plausible
spatial configuration of epileptogenic zones.

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 3.4.1 for full specifications.

Key dependencies:
- numpy: Random number generation and array operations
- source_space: Provides connectivity matrix for spatial clustering

Input data format:
- connectivity: (76, 76) float64, preprocessed structural connectivity
- config: dict from config.yaml with sampling range specifications

Output data format:
- Dict containing: x0_vector (76,), epileptogenic_mask (76,), and scalar params
"""

# Standard library imports
import logging
from typing import Any, Dict, Optional

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Sections 3.2.2 and 3.4.1)
# ---------------------------------------------------------------------------
N_REGIONS = 76  # Desikan-Killiany parcellation

# Default sampling ranges (can be overridden by config.yaml)
DEFAULT_X0_EPILEPTOGENIC = (-1.8, -1.2)   # Spontaneous spikes to seizures
DEFAULT_X0_HEALTHY = (-2.2, -2.05)        # Stable background oscillations
DEFAULT_N_EPILEPTOGENIC = (1, 8)           # 1 to 8 epileptogenic regions
DEFAULT_GLOBAL_COUPLING = (0.5, 3.0)      # Coupling strength G
DEFAULT_NOISE_INTENSITY = (1e-4, 5e-3)    # Additive noise D
DEFAULT_CONDUCTION_SPEED = (3.0, 6.0)     # m/s for propagation delays
DEFAULT_IEXT1 = (2.8, 3.4)               # External input 1
DEFAULT_IEXT2 = (0.3, 0.6)               # External input 2
DEFAULT_TAU0 = (2000.0, 4000.0)           # Slow time constant (ms)
DEFAULT_TAU2 = (6.0, 15.0)               # Fast subsystem 2 time constant (ms)
DEFAULT_CLUSTERING_PROB = 0.5             # Probability of spatial clustering

# Configure module-level logger
logger = logging.getLogger(__name__)


def sample_epileptogenic_regions(
    n_regions: int,
    connectivity: NDArray[np.float64],
    n_epileptogenic_range: tuple = DEFAULT_N_EPILEPTOGENIC,
    clustering_probability: float = DEFAULT_CLUSTERING_PROB,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.bool_]:
    """
    Select which brain regions are epileptogenic for one synthetic sample.

    This function determines the spatial pattern of epileptogenicity. Two
    selection strategies are used (50/50 split by default):

    Strategy A (Random): Select k regions uniformly at random from all 76.
        This ensures the network sees diverse, scattered patterns.

    Strategy B (Clustered): Select a seed region at random, then grow the
        cluster by iteratively adding the most-connected neighbor. This
        produces spatially contiguous epileptogenic zones, which are more
        realistic for focal epilepsy (the most common clinical scenario).

    The 50/50 split ensures the network can handle both focal (clustered)
    and multifocal (distributed) epilepsy patterns.

    Args:
        n_regions: Total number of brain regions (should be 76).
        connectivity: Preprocessed structural connectivity matrix used for
            the clustering strategy. Shape (76, 76), dtype float64.
            Higher weights mean stronger anatomical connection.
        n_epileptogenic_range: Tuple (min_k, max_k) for the number of
            epileptogenic regions. Sampled as DiscreteUniform(min_k, max_k).
        clustering_probability: Probability of using the clustered selection
            strategy instead of random. Default 0.5 per specs.
        rng: NumPy random Generator for reproducibility. If None, a new
            default Generator is created.

    Returns:
        NDArray[np.bool_]: Binary mask of shape (76,) where True indicates
            an epileptogenic region.

    References:
        Technical Specs Section 3.4.1 (steps 1-2)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Sample how many regions are epileptogenic
    # k ~ DiscreteUniform(1, 8) — at least 1, at most 8
    k = rng.integers(n_epileptogenic_range[0], n_epileptogenic_range[1] + 1)

    # Step 2: Decide which regions — random or clustered
    use_clustering = rng.random() < clustering_probability

    if use_clustering and k > 1:
        # Strategy B: Grow a spatially contiguous cluster
        selected = _select_clustered_regions(connectivity, k, rng)
    else:
        # Strategy A: Uniform random selection (also used for k=1)
        selected = rng.choice(n_regions, size=k, replace=False)

    # Convert selected indices to a boolean mask
    mask = np.zeros(n_regions, dtype=np.bool_)
    mask[selected] = True

    logger.debug(
        f"Sampled {k} epileptogenic regions "
        f"({'clustered' if use_clustering and k > 1 else 'random'}): "
        f"indices {sorted(selected)}"
    )

    return mask


def _select_clustered_regions(
    connectivity: NDArray[np.float64],
    k: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """
    Select k spatially contiguous regions by growing from a seed.

    Starting from a randomly chosen seed region, we iteratively add the
    region that has the strongest structural connection to the current
    cluster (i.e., the highest total connectivity weight to already-selected
    regions). This produces anatomically plausible epileptogenic zones
    that follow white matter pathways.

    Algorithm:
        1. Pick a random seed region
        2. For each subsequent region:
           a. Sum connectivity weights from each candidate to all selected
           b. Pick the candidate with the highest sum
           c. Add it to the cluster
        3. Return all selected indices

    Args:
        connectivity: (76, 76) float64 preprocessed connectivity matrix.
        k: Number of regions to select (≥ 2).
        rng: NumPy random Generator.

    Returns:
        NDArray[np.int64]: Array of k region indices forming a cluster.
    """
    n = connectivity.shape[0]

    # Start with a random seed region
    seed = rng.integers(0, n)
    selected = [seed]

    # Track which regions are still available for selection
    available = set(range(n)) - {seed}

    for _ in range(k - 1):
        if not available:
            # Extremely unlikely: we've selected all 76 regions
            break

        # For each available region, compute its total connectivity
        # to all currently selected regions. The region with the
        # strongest total connection to the cluster is added next.
        available_list = np.array(list(available))

        # Sum of connectivity weights from each candidate to all
        # currently selected regions
        # connectivity[available_list][:, selected] gives the submatrix
        # of connections from candidates to selected regions
        connection_strengths = connectivity[available_list][:, selected].sum(
            axis=1
        )

        # Pick the candidate with the strongest connection to the cluster
        best_idx = available_list[np.argmax(connection_strengths)]
        selected.append(best_idx)
        available.remove(best_idx)

    return np.array(selected, dtype=np.int64)


def sample_x0_vector(
    epileptogenic_mask: NDArray[np.bool_],
    x0_epileptogenic_range: tuple = DEFAULT_X0_EPILEPTOGENIC,
    x0_healthy_range: tuple = DEFAULT_X0_HEALTHY,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Sample the excitability (x0) parameter for each brain region.

    The x0 parameter is the primary variable of interest in the Epileptor
    model. It controls whether a region produces normal background activity
    or pathological epileptiform discharges:
      - Healthy regions get x0 ~ Uniform(-2.2, -2.05): stable fixed point
      - Epileptogenic regions get x0 ~ Uniform(-1.8, -1.2): spikes/seizures

    The gap between healthy and epileptogenic ranges (-2.05 to -1.8) creates
    a clear separation, but the within-range randomization ensures diversity.

    Args:
        epileptogenic_mask: Boolean mask of shape (76,). True = epileptogenic.
        x0_epileptogenic_range: (low, high) for epileptogenic x0 values.
            Default (-1.8, -1.2) covers interictal spikes to seizures.
        x0_healthy_range: (low, high) for healthy x0 values.
            Default (-2.2, -2.05) produces only background oscillations.
        rng: NumPy random Generator for reproducibility.

    Returns:
        NDArray[np.float64]: Vector of x0 values, shape (76,).
            Each element is in the appropriate range based on the mask.

    References:
        Technical Specs Section 3.4.1 (step 3) and Section 3.2.2
    """
    if rng is None:
        rng = np.random.default_rng()

    x0_vector = np.zeros(N_REGIONS, dtype=np.float64)

    # Assign healthy x0 values to non-epileptogenic regions
    n_healthy = np.sum(~epileptogenic_mask)
    x0_vector[~epileptogenic_mask] = rng.uniform(
        x0_healthy_range[0], x0_healthy_range[1], size=n_healthy
    )

    # Assign epileptogenic x0 values to marked regions
    n_epi = np.sum(epileptogenic_mask)
    x0_vector[epileptogenic_mask] = rng.uniform(
        x0_epileptogenic_range[0], x0_epileptogenic_range[1], size=n_epi
    )

    logger.debug(
        f"Sampled x0 vector: {n_epi} epileptogenic "
        f"(range [{x0_vector[epileptogenic_mask].min():.3f}, "
        f"{x0_vector[epileptogenic_mask].max():.3f}]), "
        f"{n_healthy} healthy "
        f"(range [{x0_vector[~epileptogenic_mask].min():.3f}, "
        f"{x0_vector[~epileptogenic_mask].max():.3f}])"
    )

    return x0_vector


def sample_simulation_parameters(
    connectivity: NDArray[np.float64],
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate a complete set of randomized parameters for one TVB simulation.

    This is the main entry point for parameter sampling. It produces all the
    parameters needed to configure and run one Epileptor simulation:
      - x0_vector: Per-region excitability (76-dimensional)
      - epileptogenic_mask: Which regions are epileptogenic
      - global_coupling: Network-wide coupling strength G
      - noise_intensity: Stochastic integrator noise level D
      - conduction_speed: White matter conduction velocity (m/s)
      - Iext1, Iext2, tau0, tau2: Epileptor model parameters

    Each call to this function produces a unique parameter combination,
    ensuring diversity in the synthetic training dataset.

    Args:
        connectivity: Preprocessed structural connectivity matrix.
            Shape (76, 76), dtype float64. Used for clustering strategy.
        config: Optional configuration dict (from config.yaml). If provided,
            sampling ranges are read from config['neural_mass_model']
            and config['synthetic_data']. If None, defaults from the
            technical specs are used.
        rng: NumPy random Generator for reproducibility. If None, a new
            default Generator is created.

    Returns:
        Dict[str, Any] containing:
            - "x0_vector": NDArray (76,) float64
            - "epileptogenic_mask": NDArray (76,) bool
            - "global_coupling": float
            - "noise_intensity": float
            - "conduction_speed": float
            - "Iext1": float
            - "Iext2": float
            - "tau0": float
            - "tau2": float

    References:
        Technical Specs Section 3.4.1 (complete sampling strategy)
        Technical Specs Section 3.2.2 (parameter table with ranges)
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Extract sampling ranges from config, or use defaults ---
    if config is not None:
        nmm_config = config.get("neural_mass_model", {})
        syn_config = config.get("synthetic_data", {})
        ranges = nmm_config.get("sampling_ranges", {})

        x0_epi = tuple(ranges.get("x0_epileptogenic", DEFAULT_X0_EPILEPTOGENIC))
        x0_healthy = tuple(ranges.get("x0_healthy", DEFAULT_X0_HEALTHY))
        gc_range = tuple(ranges.get("global_coupling", DEFAULT_GLOBAL_COUPLING))
        noise_range = tuple(ranges.get("noise_intensity", DEFAULT_NOISE_INTENSITY))
        speed_range = tuple(ranges.get("conduction_speed", DEFAULT_CONDUCTION_SPEED))
        iext1_range = tuple(ranges.get("Iext1", DEFAULT_IEXT1))
        iext2_range = tuple(ranges.get("Iext2", DEFAULT_IEXT2))
        tau0_range = tuple(ranges.get("tau0", DEFAULT_TAU0))
        tau2_range = tuple(ranges.get("tau2", DEFAULT_TAU2))
        n_epi_range = tuple(
            syn_config.get("n_epileptogenic_range", DEFAULT_N_EPILEPTOGENIC)
        )
        cluster_prob = syn_config.get(
            "clustering_probability", DEFAULT_CLUSTERING_PROB
        )
    else:
        x0_epi = DEFAULT_X0_EPILEPTOGENIC
        x0_healthy = DEFAULT_X0_HEALTHY
        gc_range = DEFAULT_GLOBAL_COUPLING
        noise_range = DEFAULT_NOISE_INTENSITY
        speed_range = DEFAULT_CONDUCTION_SPEED
        iext1_range = DEFAULT_IEXT1
        iext2_range = DEFAULT_IEXT2
        tau0_range = DEFAULT_TAU0
        tau2_range = DEFAULT_TAU2
        n_epi_range = DEFAULT_N_EPILEPTOGENIC
        cluster_prob = DEFAULT_CLUSTERING_PROB

    # --- Sample epileptogenic regions and x0 vector ---
    # Step 1-2: Which regions, how many, random vs clustered
    epileptogenic_mask = sample_epileptogenic_regions(
        n_regions=N_REGIONS,
        connectivity=connectivity,
        n_epileptogenic_range=n_epi_range,
        clustering_probability=cluster_prob,
        rng=rng,
    )

    # Step 3: Excitability values for each region
    x0_vector = sample_x0_vector(
        epileptogenic_mask=epileptogenic_mask,
        x0_epileptogenic_range=x0_epi,
        x0_healthy_range=x0_healthy,
        rng=rng,
    )

    # Step 4: Global coupling strength G
    # Controls how strongly regions influence each other through the
    # structural connectivity. Higher G → more synchronization.
    global_coupling = rng.uniform(gc_range[0], gc_range[1])

    # Step 5: Noise intensity D (log-uniform sampling)
    # Log-uniform is appropriate because noise varies over orders of magnitude
    # log-uniform between [a, b] → exp(uniform(log(a), log(b)))
    noise_intensity = np.exp(
        rng.uniform(np.log(noise_range[0]), np.log(noise_range[1]))
    )

    # Step 6: Conduction speed v (m/s)
    # Determines propagation delays: delay_ij = tract_length_ij / v
    conduction_speed = rng.uniform(speed_range[0], speed_range[1])

    # Additional Epileptor parameters sampled from their respective ranges
    iext1 = rng.uniform(iext1_range[0], iext1_range[1])
    iext2 = rng.uniform(iext2_range[0], iext2_range[1])
    tau0 = rng.uniform(tau0_range[0], tau0_range[1])
    tau2 = rng.uniform(tau2_range[0], tau2_range[1])

    params = {
        "x0_vector": x0_vector,
        "epileptogenic_mask": epileptogenic_mask,
        "global_coupling": global_coupling,
        "noise_intensity": noise_intensity,
        "conduction_speed": conduction_speed,
        "Iext1": iext1,
        "Iext2": iext2,
        "tau0": tau0,
        "tau2": tau2,
    }

    logger.debug(
        f"Sampled parameters: "
        f"n_epi={int(epileptogenic_mask.sum())}, "
        f"G={global_coupling:.3f}, "
        f"D={noise_intensity:.6f}, "
        f"v={conduction_speed:.2f} m/s"
    )

    return params
