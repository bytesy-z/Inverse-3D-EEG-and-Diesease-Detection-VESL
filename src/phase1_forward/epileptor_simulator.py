"""
Module: epileptor_simulator.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Wrapper around TVB's Epileptor neural mass model for running batch
         simulations that generate realistic source-level brain activity.

This module bridges the gap between our parameter sampling (parameter_sampler.py)
and the synthetic dataset generation (synthetic_dataset.py). Given a set of
Epileptor parameters (x0 vector, coupling, noise, etc.), it configures and runs
a TVB simulation, then extracts the LFP proxy (x2 - x1) as the source signal.

The Epileptor model (Jirsa et al., 2014) is a 6-variable ODE system per brain
region that reproduces realistic seizure dynamics. The x0 parameter controls
epileptogenicity: healthy regions remain stable, while epileptogenic regions
produce spontaneous spikes or seizure-like oscillations.

Key simulation details:
  - Integrator: HeunStochastic (second-order stochastic Runge-Kutta)
  - Time step: 1.0 ms
  - Output: TemporalAverage at 200 Hz (period=5.0 ms)
  - Duration: 12 seconds total, first 2 seconds discarded as transient
  - Output variable: x2 - x1 (LFP proxy)
  - Result shape: (76, 2000) — 76 regions × 2000 time points (10s at 200Hz)

See docs/02_TECHNICAL_SPECIFICATIONS.md Sections 3.2 and 3.4.2 for full specs.

Key dependencies:
- tvb-library: Epileptor model, simulator engine, coupling, integrator
- numpy: Array operations for output extraction

Input data format:
- params: Dict from parameter_sampler.sample_simulation_parameters()
- connectivity: (76, 76) float64, preprocessed structural connectivity
- tract_lengths: (76, 76) float64, for conduction delays

Output data format:
- source_activity: (76, 2000) float64 — LFP proxy time series at 200 Hz
"""

# Standard library imports
import logging
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# TVB imports — these are the core simulation components
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Difference
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.noise import Additive
from tvb.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Sections 3.2 and 3.4.2)
# ---------------------------------------------------------------------------
N_REGIONS = 76              # Desikan-Killiany parcellation
DT = 1.0                    # Integration time step in ms (HeunStochastic)
MONITOR_PERIOD = 5.0        # TemporalAverage period in ms → 200 Hz output
SIMULATION_LENGTH_MS = 12000.0   # Total simulation length: 12 seconds
TRANSIENT_MS = 2000.0       # First 2 seconds discarded as initial transient
SAMPLING_RATE = 200.0       # Output sampling rate in Hz (1000/5 = 200)

# After discarding transient: 10 seconds at 200 Hz = 2000 time points
EXPECTED_OUTPUT_TIMEPOINTS = int(
    (SIMULATION_LENGTH_MS - TRANSIENT_MS) / 1000.0 * SAMPLING_RATE
)

# Configure module-level logger
logger = logging.getLogger(__name__)


def build_tvb_connectivity(
    weights: NDArray[np.float64],
    centres: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    conduction_speed: float = 4.0,
) -> Connectivity:
    """
    Build a TVB Connectivity object from our preprocessed source space data.

    TVB's simulator requires a Connectivity object that bundles together the
    structural connectivity weights, region positions, labels, and tract
    lengths. This function creates that object from the data files we saved
    in source_space.py.

    We set the conduction speed on the connectivity object, which TVB uses
    internally to compute propagation delays:
        delay_ij = tract_length_ij / conduction_speed

    Args:
        weights: Preprocessed connectivity weight matrix.
            Shape (76, 76), dtype float64. Values in [0, 1].
        centres: Region centroid coordinates in MNI space.
            Shape (76, 3), dtype float64. Units: mm.
        region_labels: List of 76 region name strings.
        tract_lengths: Inter-region tract lengths.
            Shape (76, 76), dtype float64. Units: mm.
        conduction_speed: Signal propagation velocity along white matter
            tracts, in m/s. Default 4.0 m/s. Sampled from [3.0, 6.0] m/s
            during dataset generation.

    Returns:
        Connectivity: A fully configured TVB Connectivity object ready
            for use in a Simulator.

    References:
        Technical Specs Section 3.2.3 (simulator configuration)
    """
    # Create a new Connectivity object and populate its fields
    connectivity = Connectivity()

    # TVB expects numpy arrays for these fields
    connectivity.weights = weights.copy()
    connectivity.centres = centres.copy()
    connectivity.region_labels = np.array(region_labels)
    connectivity.tract_lengths = tract_lengths.copy()

    # Conduction speed determines propagation delays
    # TVB stores this as a scalar on the connectivity object
    connectivity.speed = np.array([conduction_speed])

    # Configure the connectivity (computes internal derived quantities
    # like the number of regions, hemispheres, etc.)
    connectivity.configure()

    logger.debug(
        f"Built TVB Connectivity: {connectivity.number_of_regions} regions, "
        f"speed={conduction_speed:.2f} m/s"
    )

    return connectivity


def configure_epileptor(
    params: Dict[str, Any],
) -> Epileptor:
    """
    Create and configure a TVB Epileptor model instance with given parameters.

    The Epileptor is a 6-variable neural mass model per region (Jirsa et al.,
    2014). Its dynamics are governed by coupled fast and slow subsystems that
    produce realistic seizure onset, evolution, and termination. The key
    parameter is x0 (excitability), which determines whether a region is
    healthy or epileptogenic.

    Note: x0 is set per-region via the model's state variable initialization,
    not as a model parameter. TVB's Epileptor model has x0 as a parameter
    that can be set as an array with one value per region.

    Args:
        params: Dictionary containing Epileptor parameters:
            - "x0_vector": NDArray (76,) — per-region excitability
            - "Iext1": float — external input to fast subsystem
            - "Iext2": float — external input to slow subsystem
            - "tau0": float — slow time constant (ms)
            - "tau2": float — fast subsystem 2 time constant (ms)

    Returns:
        Epileptor: A configured TVB Epileptor model instance.

    References:
        Technical Specs Section 3.2.1 (Epileptor equations)
        Technical Specs Section 3.2.2 (parameter table)
    """
    model = Epileptor()

    # Set the per-region excitability vector
    # TVB expects this as a 2D array with shape (1, n_regions) or (n_regions,)
    # depending on the TVB version. We use the standard approach.
    x0_vector = params["x0_vector"]
    model.x0 = x0_vector.reshape(-1)

    # Set scalar Epileptor parameters
    # These control the dynamics of the fast and slow subsystems
    model.Iext = np.array([params.get("Iext1", 3.1)])    # External input 1
    model.Iext2 = np.array([params.get("Iext2", 0.45)])  # External input 2
    model.tau_s = np.array([params.get("tau0", 2857.0)])  # Slow time constant
    model.tt = np.array([params.get("tau2", 10.0)])       # Fast time constant 2

    # y0 is fixed at 1.0 per technical specs (equilibrium point)
    model.r = np.array([0.00035])  # Default value from TVB

    logger.debug(
        f"Configured Epileptor: x0 range [{x0_vector.min():.3f}, "
        f"{x0_vector.max():.3f}], "
        f"Iext1={float(model.Iext):.3f}, Iext2={float(model.Iext2):.3f}"
    )

    return model


def run_simulation(
    params: Dict[str, Any],
    connectivity_weights: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    simulation_length_ms: float = SIMULATION_LENGTH_MS,
    dt: float = DT,
    monitor_period: float = MONITOR_PERIOD,
) -> NDArray[np.float64]:
    """
    Run one complete TVB Epileptor simulation and return source activity.

    This is the main simulation function. It:
      1. Builds a TVB Connectivity from our data files
      2. Configures the Epileptor model with the given parameters
      3. Sets up the HeunStochastic integrator with additive noise
      4. Runs the simulation for the specified duration
      5. Extracts the LFP proxy (x2 - x1) from the Epileptor state variables
      6. Discards the initial transient
      7. Returns the clean source activity time series

    The output variable (x2 - x1) is the standard LFP proxy in the Epileptor
    literature (Jirsa et al., 2014; Proix et al., 2017). It combines the fast
    subsystem variable x1 with the slow subsystem variable x2 to produce a
    signal that resembles local field potentials recorded intracranially.

    Args:
        params: Complete parameter dictionary from
            parameter_sampler.sample_simulation_parameters(). Must contain:
            - "x0_vector": NDArray (76,) float64
            - "global_coupling": float
            - "noise_intensity": float
            - "conduction_speed": float
            - "Iext1", "Iext2", "tau0", "tau2": floats
        connectivity_weights: Preprocessed connectivity weight matrix.
            Shape (76, 76), dtype float64.
        region_centers: Region centroid coordinates.
            Shape (76, 3), dtype float64.
        region_labels: List of 76 region name strings.
        tract_lengths: Inter-region tract lengths.
            Shape (76, 76), dtype float64.
        simulation_length_ms: Total simulation length in milliseconds.
            Default 12000.0 (12 seconds).
        dt: Integration time step in milliseconds. Default 1.0.
        monitor_period: TemporalAverage monitor period in ms. Default 5.0
            (gives 200 Hz output).

    Returns:
        NDArray[np.float64]: Source activity (LFP proxy: x2 - x1).
            Shape (76, T) where T = (simulation_length - transient) / period * 1000.
            For default settings: (76, 2000). dtype float64.

    Raises:
        RuntimeError: If TVB simulation fails to produce expected output.

    References:
        Technical Specs Section 3.2.3 (TVB simulator configuration)
        Technical Specs Section 3.4.2 (simulation and projection pipeline)
    """
    logger.debug(
        f"Starting TVB simulation: {simulation_length_ms:.0f} ms, "
        f"dt={dt:.1f} ms, G={params['global_coupling']:.3f}"
    )

    # --- Step 1: Build TVB connectivity ---
    tvb_connectivity = build_tvb_connectivity(
        weights=connectivity_weights,
        centres=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
        conduction_speed=params["conduction_speed"],
    )

    # --- Step 2: Configure the Epileptor model ---
    model = configure_epileptor(params)

    # --- Step 3: Set up the stochastic integrator ---
    # HeunStochastic is a second-order stochastic Runge-Kutta method
    # that handles the additive noise in the Epileptor equations
    noise_intensity = params["noise_intensity"]

    # TVB's noise expects an array of noise intensities, one per state
    # variable. The Epileptor has 6 state variables per region:
    # [x1, y1, z, x2, y2, g]. We add noise primarily to x1 and x2.
    # Standard practice: equal noise on all state variables
    n_state_vars = model.nvar  # Should be 6 for Epileptor
    nsig = np.array([noise_intensity] * n_state_vars)

    noise = Additive(nsig=nsig)
    noise.random_stream.seed(
        int(np.abs(hash(tuple(params["x0_vector"][:5]))) % (2**31))
    )

    integrator = HeunStochastic(dt=dt, noise=noise)

    # --- Step 4: Configure coupling ---
    # Difference coupling: coupling term = G * sum_j W_ij * (x1_j - x1_i)
    # This is the standard coupling function for Epileptor in TVB
    coupling = Difference(a=np.array([params["global_coupling"]]))

    # --- Step 5: Set up the monitor ---
    # TemporalAverage with period=5.0 ms downsamples from the 1ms integration
    # step to 200 Hz output, matching the NMT dataset sampling rate
    monitor = TemporalAverage(period=monitor_period)

    # --- Step 6: Assemble and configure the simulator ---
    simulator = Simulator(
        model=model,
        connectivity=tvb_connectivity,
        coupling=coupling,
        integrator=integrator,
        monitors=[monitor],
        simulation_length=simulation_length_ms,
    )
    simulator.configure()

    # --- Step 7: Run the simulation ---
    # TVB returns a generator of (time, data) tuples from each monitor
    # We collect all output from the TemporalAverage monitor
    raw_output = []
    for (time_array, data_array), in simulator():
        # data_array has shape (n_timepoints, n_state_vars, n_regions, n_modes)
        # For TemporalAverage, n_timepoints is typically 1 per yield
        raw_output.append(data_array)

    # Concatenate all time steps into a single array
    # Shape: (total_timepoints, n_state_vars, n_regions, n_modes)
    full_output = np.concatenate(raw_output, axis=0)

    logger.debug(
        f"Simulation complete. Raw output shape: {full_output.shape}"
    )

    # --- Step 8: Extract the LFP proxy (x2 - x1) ---
    # In TVB's Epileptor, the state variables are ordered:
    # [x1, y1, z, x2, y2, g] → indices [0, 1, 2, 3, 4, 5]
    # LFP proxy = x2 (index 3) - x1 (index 0)
    # We take mode 0 (the only mode for standard Epileptor)
    x1 = full_output[:, 0, :, 0]  # Shape: (total_timepoints, n_regions)
    x2 = full_output[:, 3, :, 0]  # Shape: (total_timepoints, n_regions)
    lfp_proxy = x2 - x1           # Shape: (total_timepoints, n_regions)

    # --- Step 9: Discard initial transient ---
    # The first TRANSIENT_MS milliseconds contain the model's initial
    # settling dynamics, which are not representative of steady-state
    # activity. At 200 Hz, 2000 ms = 400 time points to discard.
    transient_samples = int(TRANSIENT_MS / 1000.0 * SAMPLING_RATE)
    source_activity = lfp_proxy[transient_samples:, :]  # (T_remaining, n_regions)

    # Transpose to (n_regions, T) format matching our pipeline convention
    source_activity = source_activity.T  # (n_regions, T)

    # --- Step 10: Validate output shape ---
    expected_t = EXPECTED_OUTPUT_TIMEPOINTS
    if source_activity.shape[0] != N_REGIONS:
        raise RuntimeError(
            f"Expected {N_REGIONS} regions in output, got "
            f"{source_activity.shape[0]}. TVB simulation produced "
            f"unexpected output."
        )

    if source_activity.shape[1] != expected_t:
        logger.warning(
            f"Expected {expected_t} time points after transient removal, "
            f"got {source_activity.shape[1]}. This may be due to TVB's "
            f"internal temporal averaging. Proceeding with actual length."
        )

    logger.debug(
        f"Source activity extracted: shape {source_activity.shape}, "
        f"value range [{source_activity.min():.4f}, "
        f"{source_activity.max():.4f}]"
    )

    return source_activity.astype(np.float64)


def segment_source_activity(
    source_activity: NDArray[np.float64],
    window_length_samples: int = 400,
    n_windows: int = 5,
) -> NDArray[np.float64]:
    """
    Segment a long source activity time series into fixed-length windows.

    After removing the 2-second transient, each simulation produces ~10
    seconds of source activity at 200 Hz (2000 time points). We segment
    this into non-overlapping 2-second windows (400 samples each), yielding
    5 windows per simulation. This increases the effective dataset size by
    5× without requiring additional simulations.

    Args:
        source_activity: Source activity from one simulation.
            Shape (76, T), dtype float64, where T ≥ window_length * n_windows.
        window_length_samples: Number of samples per window. Default 400
            (2 seconds at 200 Hz).
        n_windows: Number of non-overlapping windows to extract. Default 5.

    Returns:
        NDArray[np.float64]: Segmented windows.
            Shape (n_windows, 76, window_length_samples), dtype float64.

    Raises:
        ValueError: If the source activity doesn't have enough time points
            for the requested number of windows.

    References:
        Technical Specs Section 3.4.2 (step 4 — segmentation)
    """
    n_regions, total_timepoints = source_activity.shape
    required_timepoints = window_length_samples * n_windows

    if total_timepoints < required_timepoints:
        raise ValueError(
            f"Source activity has {total_timepoints} time points, but "
            f"need {required_timepoints} for {n_windows} windows of "
            f"{window_length_samples} samples each. "
            f"Check simulation_length and transient settings."
        )

    # Extract non-overlapping windows
    windows = np.zeros(
        (n_windows, n_regions, window_length_samples), dtype=np.float64
    )
    for i in range(n_windows):
        start = i * window_length_samples
        end = start + window_length_samples
        windows[i] = source_activity[:, start:end]

    logger.debug(
        f"Segmented source activity into {n_windows} windows of "
        f"{window_length_samples} samples each"
    )

    return windows
