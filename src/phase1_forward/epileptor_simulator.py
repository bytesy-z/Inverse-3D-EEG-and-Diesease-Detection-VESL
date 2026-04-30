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
  - Time step: 0.1 ms (must be ≤ 0.1 for numerical stability)
  - Output: Raw monitor at full dt resolution, then anti-aliased decimation
    to 200 Hz using scipy.signal.decimate (FIR low-pass filter).
    TVB's TemporalAverage monitor does NOT apply anti-aliasing, which causes
    severe spectral aliasing (>90% power folding to Nyquist). Proper FIR
    decimation preserves the biologically correct spectral content.
  - Duration: 12 seconds total, first 2 seconds discarded as transient
  - Output variable: x2 - x1 (LFP proxy)
  - Result shape: (76, 2000) — 76 regions × 2000 time points (10s at 200Hz)
  - Noise: Structured — applied only to fast subsystem (x1, y1, x2, y2),
    NOT to slow variables (z, g) to prevent numerical divergence.
  - TVB 2× rate factor: TVB outputs at 2× the nominal rate (dt=0.1 ms yields
    20 kHz, not 10 kHz). Decimation factor accounts for this.

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
from scipy.signal import decimate

# TVB imports — these are the core simulation components
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Difference
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.monitors import Raw
from tvb.simulator.noise import Additive
from tvb.simulator.simulator import Simulator

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Sections 3.2 and 3.4.2)
# ---------------------------------------------------------------------------
N_REGIONS = 76              # Desikan-Killiany parcellation
N_STATE_VARS = 6            # Epileptor has 6 state variables: x1, y1, z, x2, y2, g
DT = 0.1                    # Integration time step in ms (HeunStochastic)
                            # Must be ≤ 0.1 for numerical stability with stochastic
                            # Epileptor. dt=1.0 causes overflow in the fast subsystem.
SIMULATION_LENGTH_MS = 12000.0   # Total simulation length: 12 seconds
TRANSIENT_MS = 2000.0       # First 2 seconds discarded as initial transient
SAMPLING_RATE = 200.0       # Target output sampling rate in Hz

# TVB 2× output rate factor: TVB outputs at 2× the nominal rate.
# For dt=0.1 ms, the nominal rate is 1000/0.1 = 10,000 Hz, but TVB's
# Raw monitor actually yields at 20,000 Hz. This factor is used to
# compute the correct decimation ratio: actual_fs / target_fs.
TVB_RATE_FACTOR = 2.0

# After discarding transient: 10 seconds at 200 Hz = 2000 time points
EXPECTED_OUTPUT_TIMEPOINTS = int(
    (SIMULATION_LENGTH_MS - TRANSIENT_MS) / 1000.0 * SAMPLING_RATE
)

# Configure module-level logger
logger = logging.getLogger(__name__)


def _factorize_decimation(total_factor: int, max_stage: int = 10) -> list:
    """
    Factorize a large decimation factor into stages of at most max_stage.

    scipy.signal.decimate works best with small decimation factors (≤10-13).
    For larger factors, multi-stage decimation produces cleaner anti-aliasing.
    For example, a total factor of 100 is factored as [10, 10].

    This function uses a greedy approach: repeatedly extract the largest
    factor ≤ max_stage until the remainder is 1.

    Args:
        total_factor: The total decimation factor to achieve.
            Must be a positive integer ≥ 1.
        max_stage: Maximum decimation factor per stage. Default 10.

    Returns:
        List of integer factors whose product equals total_factor.
        Each factor is ≤ max_stage.

    Raises:
        ValueError: If total_factor < 1 or has a prime factor > max_stage.

    Examples:
        >>> _factorize_decimation(100)
        [10, 10]
        >>> _factorize_decimation(50)
        [10, 5]
        >>> _factorize_decimation(200)
        [10, 10, 2]
    """
    if total_factor < 1:
        raise ValueError(
            f"Decimation factor must be ≥ 1, got {total_factor}."
        )
    if total_factor == 1:
        return [1]

    stages = []
    remaining = total_factor
    while remaining > 1:
        # Find the largest factor of 'remaining' that is ≤ max_stage
        found = False
        for divisor in range(min(remaining, max_stage), 1, -1):
            if remaining % divisor == 0:
                stages.append(divisor)
                remaining //= divisor
                found = True
                break
        if not found:
            # remaining is a prime > max_stage — cannot factorize cleanly
            raise ValueError(
                f"Cannot factorize decimation factor {total_factor} into "
                f"stages of at most {max_stage}. Remaining prime factor: "
                f"{remaining}."
            )

    return stages


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
    # These control the dynamics of the fast and slow subsystems.
    #
    # CRITICAL PARAMETER MAPPING (TVB attribute ↔ our naming):
    #   tau0 (slow time constant, ms) → model.r = 1.0 / tau0
    #       TVB uses r (the rate) directly in the z-equation:
    #       dz/dt = r * (4*(x1 - x0) - z)
    #       Default r=0.00035 corresponds to tau0 ≈ 2857 ms.
    #   tau2 (y2 time constant) → model.tau
    #       Used in the y2-equation: dy2/dt = (-y2 + f2(x2)) / tau
    #       Default tau=10.0.
    #   model.tt is a global time-scaling factor (dimensionless), NOT tau2.
    #       Must stay at the default value of 1.0.
    #
    # WARNING: Python silently creates new attributes on assignment
    # (e.g., model.tau_s = X creates a new attr ignored by TVB).
    # Only set attributes that exist in TVB's Epileptor declarative attrs.
    model.Iext = np.array([params.get("Iext1", 3.1)])    # External input 1
    model.Iext2 = np.array([params.get("Iext2", 0.45)])  # External input 2

    # tau0 → r: Convert slow time constant (ms) to rate (1/ms)
    tau0_ms = params.get("tau0", 2857.0)
    model.r = np.array([1.0 / tau0_ms])

    # tau2 → tau: Set the y2-equation time constant directly
    tau2_val = params.get("tau2", 10.0)
    model.tau = np.array([tau2_val])

    # model.tt stays at default 1.0 — it is a global time scaling factor,
    # NOT related to our tau2 parameter.

    # y0 is fixed at 1.0 per technical specs (equilibrium point)
    # (y0 does not need explicit setting — TVB default is 1.0)

    logger.debug(
        f"Configured Epileptor: x0 range [{x0_vector.min():.3f}, "
        f"{x0_vector.max():.3f}], "
        f"Iext1={float(np.asarray(model.Iext).flat[0]):.3f}, "
        f"Iext2={float(np.asarray(model.Iext2).flat[0]):.3f}"
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
) -> NDArray[np.float64]:
    """
    Run one complete TVB Epileptor simulation and return source activity.

    This is the main simulation function. It:
      1. Builds a TVB Connectivity from our data files
      2. Configures the Epileptor model with the given parameters
      3. Sets up the HeunStochastic integrator with additive noise
      4. Runs the simulation at full dt resolution (Raw monitor)
      5. Extracts the LFP proxy (x2 - x1) from the Epileptor output
      6. Applies anti-aliased decimation to reach 200 Hz
      7. Discards the initial transient
      8. Returns the clean source activity time series

    The output variable (x2 - x1) is the standard LFP proxy in the Epileptor
    literature (Jirsa et al., 2014; Proix et al., 2017). It combines the fast
    subsystem variable x1 with the slow subsystem variable x2 to produce a
    signal that resembles local field potentials recorded intracranially.

    IMPORTANT — Anti-aliasing:
        TVB's TemporalAverage monitor performs simple box-car averaging
        (no proper anti-alias filter). The Epileptor fast subsystem generates
        significant high-frequency energy (>500 Hz at dt=0.1ms), which
        TemporalAverage aliases into the Nyquist bin (~100 Hz at 200 Hz
        output), producing unphysical spectral content (~90% of power at
        Nyquist). We instead use the Raw monitor and apply scipy's FIR-based
        decimate() for proper anti-aliased downsampling. This preserves the
        biologically correct spectral profile (dominant power <30 Hz).

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
        dt: Integration time step in milliseconds. Default 0.1.
            Must be ≤ 0.1 for numerical stability with stochastic Epileptor.

    Returns:
        NDArray[np.float64]: Source activity (LFP proxy: x2 - x1).
            Shape (76, 2000). dtype float64.
            Anti-alias filtered and decimated to 200 Hz.

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
    # [x1, y1, z, x2, y2, g] at indices [0, 1, 2, 3, 4, 5].
    #
    # CRITICAL: Noise must only be applied to the FAST subsystem variables
    # (x1, y1, x2, y2) and NOT to the slow variables (z at index 2, g at
    # index 5). The slow variables z and g have very large effective time
    # constants (tau0 ≈ 2857 ms), so even small noise perturbations
    # accumulate and cause numerical divergence. This is consistent with
    # the Epileptor literature (Jirsa et al., 2014; Proix et al., 2017)
    # where stochastic terms enter only the fast subsystem.
    #
    # nsig structure: [D, D, 0.0, D, D, 0.0]
    #                  x1 y1  z   x2 y2  g
    nsig = np.array([
        noise_intensity,   # x1 — fast subsystem population 1
        noise_intensity,   # y1 — fast subsystem population 1
        0.0,               # z  — slow energy variable (NO noise)
        noise_intensity,   # x2 — fast subsystem population 2
        noise_intensity,   # y2 — fast subsystem population 2
        0.0,               # g  — slow permittivity variable (NO noise)
    ])

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
    # We use the Raw monitor to capture all integration steps at full dt
    # resolution. This is necessary because TVB's TemporalAverage monitor
    # applies only a simple box-car average (no anti-aliasing filter),
    # which causes severe spectral aliasing — the Epileptor's fast subsystem
    # generates energy up to several kHz, and without a proper low-pass
    # filter, this energy folds back to the Nyquist frequency of the output
    # (100 Hz at 200 Hz sampling), producing unphysical spectral content.
    #
    # Instead, we capture the full-resolution output and apply proper
    # anti-aliased decimation in Step 8 using scipy.signal.decimate (FIR).
    monitor = Raw()

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
    # TVB returns a generator of (time, data) tuples from each monitor.
    # The Raw monitor yields at every integration step (dt=0.1 ms).
    # Due to TVB's internal 2× rate factor, the actual output rate is
    # 2 × (1000/dt) Hz = 20,000 Hz for dt=0.1.
    raw_output = []
    for (time_array, data_array), in simulator():
        raw_output.append(data_array)

    # Concatenate all time steps into a single array.
    # For the Raw monitor with Epileptor, TVB computes the default
    # observable (x2 - x1) and returns it as shape (T, n_regions, 1).
    full_output = np.concatenate(raw_output, axis=0)

    logger.debug(
        f"Simulation complete. Raw output shape: {full_output.shape}, "
        f"ndim: {full_output.ndim}"
    )

    # --- Step 8: Extract LFP proxy and apply anti-aliased decimation ---
    # TVB's Raw monitor returns the pre-computed Epileptor observable
    # (x2 - x1) with shape (T, n_regions, 1). We extract it and then
    # decimate from the actual raw sampling rate down to 200 Hz.

    if full_output.ndim == 3 and full_output.shape[2] == 1:
        # Shape: (T, n_regions, 1) — the common case.
        # TVB already computed x2 - x1 (the Epileptor's default observable).
        lfp_raw = full_output[:, :, 0]  # (T_raw, n_regions)
    elif full_output.ndim == 4:
        # Shape: (T, n_state_vars, n_regions, n_modes)
        # Extract x1 and x2 manually
        x1 = full_output[:, 0, :, 0]
        x2 = full_output[:, 3, :, 0]
        lfp_raw = x2 - x1
    else:
        raise ValueError(
            f"Unexpected output shape from TVB Raw monitor: "
            f"{full_output.shape}. Expected (T, 76, 1) or (T, 6, 76, 1)."
        )

    # Compute the actual raw sampling rate.
    # TVB outputs at 2× the nominal rate: actual_fs = 2 × (1000/dt).
    actual_raw_fs = TVB_RATE_FACTOR * (1000.0 / dt)

    # Total decimation factor to reach target sampling rate (200 Hz).
    total_decimation = int(actual_raw_fs / SAMPLING_RATE)
    logger.debug(
        f"Anti-aliased decimation: {actual_raw_fs:.0f} Hz → "
        f"{SAMPLING_RATE:.0f} Hz (factor {total_decimation})"
    )

    # Apply multi-stage anti-aliased decimation using scipy's FIR filter.
    # For large decimation factors (e.g., 100), we split into stages
    # for better filter performance: 100 = 10 × 10, or 50 = 10 × 5, etc.
    # scipy.signal.decimate applies a FIR low-pass anti-aliasing filter
    # before downsampling, preventing spectral aliasing.
    n_raw_timepoints, n_regions_out = lfp_raw.shape

    # Factorize the decimation into stages of at most 10
    stages = _factorize_decimation(total_decimation)

    # Compute expected output length after decimation
    expected_decimated_length = n_raw_timepoints
    for s in stages:
        expected_decimated_length = expected_decimated_length // s

    # Apply decimation stage-by-stage, region-by-region
    lfp_decimated = np.zeros(
        (expected_decimated_length, n_regions_out), dtype=np.float64
    )
    for r in range(n_regions_out):
        signal_1d = lfp_raw[:, r].astype(np.float64)
        for stage_factor in stages:
            signal_1d = decimate(signal_1d, stage_factor, ftype='fir')
        lfp_decimated[:, r] = signal_1d

    logger.debug(
        f"Decimated LFP shape: {lfp_decimated.shape}, "
        f"effective rate: {SAMPLING_RATE:.0f} Hz"
    )

    # --- Step 9: Discard initial transient ---
    # The first TRANSIENT_MS milliseconds contain the model's initial
    # settling dynamics, which are not representative of steady-state
    # activity. At 200 Hz, 2000 ms = 400 time points to discard.
    transient_samples = int(TRANSIENT_MS / 1000.0 * SAMPLING_RATE)
    source_activity = lfp_decimated[transient_samples:, :]

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
            f"got {source_activity.shape[1]}. This may indicate incorrect "
            f"decimation factor or TVB rate factor. Proceeding with actual "
            f"length."
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
