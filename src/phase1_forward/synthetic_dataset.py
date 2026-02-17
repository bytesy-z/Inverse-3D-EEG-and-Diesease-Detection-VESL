"""
Module: synthetic_dataset.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Orchestrate the full synthetic EEG dataset generation pipeline by
         combining TVB simulations, leadfield projection, noise addition,
         and HDF5 storage.

This module ties together all Phase 1 components to produce the training,
validation, and test datasets for the PhysDeepSIF network. For each
synthetic sample:
  1. Sample random Epileptor parameters (parameter_sampler.py)
  2. Run a TVB simulation to generate source activity (epileptor_simulator.py)
  3. Project source activity through the leadfield to get clean EEG
  4. Add realistic measurement noise (white Gaussian + colored 1/f)
  5. Segment into 2-second windows
  6. Store everything in HDF5 format

The HDF5 files follow the exact schema specified in Section 3.4.4 and serve
as the primary data interface between Phase 1 (forward modeling) and Phase 2
(network training). Normalization is NOT applied here — it happens on-the-fly
in the PyTorch Dataset's __getitem__() method during training.

Generation is parallelized using joblib to utilize multiple CPU cores, since
each TVB simulation is independent and takes ~1 second.

See docs/02_TECHNICAL_SPECIFICATIONS.md Sections 3.4.2–3.4.5 for full specs.

Key dependencies:
- h5py: HDF5 file I/O for dataset storage
- joblib: Parallel execution of independent simulations
- numpy: Array operations, noise generation
- scipy.signal: Colored noise generation (1/f^alpha spectrum)

Input data format:
- connectivity: (76, 76) float64 from source_space.py
- leadfield: (19, 76) float64 from leadfield_builder.py
- config: dict from config.yaml

Output data format:
- HDF5 files with schema per Section 3.4.4:
    eeg: (N, 19, 400) float32
    source_activity: (N, 76, 400) float32
    epileptogenic_mask: (N, 76) bool
    x0_vector: (N, 76) float32
    snr_db: (N,) float32
    global_coupling: (N,) float32
    metadata group with channel_names, region_names, etc.
"""

# Standard library imports
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Third-party imports
import h5py
import numpy as np
from numpy.typing import NDArray

# Suppress runtime warnings from TVB, numba, and scipy in worker processes
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.history')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numba.np.ufunc.gufunc')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.coupling')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tvb.simulator.models.epileptor')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.signal')

# Local imports
from .epileptor_simulator import run_simulation, segment_source_activity
from .parameter_sampler import sample_simulation_parameters

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Sections 3.3.2, 3.4)
# ---------------------------------------------------------------------------
N_REGIONS = 76              # Desikan-Killiany parcellation
N_CHANNELS = 19             # Standard 10-20 montage
SAMPLING_RATE = 200.0       # Hz
WINDOW_LENGTH_SAMPLES = 400 # 2 seconds at 200 Hz
WINDOWS_PER_SIM = 5         # Non-overlapping 2-second windows from 10s signal

# Default noise parameters
DEFAULT_SNR_RANGE_DB = (5, 30)
DEFAULT_COLORED_NOISE_ALPHA_RANGE = (0.5, 1.5)
DEFAULT_COLORED_NOISE_AMPLITUDE_FRACTION = (0.1, 0.3)

# Channel names in the standard NMT order
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Configure module-level logger
logger = logging.getLogger(__name__)


def project_to_eeg(
    source_activity: NDArray[np.float64],
    leadfield: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Project source-level activity through the leadfield to produce clean EEG.

    This implements the forward model equation:
        EEG(t) = L · S(t)

    where L is the 19×76 leadfield matrix and S(t) is the 76-dimensional
    source activity at time t. The result is the "clean" scalp EEG before
    any measurement noise is added.

    Args:
        source_activity: Source-level time series from TVB simulation.
            Shape (76, T), dtype float64.
        leadfield: The leadfield matrix mapping sources to sensors.
            Shape (19, 76), dtype float64.

    Returns:
        NDArray[np.float64]: Clean scalp EEG.
            Shape (19, T), dtype float64.

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication.

    References:
        Technical Specs Section 3.4.2 (step 5 — leadfield projection)
        Key equation: EEG(t) = L · S(t) + η(t)
    """
    if leadfield.shape[1] != source_activity.shape[0]:
        raise ValueError(
            f"Leadfield has {leadfield.shape[1]} columns but source activity "
            f"has {source_activity.shape[0]} rows. These must match "
            f"(both should be {N_REGIONS})."
        )

    # Matrix multiplication: (19, 76) @ (76, T) → (19, T)
    clean_eeg = leadfield @ source_activity

    return clean_eeg


def generate_white_noise(
    signal: NDArray[np.float64],
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Generate additive white Gaussian noise at a specified SNR level.

    SNR (Signal-to-Noise Ratio) in decibels is defined as:
        SNR_dB = 10 · log10(P_signal / P_noise)

    Solving for noise power:
        P_noise = P_signal / 10^(SNR_dB / 10)

    We compute signal power as the mean squared value across all channels
    and time points, then generate Gaussian noise with the corresponding
    variance.

    Args:
        signal: The clean EEG signal to add noise to.
            Shape (N_channels, T), dtype float64.
        snr_db: Target signal-to-noise ratio in decibels.
            Lower values = more noise. Range [5, 30] dB per specs.
        rng: NumPy random Generator for reproducibility.

    Returns:
        NDArray[np.float64]: White Gaussian noise array with the same
            shape as the input signal. Add this to the signal to get
            noisy EEG at the specified SNR.

    References:
        Technical Specs Section 3.4.2 (step 6 — noise addition)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute signal power (mean squared amplitude)
    signal_power = np.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    # SNR_linear = 10^(SNR_dB / 10) = P_signal / P_noise
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Compute required noise power and standard deviation
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    # Generate white Gaussian noise
    noise = rng.normal(0.0, noise_std, size=signal.shape)

    return noise


def generate_colored_noise(
    n_channels: int,
    n_timepoints: int,
    alpha: float,
    amplitude_fraction: float,
    signal_rms: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Generate colored (1/f^alpha) noise to simulate realistic EEG background.

    Real EEG noise is not purely white — it has a characteristic 1/f^alpha
    power spectrum where lower frequencies have more power. This "pink" to
    "brown" noise component models the background brain activity and
    environmental interference that isn't captured by the Epileptor model.

    Algorithm:
        1. Generate white noise in the frequency domain (random phases)
        2. Shape the spectrum by multiplying by 1/f^(alpha/2)
        3. Inverse FFT to get time-domain colored noise
        4. Scale to the desired amplitude relative to signal RMS

    Args:
        n_channels: Number of EEG channels (19).
        n_timepoints: Number of time samples (400 for 2-second windows).
        alpha: Spectral exponent. alpha=0 is white, alpha=1 is pink,
            alpha=2 is brown noise. Sampled from [0.5, 1.5] per specs.
        amplitude_fraction: Fraction of signal RMS for noise amplitude.
            Sampled from [0.1, 0.3] per specs.
        signal_rms: RMS amplitude of the clean EEG signal, used to scale
            the colored noise to the desired fraction.
        rng: NumPy random Generator for reproducibility.

    Returns:
        NDArray[np.float64]: Colored noise array.
            Shape (n_channels, n_timepoints), dtype float64.

    References:
        Technical Specs Section 3.4.2 (step 6 — colored noise component)
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = np.zeros((n_channels, n_timepoints), dtype=np.float64)

    for ch in range(n_channels):
        # Generate white noise
        white = rng.normal(0.0, 1.0, size=n_timepoints)

        # Transform to frequency domain
        white_fft = np.fft.rfft(white)

        # Create 1/f^(alpha/2) filter
        # The power spectrum goes as 1/f^alpha, so the amplitude
        # spectrum goes as 1/f^(alpha/2)
        freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / SAMPLING_RATE)

        # Avoid division by zero at DC (freq=0)
        freqs[0] = 1.0  # Set DC to 1 Hz to avoid inf

        # Shape the spectrum: multiply by 1/f^(alpha/2)
        spectral_filter = 1.0 / (freqs ** (alpha / 2.0))

        # Apply the filter
        colored_fft = white_fft * spectral_filter

        # Transform back to time domain
        colored = np.fft.irfft(colored_fft, n=n_timepoints)

        # Normalize to unit variance, then scale
        std = np.std(colored)
        if std > 1e-10:
            colored = colored / std

        noise[ch, :] = colored

    # Scale to desired amplitude (fraction of signal RMS)
    target_rms = amplitude_fraction * signal_rms
    noise = noise * target_rms

    return noise


def add_measurement_noise(
    clean_eeg: NDArray[np.float64],
    snr_db: float,
    colored_noise_alpha: float,
    colored_noise_amplitude: float,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Add combined white and colored noise to clean EEG to simulate measurement.

    Real EEG recordings contain multiple noise sources:
      - Thermal/electronic noise (white Gaussian)
      - Background brain activity not modeled by Epileptor (colored 1/f)
      - Environmental interference (partially captured by both)

    This function combines both noise types to produce realistic synthetic
    EEG training data. The white noise level is controlled by SNR_dB, and
    the colored noise is added as a fraction of the clean signal's RMS.

    Args:
        clean_eeg: Noise-free EEG from leadfield projection.
            Shape (19, T), dtype float64.
        snr_db: Target SNR for white noise, in dB. Range [5, 30].
        colored_noise_alpha: Spectral exponent for 1/f noise. Range [0.5, 1.5].
        colored_noise_amplitude: Fraction of signal RMS for colored noise.
            Range [0.1, 0.3].
        rng: NumPy random Generator for reproducibility.

    Returns:
        NDArray[np.float64]: Noisy EEG signal.
            Shape (19, T), dtype float64.

    References:
        Technical Specs Section 3.4.2 (step 6)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_channels, n_timepoints = clean_eeg.shape
    signal_rms = np.sqrt(np.mean(clean_eeg ** 2))

    # Add white Gaussian noise at specified SNR
    white_noise = generate_white_noise(clean_eeg, snr_db, rng=rng)

    # Add colored 1/f noise at specified amplitude
    colored_noise = generate_colored_noise(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        alpha=colored_noise_alpha,
        amplitude_fraction=colored_noise_amplitude,
        signal_rms=signal_rms,
        rng=rng,
    )

    # Combine: noisy_eeg = clean_eeg + white_noise + colored_noise
    noisy_eeg = clean_eeg + white_noise + colored_noise

    return noisy_eeg


def generate_one_simulation(
    sim_index: int,
    connectivity: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    leadfield: NDArray[np.float64],
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Optional[Dict[str, NDArray]]:
    """
    Generate all synthetic data samples from one TVB simulation.

    This function encapsulates the complete per-simulation pipeline:
      1. Sample random parameters
      2. Run TVB Epileptor simulation
      3. Segment source activity into 2-second windows
      4. Project each window through the leadfield to get clean EEG
      5. Add measurement noise (white + colored) to each window
      6. Return all windows with their metadata

    Each simulation produces multiple windows (default: 5), so this function
    returns data for 5 training samples from one simulation run.

    This function is designed to be called in parallel by joblib — it is
    self-contained and only uses its arguments (no shared mutable state).

    Args:
        sim_index: Index of this simulation (for logging and seed generation).
        connectivity: Preprocessed connectivity matrix. (76, 76) float64.
        region_centers: Region centroids. (76, 3) float64.
        region_labels: List of 76 region names.
        tract_lengths: Tract length matrix. (76, 76) float64.
        leadfield: Leadfield matrix. (19, 76) float64.
        config: Configuration dict from config.yaml.
        seed: Random seed for reproducibility. If None, uses sim_index.

    Returns:
        Dict containing:
            - "eeg": (n_windows, 19, 400) float64, noisy EEG
            - "source_activity": (n_windows, 76, 400) float64
            - "epileptogenic_mask": (76,) bool (same for all windows)
            - "x0_vector": (76,) float64 (same for all windows)
            - "snr_db": (n_windows,) float64
            - "global_coupling": float64
        Returns None if the simulation fails (logged as warning).

    References:
        Technical Specs Sections 3.4.1–3.4.2
    """
    # Set up reproducible random state for this simulation
    if seed is None:
        seed = sim_index
    rng = np.random.default_rng(seed)

    try:
        # Step 1: Sample random parameters for this simulation
        params = sample_simulation_parameters(
            connectivity=connectivity, config=config, rng=rng
        )

        # Step 2: Run TVB Epileptor simulation
        source_activity = run_simulation(
            params=params,
            connectivity_weights=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
        )

        # Step 2b: Validate simulation output — reject diverged simulations
        # Some parameter combinations (high coupling + high noise + extreme x0)
        # cause the Epileptor to overflow. TVB may produce NaN or Inf in some
        # regions without raising an exception (RuntimeWarning only). We must
        # detect and discard these simulations to keep the dataset clean.
        if np.isnan(source_activity).any() or np.isinf(source_activity).any():
            nan_count = np.isnan(source_activity).sum()
            inf_count = np.isinf(source_activity).sum()
            logger.warning(
                f"Simulation {sim_index} produced invalid values "
                f"(NaN={nan_count}, Inf={inf_count}). Discarding."
            )
            return None

        # Step 3: Segment into 2-second non-overlapping windows
        # source_activity shape: (76, ~2000) → windows shape: (5, 76, 400)
        syn_config = config.get("synthetic_data", {}) if config else {}
        n_windows = syn_config.get("windows_per_simulation", WINDOWS_PER_SIM)
        wl = syn_config.get("window_length_samples", WINDOW_LENGTH_SAMPLES)

        source_windows = segment_source_activity(
            source_activity, window_length_samples=wl, n_windows=n_windows
        )

        # Steps 4-5: Project to EEG and add noise for each window
        snr_range = tuple(syn_config.get("snr_range_db", DEFAULT_SNR_RANGE_DB))
        alpha_range = tuple(
            syn_config.get("colored_noise_alpha_range",
                           DEFAULT_COLORED_NOISE_ALPHA_RANGE)
        )
        amp_range = tuple(
            syn_config.get("colored_noise_amplitude_fraction",
                           DEFAULT_COLORED_NOISE_AMPLITUDE_FRACTION)
        )

        eeg_windows = np.zeros(
            (n_windows, N_CHANNELS, wl), dtype=np.float64
        )
        snr_values = np.zeros(n_windows, dtype=np.float64)

        for w in range(n_windows):
            # Project through leadfield: clean EEG
            clean_eeg = project_to_eeg(source_windows[w], leadfield)

            # Sample noise parameters for this window
            snr_db = rng.uniform(snr_range[0], snr_range[1])
            alpha = rng.uniform(alpha_range[0], alpha_range[1])
            amp_frac = rng.uniform(amp_range[0], amp_range[1])

            # Add measurement noise
            noisy_eeg = add_measurement_noise(
                clean_eeg, snr_db, alpha, amp_frac, rng=rng
            )

            eeg_windows[w] = noisy_eeg
            snr_values[w] = snr_db

        return {
            "eeg": eeg_windows,
            "source_activity": source_windows,
            "epileptogenic_mask": params["epileptogenic_mask"],
            "x0_vector": params["x0_vector"],
            "snr_db": snr_values,
            "global_coupling": params["global_coupling"],
        }

    except Exception as e:
        import traceback
        logger.warning(
            f"Simulation {sim_index} failed: {e}\n{traceback.format_exc()}"
        )
        return None


def _create_hdf5_file(
    output_file: Path,
    expected_n_samples: int,
    channel_names: List[str],
    region_names: List[str],
) -> None:
    """
    Create an empty HDF5 file with resizable datasets ready for incremental writing.

    This function pre-allocates the HDF5 structure with maxshape=None (unlimited),
    allowing data to be appended incrementally without reallocating the entire file.

    Args:
        output_file: Path to create the HDF5 file.
        expected_n_samples: Approximate upper bound on number of samples.
            Used for initial allocation; datasets will grow as needed.
        channel_names: List of 19 channel name strings.
        region_names: List of 76 region name strings.

    References:
        Technical Specs Section 3.4.4 (HDF5 Storage Format)
    """
    with h5py.File(output_file, "w") as f:
        # Create resizable datasets with initial size 0, max size unlimited
        # The 'chunks' parameter enables incremental I/O and compression
        f.create_dataset(
            "eeg",
            shape=(0, N_CHANNELS, WINDOW_LENGTH_SAMPLES),
            dtype=np.float32,
            maxshape=(None, N_CHANNELS, WINDOW_LENGTH_SAMPLES),
            chunks=(1, N_CHANNELS, WINDOW_LENGTH_SAMPLES),
            compression="gzip",
        )
        f.create_dataset(
            "source_activity",
            shape=(0, N_REGIONS, WINDOW_LENGTH_SAMPLES),
            dtype=np.float32,
            maxshape=(None, N_REGIONS, WINDOW_LENGTH_SAMPLES),
            chunks=(1, N_REGIONS, WINDOW_LENGTH_SAMPLES),
            compression="gzip",
        )
        f.create_dataset(
            "epileptogenic_mask",
            shape=(0, N_REGIONS),
            dtype=np.bool_,
            maxshape=(None, N_REGIONS),
            chunks=(1, N_REGIONS),
            compression="gzip",
        )
        f.create_dataset(
            "x0_vector",
            shape=(0, N_REGIONS),
            dtype=np.float32,
            maxshape=(None, N_REGIONS),
            chunks=(1, N_REGIONS),
            compression="gzip",
        )
        f.create_dataset(
            "snr_db",
            shape=(0,),
            dtype=np.float32,
            maxshape=(None,),
            chunks=(256,),
            compression=None,
        )
        f.create_dataset(
            "global_coupling",
            shape=(0,),
            dtype=np.float32,
            maxshape=(None,),
            chunks=(256,),
            compression=None,
        )

        # Create metadata group with fixed metadata
        metadata = f.create_group("metadata")
        metadata.create_dataset(
            "channel_names",
            data=np.array(channel_names, dtype="S"),
        )
        metadata.create_dataset(
            "region_names",
            data=np.array(region_names, dtype="S"),
        )
        metadata.create_dataset(
            "sampling_rate", data=SAMPLING_RATE
        )
        metadata.create_dataset(
            "window_length_sec", data=WINDOW_LENGTH_SAMPLES / SAMPLING_RATE
        )


def _append_to_hdf5(
    output_file: Path,
    eeg: NDArray[np.float32],
    source_activity: NDArray[np.float32],
    epileptogenic_mask: NDArray[np.bool_],
    x0_vector: NDArray[np.float32],
    snr_db: NDArray[np.float32],
    global_coupling: NDArray[np.float32],
) -> int:
    """
    Append a batch of samples to an existing HDF5 file.

    This function increments the size of each dataset and writes the new data.
    It is designed for periodic/batch writing to maintain constant memory usage.

    Args:
        output_file: Path to the HDF5 file (must already exist).
        eeg: Batch of EEG data. Shape (batch_size, 19, 400).
        source_activity: Batch of source activity. Shape (batch_size, 76, 400).
        epileptogenic_mask: Batch of masks. Shape (batch_size, 76).
        x0_vector: Batch of x0 vectors. Shape (batch_size, 76).
        snr_db: Batch of SNR values. Shape (batch_size,).
        global_coupling: Batch of coupling values. Shape (batch_size,).

    Returns:
        int: Total number of samples now in the file.

    References:
        Technical Specs Section 3.4.4 (HDF5 Storage Format)
    """
    batch_size = eeg.shape[0]

    with h5py.File(output_file, "a") as f:  # Open in append mode
        # Get current dataset sizes
        current_size = f["eeg"].shape[0]

        # Resize all datasets to accommodate new data
        f["eeg"].resize(current_size + batch_size, axis=0)
        f["source_activity"].resize(current_size + batch_size, axis=0)
        f["epileptogenic_mask"].resize(current_size + batch_size, axis=0)
        f["x0_vector"].resize(current_size + batch_size, axis=0)
        f["snr_db"].resize(current_size + batch_size, axis=0)
        f["global_coupling"].resize(current_size + batch_size, axis=0)

        # Write new data to the end of each dataset
        f["eeg"][current_size : current_size + batch_size] = eeg
        f["source_activity"][current_size : current_size + batch_size] = (
            source_activity
        )
        f["epileptogenic_mask"][current_size : current_size + batch_size] = (
            epileptogenic_mask
        )
        f["x0_vector"][current_size : current_size + batch_size] = x0_vector
        f["snr_db"][current_size : current_size + batch_size] = snr_db
        f["global_coupling"][current_size : current_size + batch_size] = (
            global_coupling
        )

        new_total = f["eeg"].shape[0]

    return new_total


def generate_dataset(
    n_simulations: int,
    connectivity: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    leadfield: NDArray[np.float64],
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1,
    base_seed: int = 0,
) -> Path:
    """
    Generate a complete synthetic EEG dataset and save to HDF5 with incremental writing.

    This function implements fault-tolerant dataset generation by writing results
    to HDF5 incrementally in batches. This approach:
    - Uses constant memory (~1-2 GB) instead of accumulating in RAM
    - Provides fault tolerance: if interrupted, completed samples are preserved
    - Allows real-time monitoring of progress by reading the HDF5 file
    - Avoids the need to hold 100,000+ samples in RAM simultaneously

    The parallelization uses joblib's Parallel with the "loky" backend,
    which spawns independent processes. Each simulation is fully self-contained
    with its own random seed for reproducibility.

    Dataset sizes per specs:
      - Training: 16,000 simulations × 5 windows = 80,000 samples
      - Validation: 2,000 simulations × 5 windows = 10,000 samples
      - Test: 2,000 simulations × 5 windows = 10,000 samples

    Args:
        n_simulations: Number of TVB simulations to run. Each produces
            windows_per_simulation samples.
        connectivity: Preprocessed connectivity matrix. (76, 76) float64.
        region_centers: Region centroids. (76, 3) float64.
        region_labels: List of 76 region names.
        tract_lengths: Tract length matrix. (76, 76) float64.
        leadfield: Leadfield matrix. (19, 76) float64.
        output_path: File path for the output HDF5 file.
        config: Configuration dict from config.yaml.
        n_jobs: Number of parallel jobs. -1 = use all CPUs.
        base_seed: Base random seed. Each simulation gets seed = base_seed + index.

    Returns:
        Path: Path to the saved HDF5 file.

    References:
        Technical Specs Sections 3.4.3–3.4.4, Section 3.4.6 (Incremental HDF5 Writing)
    """
    logger.info("=" * 60)
    logger.info(f"GENERATING SYNTHETIC DATASET: {n_simulations} simulations")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Create empty HDF5 file with resizable datasets
    logger.info("Creating HDF5 file with resizable datasets...")
    _create_hdf5_file(
        output_file=output_file,
        expected_n_samples=n_simulations * WINDOWS_PER_SIM,
        channel_names=CHANNEL_NAMES,
        region_names=region_labels,
    )
    logger.info(f"✓ HDF5 file created: {output_file}")

    # Step 2: Run simulations in parallel and process results as they complete
    logger.info(f"Running {n_simulations} simulations...")

    syn_config = config.get("synthetic_data", {}) if config else {}
    batch_size = syn_config.get("hdf5_batch_size", 500)  # Write every 500 samples
    
    # Determine number of workers
    if n_jobs == -1:
        # Count available CPUs
        import os
        n_workers = os.cpu_count() or 8
    elif n_jobs > 0:
        n_workers = n_jobs
    else:
        n_workers = 1
    
    # Track statistics across all batches
    total_samples_written = 0
    total_simulations_completed = 0
    total_simulations_failed = 0
    current_batch_eeg = []
    current_batch_source = []
    current_batch_mask = []
    current_batch_x0 = []
    current_batch_snr = []
    current_batch_coupling = []

    # Helper function to create kwargs for a simulation
    def create_sim_kwargs(sim_index: int):
        return dict(
            sim_index=sim_index,
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            config=config,
            seed=base_seed + sim_index,
        )

    # Use ProcessPoolExecutor to process results as they complete (not waiting for all)
    logger.info(f"Using {n_workers} worker processes")
    logger.info("Processing results and writing to HDF5 incrementally...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(generate_one_simulation, **create_sim_kwargs(i)): i
            for i in range(n_simulations)
        }
        
        # Process results as they complete (not waiting for all to finish first)
        for future in as_completed(futures):
            result = future.result()
            
            if result is None:
                total_simulations_failed += 1
                continue

            total_simulations_completed += 1
            n_w = result["eeg"].shape[0]

            # Accumulate windows from this simulation into batch
            for w in range(n_w):
                current_batch_eeg.append(result["eeg"][w])
                current_batch_source.append(result["source_activity"][w])
                current_batch_mask.append(result["epileptogenic_mask"])
                current_batch_x0.append(result["x0_vector"])
                current_batch_snr.append(result["snr_db"][w])
                current_batch_coupling.append(result["global_coupling"])

                # Write batch to HDF5 when we reach batch_size
                if len(current_batch_eeg) >= batch_size:
                    batch_eeg = np.array(current_batch_eeg, dtype=np.float32)
                    batch_source = np.array(current_batch_source, dtype=np.float32)
                    batch_mask = np.array(current_batch_mask, dtype=np.bool_)
                    batch_x0 = np.array(current_batch_x0, dtype=np.float32)
                    batch_snr = np.array(current_batch_snr, dtype=np.float32)
                    batch_coupling = np.array(current_batch_coupling, dtype=np.float32)

                    total_samples_written = _append_to_hdf5(
                        output_file=output_file,
                        eeg=batch_eeg,
                        source_activity=batch_source,
                        epileptogenic_mask=batch_mask,
                        x0_vector=batch_x0,
                        snr_db=batch_snr,
                        global_coupling=batch_coupling,
                    )

                    # Log progress
                    logger.info(
                        f"Progress: {total_samples_written} samples written "
                        f"({total_simulations_completed}/{n_simulations} simulations completed, "
                        f"{total_simulations_failed} failed)"
                    )

                    # Reset batch accumulators
                    current_batch_eeg = []
                    current_batch_source = []
                    current_batch_mask = []
                    current_batch_x0 = []
                    current_batch_snr = []
                    current_batch_coupling = []

    # Write any remaining samples in the final incomplete batch
    if len(current_batch_eeg) > 0:
        batch_eeg = np.array(current_batch_eeg, dtype=np.float32)
        batch_source = np.array(current_batch_source, dtype=np.float32)
        batch_mask = np.array(current_batch_mask, dtype=np.bool_)
        batch_x0 = np.array(current_batch_x0, dtype=np.float32)
        batch_snr = np.array(current_batch_snr, dtype=np.float32)
        batch_coupling = np.array(current_batch_coupling, dtype=np.float32)

        total_samples_written = _append_to_hdf5(
            output_file=output_file,
            eeg=batch_eeg,
            source_activity=batch_source,
            epileptogenic_mask=batch_mask,
            x0_vector=batch_x0,
            snr_db=batch_snr,
            global_coupling=batch_coupling,
        )

    logger.info("=" * 60)
    logger.info(f"DATASET GENERATION COMPLETE")
    logger.info(f"Total samples written: {total_samples_written}")
    logger.info(f"Simulations completed: {total_simulations_completed}/{n_simulations}")
    logger.info(f"Simulations failed: {total_simulations_failed}")
    logger.info(f"Saved to: {output_file}")
    logger.info("=" * 60)

    return output_file





def generate_all_splits(
    connectivity: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    leadfield: NDArray[np.float64],
    config: Dict[str, Any],
) -> Dict[str, Path]:
    """
    Generate training, validation, and test datasets in one call.

    This convenience function generates all three dataset splits using the
    simulation counts specified in config.yaml. Each split gets a different
    base seed to ensure no overlap in random parameter space.

    Args:
        connectivity: Preprocessed connectivity. (76, 76) float64.
        region_centers: Region centroids. (76, 3) float64.
        region_labels: List of 76 region names.
        tract_lengths: Tract lengths. (76, 76) float64.
        leadfield: Leadfield matrix. (19, 76) float64.
        config: Full configuration dict from config.yaml.

    Returns:
        Dict[str, Path]: Mapping from split names ("train", "val", "test")
            to their HDF5 file paths.

    References:
        Technical Specs Section 3.4.3 (dataset size table)
    """
    syn_config = config.get("synthetic_data", {})
    output_dir = syn_config.get("output_dir", "data/synthetic/")
    n_jobs = syn_config.get("n_jobs", -1)

    splits = {
        "train": {
            "n_sims": syn_config.get("n_simulations_train", 16000),
            "file": "train_dataset.h5",
            "base_seed": 0,
        },
        "val": {
            "n_sims": syn_config.get("n_simulations_val", 2000),
            "file": "val_dataset.h5",
            "base_seed": 100000,  # Offset to avoid seed overlap
        },
        "test": {
            "n_sims": syn_config.get("n_simulations_test", 2000),
            "file": "test_dataset.h5",
            "base_seed": 200000,
        },
    }

    saved_paths = {}

    for split_name, split_config in splits.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {split_name.upper()} split...")
        logger.info(f"{'='*60}")

        output_path = str(Path(output_dir) / split_config["file"])

        saved_path = generate_dataset(
            n_simulations=split_config["n_sims"],
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            output_path=output_path,
            config=config,
            n_jobs=n_jobs,
            base_seed=split_config["base_seed"],
        )

        saved_paths[split_name] = saved_path

    return saved_paths
