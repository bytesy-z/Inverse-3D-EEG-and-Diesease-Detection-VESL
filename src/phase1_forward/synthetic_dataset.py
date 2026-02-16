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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import h5py
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

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
        logger.warning(
            f"Simulation {sim_index} failed: {e}. Skipping."
        )
        return None


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
    Generate a complete synthetic EEG dataset and save to HDF5.

    This is the main entry point for dataset generation. It runs n_simulations
    TVB simulations in parallel, collects all windows, and writes them to a
    single HDF5 file following the schema in Section 3.4.4.

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
        Technical Specs Sections 3.4.3–3.4.4
    """
    logger.info("=" * 60)
    logger.info(f"GENERATING SYNTHETIC DATASET: {n_simulations} simulations")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Run all simulations in parallel using joblib
    logger.info(f"Running {n_simulations} simulations (n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(generate_one_simulation)(
            sim_index=i,
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            config=config,
            seed=base_seed + i,
        )
        for i in range(n_simulations)
    )

    # Filter out failed simulations (None results)
    successful_results = [r for r in results if r is not None]
    n_failed = n_simulations - len(successful_results)
    if n_failed > 0:
        logger.warning(
            f"{n_failed}/{n_simulations} simulations failed. "
            f"Proceeding with {len(successful_results)} successful ones."
        )

    if len(successful_results) == 0:
        raise RuntimeError(
            "All simulations failed. Check TVB installation and parameters."
        )

    # Collect all windows from all simulations into flat arrays
    logger.info("Collecting results into arrays...")
    all_eeg = []
    all_source = []
    all_mask = []
    all_x0 = []
    all_snr = []
    all_coupling = []

    syn_config = config.get("synthetic_data", {}) if config else {}
    n_windows = syn_config.get("windows_per_simulation", WINDOWS_PER_SIM)

    for result in successful_results:
        n_w = result["eeg"].shape[0]
        for w in range(n_w):
            all_eeg.append(result["eeg"][w])
            all_source.append(result["source_activity"][w])
            all_mask.append(result["epileptogenic_mask"])
            all_x0.append(result["x0_vector"])
            all_snr.append(result["snr_db"][w])
            all_coupling.append(result["global_coupling"])

    # Convert lists to numpy arrays with the dtypes specified in Section 3.4.4
    eeg_array = np.array(all_eeg, dtype=np.float32)
    source_array = np.array(all_source, dtype=np.float32)
    mask_array = np.array(all_mask, dtype=np.bool_)
    x0_array = np.array(all_x0, dtype=np.float32)
    snr_array = np.array(all_snr, dtype=np.float32)
    coupling_array = np.array(all_coupling, dtype=np.float32)

    n_total = eeg_array.shape[0]
    logger.info(
        f"Collected {n_total} total samples from "
        f"{len(successful_results)} simulations"
    )

    # Save to HDF5 with the exact schema from Section 3.4.4
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _save_to_hdf5(
        output_file=output_file,
        eeg=eeg_array,
        source_activity=source_array,
        epileptogenic_mask=mask_array,
        x0_vector=x0_array,
        snr_db=snr_array,
        global_coupling=coupling_array,
        channel_names=CHANNEL_NAMES,
        region_names=region_labels,
    )

    logger.info("=" * 60)
    logger.info(f"DATASET GENERATION COMPLETE: {n_total} samples")
    logger.info(f"Saved to: {output_file}")
    logger.info("=" * 60)

    return output_file


def _save_to_hdf5(
    output_file: Path,
    eeg: NDArray[np.float32],
    source_activity: NDArray[np.float32],
    epileptogenic_mask: NDArray[np.bool_],
    x0_vector: NDArray[np.float32],
    snr_db: NDArray[np.float32],
    global_coupling: NDArray[np.float32],
    channel_names: List[str],
    region_names: List[str],
) -> None:
    """
    Save the synthetic dataset to HDF5 with the standardized schema.

    The HDF5 file structure exactly matches Section 3.4.4 of the technical
    specifications. This format is the primary data interface between
    Phase 1 (forward modeling) and Phase 2 (network training).

    All main datasets use gzip compression to reduce disk usage. The
    metadata group stores string arrays for channel and region names,
    plus scalar metadata (sampling rate, window length).

    Args:
        output_file: Path to the output HDF5 file.
        eeg: Noisy EEG data. Shape (N, 19, 400), dtype float32.
        source_activity: Source time series. Shape (N, 76, 400), dtype float32.
        epileptogenic_mask: Binary mask. Shape (N, 76), dtype bool.
        x0_vector: Excitability parameters. Shape (N, 76), dtype float32.
        snr_db: SNR values per sample. Shape (N,), dtype float32.
        global_coupling: Coupling strength per sample. Shape (N,), dtype float32.
        channel_names: List of 19 channel name strings.
        region_names: List of 76 region name strings.

    References:
        Technical Specs Section 3.4.4 (HDF5 Storage Format)
    """
    n_samples = eeg.shape[0]

    logger.info(
        f"Saving {n_samples} samples to HDF5: {output_file}"
    )

    with h5py.File(output_file, "w") as f:
        # Main data arrays — exact names from specs
        f.create_dataset(
            "eeg", data=eeg, dtype=np.float32, compression="gzip"
        )
        f.create_dataset(
            "source_activity", data=source_activity,
            dtype=np.float32, compression="gzip"
        )
        f.create_dataset(
            "epileptogenic_mask", data=epileptogenic_mask,
            dtype=np.bool_, compression="gzip"
        )
        f.create_dataset(
            "x0_vector", data=x0_vector,
            dtype=np.float32, compression="gzip"
        )
        f.create_dataset(
            "snr_db", data=snr_db, dtype=np.float32
        )
        f.create_dataset(
            "global_coupling", data=global_coupling, dtype=np.float32
        )

        # Metadata group — string arrays and scalars
        metadata = f.create_group("metadata")
        metadata.create_dataset(
            "channel_names",
            data=np.array(channel_names, dtype="S")
        )
        metadata.create_dataset(
            "region_names",
            data=np.array(region_names, dtype="S")
        )
        metadata.create_dataset(
            "sampling_rate", data=SAMPLING_RATE
        )
        metadata.create_dataset(
            "window_length_sec", data=WINDOW_LENGTH_SAMPLES / SAMPLING_RATE
        )

    logger.info(
        f"HDF5 saved successfully. File size: "
        f"{output_file.stat().st_size / (1024**2):.1f} MB"
    )


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
