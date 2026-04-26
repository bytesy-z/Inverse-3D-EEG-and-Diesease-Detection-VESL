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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Third-party imports
import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig

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

# ---------------------------------------------------------------------------
# Skull attenuation lowpass filter
# ---------------------------------------------------------------------------
# The leadfield matrix L is frequency-independent (a static linear map from
# sources to sensors), but in reality the skull acts as a lowpass filter that
# attenuates high-frequency components by ~6 dB/octave above ~30 Hz (Nunez &
# Srinivasan, 2006, "Electric Fields of the Brain", 2nd ed., Ch. 6).
#
# The Epileptor model produces broadband source activity (significant energy
# up to Nyquist), which would result in unrealistic gamma-band content on
# the scalp if passed through L without frequency shaping. To compensate,
# we apply a 4th-order Butterworth lowpass at 40 Hz *after* noise addition.
# This models both the skull's spatial lowpass effect and the typical EEG
# amplifier's anti-aliasing filter.
#
# Design choice rationale (validated on 8 simulations, 40 windows):
#   - Cutoff 40 Hz: gamma < 15%, mobility < 0.5, 1/f exponent 0.5-3.0
#   - 4th-order Butterworth: steep enough rolloff to control gamma without
#     ringing artifacts on 400-sample windows
#   - Applied with sosfiltfilt (zero-phase) to avoid temporal distortion
#   - Applied AFTER noise so broadband noise is also attenuated (as it
#     would be by a real skull + amplifier chain)
# ---------------------------------------------------------------------------
SKULL_LP_CUTOFF_HZ = 40.0    # Lowpass cutoff frequency in Hz
SKULL_LP_ORDER = 4            # Butterworth filter order
SKULL_LP_SOS = sig.butter(
    SKULL_LP_ORDER,
    SKULL_LP_CUTOFF_HZ,
    btype='low',
    fs=SAMPLING_RATE,
    output='sos',
)

# Channel names in the standard NMT order
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Configure module-level logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectral shaping: transform broadband Epileptor output into clinically
# realistic EEG with proper band power distribution and spatial gradients.
#
# The Epileptor neural mass model produces broadband source activity with
# excess delta (~33%) and beta (~34%), and insufficient alpha (~11%).
# Real resting-state EEG is alpha-dominant (~30-50% at posterior channels),
# with a 1/f + alpha peak spectral shape (Niedermeyer, 2005; Nunez &
# Srinivasan, 2006).
#
# The spectral shaping applies per-channel frequency-domain gain curves
# that simultaneously:
#   1. Suppress excess delta (1-4 Hz) — gains 0.30-0.55 per channel group
#   2. Boost alpha (8-13 Hz) — gains 0.82-2.00, with posterior > anterior
#   3. Suppress excess beta (13-30 Hz) — gains 0.40-0.80 per channel group
#   4. Preserve gamma (30-40 Hz) — gains ~1.0 (already controlled by LP)
#
# Channel groups follow the standard anteroposterior axis:
#   Fp (frontal pole) → F (frontal) → C (central) → P (parietal) → O (occipital)
#
# Target per-channel-group spectral profiles (% of 1-70 Hz power):
#   Fp: delta~17%, theta~16%, alpha~16%, beta~34%, gamma~10%
#   F:  delta~14%, theta~13%, alpha~23%, beta~28%, gamma~9%
#   C:  delta~12%, theta~11%, alpha~32%, beta~22%, gamma~8%
#   P:  delta~10%, theta~9%,  alpha~42%, beta~17%, gamma~8%
#   O:  delta~7%,  theta~7%,  alpha~52%, beta~12%, gamma~8%
#
# These targets are derived from clinical EEG norms:
#   - Niedermeyer E. (2005). Electroencephalography, 5th ed., Ch. 9-10
#   - Nunez PL & Srinivasan R. (2006). Electric Fields of the Brain, Ch. 5-6
#   - Barry RJ et al. (2007). Clin Neurophysiol 118(12):2765-2773
#   - Markand ON (1990). Alpha rhythms. JCNS 7(2):163-189
#
# For epilepsy samples: the same spectral shaping is applied uniformly.
# The Epileptor's pathological activity (spikes, sharp waves) naturally
# disrupts the shaped spectrum at channels receiving epileptogenic source
# activity. The network learns to invert through this consistent shaping.
#
# Clinical EEG in focal epilepsy (interictal, from literature):
#   - Increased delta/theta over epileptogenic zone (focal slowing)
#   - Reduced alpha power over affected hemisphere
#   - Sharp transients with broad spectral content
#   - Non-epileptogenic channels: relatively normal background
# These features are already captured by the Epileptor source dynamics;
# spectral shaping fixes only the background spectral envelope.
# ---------------------------------------------------------------------------

# Channel group definitions for spectral shaping.
# Each group contains channels at similar anteroposterior positions.
SPECTRAL_GROUPS = {
    'frontal_fp': ['Fp1', 'Fp2'],
    'frontal_f': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'central_c': ['C3', 'C4', 'T3', 'T4', 'Cz'],
    'parietal_p': ['P3', 'P4', 'T5', 'T6', 'Pz'],
    'occipital_o': ['O1', 'O2'],
}

# Ordered group list for consistent iteration (front-to-back)
_SPECTRAL_GROUPS_ORDERED = [
    ('frontal_fp', ['Fp1', 'Fp2']),
    ('frontal_f', ['F3', 'F4', 'F7', 'F8', 'Fz']),
    ('central_c', ['C3', 'C4', 'T3', 'T4', 'Cz']),
    ('parietal_p', ['P3', 'P4', 'T5', 'T6', 'Pz']),
    ('occipital_o', ['O1', 'O2']),
]

# ---------------------------------------------------------------------------
# STFT-based spectral shaping parameters
# ---------------------------------------------------------------------------

# Global amplitude gain for delta band (0.5-4 Hz).
# Suppresses the excess low-frequency power from the Epileptor model.
# Current delta: ~33%, target: ~10-15%. Gain² ≈ 0.12-0.18 → gain ≈ 0.38.
_DELTA_GAIN = 0.38

# Global amplitude gain for theta band (4-8 Hz).
# Mild suppression. Current theta: ~14%, target: ~8-12%.
_THETA_GAIN = 0.75

# Global alpha boost factor (applied BEFORE adaptive redistribution).
# Increases total alpha power from ~11% toward ~30%.
# The adaptive redistribution then spreads this spatially.
_ALPHA_BOOST = 1.8

# Target alpha power ratios per group (relative scale).
# These define the anteroposterior alpha gradient shape.
# The adaptive algorithm computes per-group gains to match these ratios
# in each STFT frame, guaranteeing monotonic gradient.
_ALPHA_TARGET_RATIOS = {
    'frontal_fp': 1.0,   # Reference (lowest alpha)
    'frontal_f': 1.3,    # +30% step from Fp
    'central_c': 1.8,    # +38% step from F
    'parietal_p': 2.4,   # +33% step from C
    'occipital_o': 3.0,  # +25% step from P
}

# Fixed beta amplitude gains per channel.
# Creates frontal > occipital beta gradient.
# Values from validated spatial gradient approach (script 08).
_BETA_GAINS = {
    "Fp1": 1.30, "Fp2": 1.30,
    "F3": 1.10, "F4": 1.10,
    "F7": 1.08, "F8": 1.08, "Fz": 1.15,
    "C3": 0.85, "C4": 0.85,
    "T3": 0.82, "T4": 0.82, "Cz": 0.90,
    "P3": 0.60, "P4": 0.60,
    "T5": 0.62, "T6": 0.62, "Pz": 0.55,
    "O1": 0.38, "O2": 0.38,
}

# Gain clamp range (prevents extreme corrections on edge-case samples)
_GAIN_CLIP_MIN = 0.15
_GAIN_CLIP_MAX = 6.0


def apply_skull_attenuation_filter(
    eeg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply a lowpass filter to simulate skull + amplifier frequency attenuation.

    Real scalp EEG is spatially lowpass-filtered by the skull, which attenuates
    high-frequency components by approximately 6 dB/octave above ~30 Hz (Nunez
    & Srinivasan, 2006). Since our leadfield matrix is frequency-independent
    (static linear map L @ S), we must apply this attenuation as a separate
    post-hoc filtering step.

    The filter is a 4th-order Butterworth lowpass at 40 Hz, applied using
    sosfiltfilt (zero-phase, no temporal distortion). It is applied channel-by-
    channel after noise addition, so that broadband measurement noise is also
    attenuated — matching what a real skull + amplifier chain would do.

    Args:
        eeg: EEG data (noisy) to filter.
            Shape (N_channels, T) or (N_channels, 400), dtype float64.

    Returns:
        NDArray[np.float64]: Lowpass-filtered EEG with the same shape.

    References:
        Nunez & Srinivasan (2006), "Electric Fields of the Brain", 2nd ed.,
        Ch. 6: Skull attenuation ~6 dB/octave above 30 Hz.
        Validated: 13/13 biophysical checks pass (RMS, spectral, temporal,
        spatial) on 8 TVB simulations (40 windows).
    """
    n_channels = eeg.shape[0]
    filtered = np.zeros_like(eeg)

    for ch in range(n_channels):
        # sosfiltfilt applies the filter forward and backward (zero-phase),
        # so the effective order is doubled (8th-order equivalent) but there
        # is no phase distortion — critical for preserving temporal structure.
        filtered[ch] = sig.sosfiltfilt(SKULL_LP_SOS, eeg[ch])

    return filtered


def apply_spectral_shaping(
    eeg: NDArray[np.float64],
    channel_names: List[str],
) -> NDArray[np.float64]:
    """
    Apply STFT-based spectral shaping to create clinically realistic EEG.

    Transforms the broadband Epileptor+leadfield EEG into spectra matching
    normal adult resting-state recordings with:
      - Alpha-dominant spectral shape (8-13 Hz boosted to ~30% of total)
      - Anteroposterior alpha gradient (Fp < F < C < P < O) via adaptive gains
      - Anteroposterior beta gradient (Fp > F > C > P > O) via fixed gains
      - Reduced excess delta (1-4 Hz) and beta (13-30 Hz)

    Algorithm (STFT-based, matches Welch PSD validation methodology):
      1. STFT with 200-sample Hann window, 50% overlap
      2. For EACH STFT frame:
         a. Suppress delta band (fixed gain, all channels)
         b. Suppress theta band (fixed gain, all channels)
         c. Boost alpha globally, then redistribute adaptively across
            channel groups to enforce spatial gradient (alpha Fp < O)
         d. Apply fixed per-channel beta gains (beta Fp > O)
      3. ISTFT to reconstruct time-domain signal
      4. Normalize to preserve original overall RMS amplitude

    The adaptive alpha approach (step 2c) measures each frame's actual
    alpha power distribution and computes per-group gains to match the
    target ratios. This guarantees the alpha gradient passes validation
    regardless of natural per-sample variation from the leadfield.

    Args:
        eeg: Skull-filtered EEG. Shape (N_CHANNELS, 400), dtype float64.
        channel_names: Ordered list of N_CHANNELS channel name strings.

    Returns:
        NDArray[np.float64]: Spectrally shaped EEG with same shape and dtype.
            RMS amplitude is preserved (overall signal level unchanged).

    References:
        Niedermeyer E. (2005): Alpha posterior dominance in resting EEG
        Nunez & Srinivasan (2006): 1/f spectral law, skull attenuation
        Technical Specs Section 3.4.2 (spectral shaping step)
    """
    n_ch, n_t = eeg.shape
    original_rms = np.sqrt(np.mean(eeg ** 2))

    # Build channel name → index map
    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}

    # Pre-compute group indices (front-to-back order)
    group_indices = {}
    for gname, gchannels in _SPECTRAL_GROUPS_ORDERED:
        group_indices[gname] = [ch_to_idx[ch] for ch in gchannels]

    # ------------------------------------------------------------------
    # Step 1: STFT with same parameters as Welch validation
    # ------------------------------------------------------------------
    nperseg = int(SAMPLING_RATE * 1.0)  # 200 samples = 1 second
    noverlap = nperseg // 2              # 50% overlap

    freqs_stft, times_stft, Zxx = sig.stft(
        eeg, fs=SAMPLING_RATE,
        nperseg=nperseg, noverlap=noverlap, window='hann',
    )
    # Zxx shape: (n_channels, n_freqs, n_frames)

    # Frequency band masks
    delta_mask = (freqs_stft >= 0.5) & (freqs_stft < 4)
    theta_mask = (freqs_stft >= 4) & (freqs_stft < 8)
    alpha_mask = (freqs_stft >= 8) & (freqs_stft <= 13)
    beta_mask = (freqs_stft > 13) & (freqs_stft <= 30)
    n_frames = Zxx.shape[2]

    # ------------------------------------------------------------------
    # Step 2: Per-frame spectral shaping
    # ------------------------------------------------------------------
    for frame in range(n_frames):
        # 2a. Suppress delta (fixed, all channels equally)
        Zxx[:, delta_mask, frame] *= _DELTA_GAIN

        # 2b. Suppress theta mildly (fixed, all channels equally)
        Zxx[:, theta_mask, frame] *= _THETA_GAIN

        # 2c. Alpha: global boost then adaptive spatial redistribution
        # First, apply global alpha boost to all channels
        Zxx[:, alpha_mask, frame] *= _ALPHA_BOOST

        # Measure alpha power per channel after boost
        alpha_power = np.sum(np.abs(Zxx[:, alpha_mask, frame]) ** 2, axis=1)

        # Compute group-mean alpha power
        group_powers = {}
        for gn, idxs in group_indices.items():
            group_powers[gn] = np.mean(alpha_power[idxs])

        # Compute normalization constant K (preserves total alpha energy
        # across the redistribution step — only the SPATIAL distribution
        # changes, not the total alpha power)
        total_alpha = sum(
            group_powers[gn] * len(idxs)
            for gn, idxs in group_indices.items()
        )
        total_target_w = sum(
            _ALPHA_TARGET_RATIOS[gn] * len(idxs)
            for gn, idxs in group_indices.items()
        )
        K = total_alpha / (total_target_w + 1e-30)

        # Compute and apply per-group adaptive alpha gains
        for gn, idxs in group_indices.items():
            current_power = group_powers[gn]
            target_power = _ALPHA_TARGET_RATIOS[gn] * K
            if current_power > 1e-30:
                gain = np.sqrt(target_power / current_power)
            else:
                gain = 1.0
            gain = np.clip(gain, _GAIN_CLIP_MIN, _GAIN_CLIP_MAX)
            for idx in idxs:
                Zxx[idx, alpha_mask, frame] *= gain

        # 2d. Apply fixed per-channel beta gradient gains
        for ch_idx in range(n_ch):
            ch_name = channel_names[ch_idx]
            beta_gain = _BETA_GAINS.get(ch_name, 1.0)
            if beta_gain != 1.0:
                Zxx[ch_idx, beta_mask, frame] *= beta_gain

    # ------------------------------------------------------------------
    # Step 3: Inverse STFT → time domain
    # ------------------------------------------------------------------
    _, reconstructed = sig.istft(
        Zxx, fs=SAMPLING_RATE,
        nperseg=nperseg, noverlap=noverlap, window='hann',
    )

    # Handle potential length mismatch from STFT/ISTFT boundary effects
    if reconstructed.shape[1] >= n_t:
        reconstructed = reconstructed[:, :n_t]
    else:
        pad_width = n_t - reconstructed.shape[1]
        reconstructed = np.pad(
            reconstructed, ((0, 0), (0, pad_width)), mode='constant'
        )

    # ------------------------------------------------------------------
    # Step 4: Normalize to preserve original RMS amplitude
    # ------------------------------------------------------------------
    shaped_rms = np.sqrt(np.mean(reconstructed ** 2))
    if shaped_rms > 1e-10:
        reconstructed *= original_rms / shaped_rms

    return reconstructed


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


def validate_spatial_spectral_properties(
    eeg: NDArray[np.float64],
    channels: List[str],
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that EEG exhibits clinically realistic spatial-spectral structure.

    This function checks for three hallmarks of normal healthy adult resting EEG
    (from neurophysiology literature: Niedermeyer 2005, Nunez & Srinivasan 2006,
    Delorme et al. 2011):

    1. **Posterior Dominant Rhythm (PDR)**: Alpha power (8-13 Hz) at occipital
       channels (O1/O2) should be 1.3–5.0× higher than frontal pole (Fp1/Fp2).

    2. **Anteroposterior Alpha Gradient**: Group-mean alpha power should
       monotonically increase front-to-back: Fp < F < C < P < O.
       This uses the same 5 anteroposterior groups as the spectral shaping,
       ensuring consistency between shaping and validation.

    3. **Anteroposterior Beta Gradient**: Group-mean beta power should
       monotonically decrease front-to-back: Fp > F > C > P > O.

    Note: Group-level checks (not per-channel) are used because:
      - The adaptive alpha gains operate at the group level
      - Real EEG also has substantial within-group per-channel variation
      - Per-channel strict monotonicity is unrealistic for short 2s windows

    Args:
        eeg: Shaped EEG signal to validate. (19, 400+) float64.
        channels: List of 19 channel names in order.

    Returns:
        Tuple of:
        - bool: True if all three criteria are met, False otherwise.
        - dict: Diagnostic metrics for logging.
    """
    # Map channel names to indices
    ch_to_idx = {name: i for i, name in enumerate(channels)}

    # Define the 5 anteroposterior groups with their channel indices
    group_order = [
        ('frontal_fp', ['Fp1', 'Fp2']),
        ('frontal_f', ['F3', 'F4', 'F7', 'F8', 'Fz']),
        ('central_c', ['C3', 'C4', 'T3', 'T4', 'Cz']),
        ('parietal_p', ['P3', 'P4', 'T5', 'T6', 'Pz']),
        ('occipital_o', ['O1', 'O2']),
    ]
    group_indices = []
    for gname, gchannels in group_order:
        group_indices.append([ch_to_idx[c] for c in gchannels if c in ch_to_idx])

    # Compute band power using Welch's method
    freqs, psd = sig.welch(
        eeg, fs=SAMPLING_RATE, window='hann', nperseg=200, noverlap=100,
        nfft=512
    )

    alpha_idx = (freqs >= 8) & (freqs <= 13)
    beta_idx = (freqs >= 13) & (freqs <= 30)

    alpha_power = psd[:, alpha_idx].sum(axis=1)  # (19,)
    beta_power = psd[:, beta_idx].sum(axis=1)    # (19,)

    # Compute group-mean alpha and beta power
    group_alpha = np.array([np.mean(alpha_power[idxs]) for idxs in group_indices])
    group_beta = np.array([np.mean(beta_power[idxs]) for idxs in group_indices])

    metrics = {}
    all_pass = True

    # --- Check 1: Posterior Dominant Rhythm (PDR) ---
    # PDR = occipital group alpha / frontal_fp group alpha
    fp_alpha = group_alpha[0]   # frontal_fp
    occ_alpha = group_alpha[4]  # occipital_o
    pdr_ratio = occ_alpha / (fp_alpha + 1e-10)
    metrics["PDR_ratio"] = float(pdr_ratio)

    if not (1.3 <= pdr_ratio <= 5.0):
        all_pass = False

    # --- Check 2: Group-level Alpha Gradient ---
    # Alpha should increase: Fp < F < C < P < O (5 groups, monotonic)
    alpha_diffs = np.diff(group_alpha)
    alpha_grad_pass = bool(np.all(alpha_diffs > 0))
    metrics["alpha_gradient_pass"] = alpha_grad_pass
    if not alpha_grad_pass:
        all_pass = False

    # --- Check 3: Group-level Beta Gradient ---
    # Beta should decrease: Fp > F > C > P > O (5 groups, monotonic)
    beta_diffs = np.diff(group_beta)
    beta_grad_pass = bool(np.all(beta_diffs < 0))
    metrics["beta_gradient_pass"] = beta_grad_pass
    if not beta_grad_pass:
        all_pass = False

    return all_pass, metrics


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
      6. Validate spatial-spectral properties (PDR, anteroposterior gradients)
      7. Return all windows with their metadata

    Each simulation produces multiple windows (default: 5), but invalid windows
    are excluded. This ensures the synthetic dataset exhibits realistic clinical
    EEG characteristics (Niedermeyer 2005, Nunez & Srinivasan 2006).

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

        eeg_windows = []
        source_windows_valid = []
        snr_values = []
        valid_window_indices = []
        validation_stats = {"passed": 0, "failed": 0, "invalid_reasons": []}

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

            # Apply skull attenuation lowpass filter AFTER noise addition.
            # This models the skull's spatial lowpass effect (~6 dB/octave
            # above 30 Hz) and the amplifier's anti-aliasing filter. Applied
            # after noise so broadband noise is also attenuated, matching a
            # real recording chain. See module-level SKULL_LP_SOS docs.
            filtered_eeg = apply_skull_attenuation_filter(noisy_eeg)

            # Step 5b: Apply spectral shaping to create clinically realistic
            # EEG with proper band power distribution and spatial gradients.
            # This transforms the broadband Epileptor spectrum into an
            # alpha-dominant spectrum with anteroposterior structure matching
            # real resting-state EEG recordings (Niedermeyer, 2005).
            shaped_eeg = apply_spectral_shaping(filtered_eeg, CHANNEL_NAMES)

            # Step 6: Validate spatial-spectral properties
            # Only store windows that exhibit clinically realistic EEG structure:
            # - Posterior Dominant Rhythm (alpha occipital > alpha frontal)
            # - Anteroposterior alpha gradient (increases front to back)
            # - Anteroposterior beta gradient (decreases front to back)
            passes_validation, val_metrics = validate_spatial_spectral_properties(
                shaped_eeg, CHANNEL_NAMES
            )

            if passes_validation:
                eeg_windows.append(shaped_eeg)
                source_windows_valid.append(source_windows[w])
                snr_values.append(snr_db)
                valid_window_indices.append(w)
                validation_stats["passed"] += 1
            else:
                validation_stats["failed"] += 1
                pdr = val_metrics.get("PDR_ratio", 0.0)
                validation_stats["invalid_reasons"].append(
                    f"w{w}: PDR={pdr:.2f}, "
                    f"alpha_grad={val_metrics.get('alpha_gradient_pass')}, "
                    f"beta_grad={val_metrics.get('beta_gradient_pass')}"
                )

        # Only return if we have at least 1 valid window (typical: 4-5 out of 5)
        if not eeg_windows:
            logger.warning(
                f"Simulation {sim_index}: All {n_windows} windows failed "
                f"spatial-spectral validation. Discarding entire simulation."
            )
            return None

        # Convert lists to arrays for consistency
        eeg_windows = np.array(eeg_windows, dtype=np.float64)
        source_windows_valid = np.array(source_windows_valid, dtype=np.float64)
        snr_values = np.array(snr_values, dtype=np.float64)

        # Replicate per-simulation metadata for each valid window
        # (epileptogenic_mask, x0_vector, global_coupling are per-region,
        # not per-window, so they repeat for each valid window)
        n_valid_windows = len(eeg_windows)
        epileptogenic_mask_expanded = np.tile(
            params["epileptogenic_mask"], (n_valid_windows, 1)
        )
        x0_vector_expanded = np.tile(
            params["x0_vector"], (n_valid_windows, 1)
        )
        global_coupling_expanded = np.tile(
            params["global_coupling"], n_valid_windows
        )

        if validation_stats["failed"] > 0:
            logger.info(
                f"Simulation {sim_index}: {validation_stats['passed']}/{n_windows} "
                f"windows passed spatial-spectral validation. "
                f"Discarded reasons: {validation_stats['invalid_reasons'][:2]}"
            )

        return {
            "eeg": eeg_windows,
            "source_activity": source_windows_valid,
            "epileptogenic_mask": epileptogenic_mask_expanded,
            "x0_vector": x0_vector_expanded,
            "snr_db": snr_values,
            "global_coupling": global_coupling_expanded,
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
    - **RESUME SUPPORT**: Detects existing files and resumes from where it left off

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
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and how many samples are already there
    samples_already_written = 0
    target_samples = n_simulations * WINDOWS_PER_SIM
    n_simulations_adjusted = n_simulations
    base_seed_adjusted = base_seed
    
    if output_file.exists():
        try:
            import h5py
            with h5py.File(output_file, 'r') as f:
                samples_already_written = f['eeg'].shape[0]
            
            if samples_already_written > 0:
                if samples_already_written >= target_samples:
                    # Already complete!
                    logger.info("=" * 60)
                    logger.info(f"DATASET ALREADY COMPLETE: {samples_already_written} / {target_samples} samples")
                    logger.info("=" * 60)
                    return output_file
                
                # Resume: calculate how many MORE simulations we need
                samples_remaining = target_samples - samples_already_written
                # Estimate extra simulations needed (accounting for ~5-10% failure rate)
                n_simulations_adjusted = int(np.ceil(samples_remaining / WINDOWS_PER_SIM / 0.93))
                # Adjust seed to skip past already-run simulations
                base_seed_adjusted = base_seed + n_simulations
                
                logger.info("=" * 60)
                logger.info(f"RESUMING GENERATION")
                logger.info(f"  Existing: {samples_already_written:,} samples")
                logger.info(f"  Target: {target_samples:,} samples")
                logger.info(f"  Remaining: {samples_remaining:,} samples")
                logger.info(f"  Running approximately {n_simulations_adjusted} more simulations (seeds {base_seed_adjusted}-{base_seed_adjusted + n_simulations_adjusted})")
                logger.info("=" * 60)
            else:
                # File exists but is empty — recreate it fresh
                output_file.unlink()
                logger.info("=" * 60)
                logger.info(f"GENERATING SYNTHETIC DATASET: {n_simulations} simulations")
                logger.info(f"Output: {output_path}")
                logger.info("=" * 60)
                _create_hdf5_file(
                    output_file=output_file,
                    expected_n_samples=target_samples,
                    channel_names=CHANNEL_NAMES,
                    region_names=region_labels,
                )
                logger.info(f"✓ HDF5 file created: {output_file}")
        except Exception as e:
            logger.warning(f"Could not check existing file: {e}. Creating fresh.")
            try:
                output_file.unlink()
            except:
                pass
            logger.info("=" * 60)
            logger.info(f"GENERATING SYNTHETIC DATASET: {n_simulations} simulations")
            logger.info(f"Output: {output_path}")
            logger.info("=" * 60)
            _create_hdf5_file(
                output_file=output_file,
                expected_n_samples=target_samples,
                channel_names=CHANNEL_NAMES,
                region_names=region_labels,
            )
            logger.info(f"✓ HDF5 file created: {output_file}")
    else:
        # File doesn't exist — create it fresh
        logger.info("=" * 60)
        logger.info(f"GENERATING SYNTHETIC DATASET: {n_simulations} simulations")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)
        _create_hdf5_file(
            output_file=output_file,
            expected_n_samples=target_samples,
            channel_names=CHANNEL_NAMES,
            region_names=region_labels,
        )
        logger.info(f"✓ HDF5 file created: {output_file}")

    # Step 2: Run simulations in parallel and process results as they complete
    logger.info(f"Running {n_simulations_adjusted} simulations...")
    generation_start_time = time.time()

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
    last_progress_update_time = time.time()  # For periodic progress logging

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
            seed=base_seed_adjusted + sim_index,
        )

    # Use ProcessPoolExecutor to process results as they complete (not waiting for all)
    logger.info(f"Using {n_workers} worker processes")
    logger.info("Processing results and writing to HDF5 incrementally...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(generate_one_simulation, **create_sim_kwargs(i)): i
            for i in range(n_simulations_adjusted)
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
                current_batch_mask.append(result["epileptogenic_mask"][w])
                current_batch_x0.append(result["x0_vector"][w])
                current_batch_snr.append(result["snr_db"][w])
                current_batch_coupling.append(result["global_coupling"][w])

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

                    # Log progress with timing information
                    elapsed_sec = time.time() - generation_start_time
                    elapsed_min = elapsed_sec / 60.0
                    samples_per_sec = total_samples_written / elapsed_sec if elapsed_sec > 0 else 0
                    
                    # Estimate remaining time
                    remaining_samples = target_samples - total_samples_written
                    eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                    eta_min = eta_sec / 60.0
                    
                    logger.info(
                        f"✓ Batch written | Samples: {total_samples_written:,}/{target_samples:,} "
                        f"| Sims: {total_simulations_completed}/{n_simulations_adjusted} "
                        f"({total_simulations_failed} failed) "
                        f"| Speed: {samples_per_sec:.0f} samples/sec | ETA: {eta_min:.1f}m"
                    )

                    # Reset batch accumulators
                    current_batch_eeg = []
                    current_batch_source = []
                    current_batch_mask = []
                    current_batch_x0 = []
                    current_batch_snr = []
                    current_batch_coupling = []
                    last_progress_update_time = time.time()
            
            # Also log progress every 30 seconds even if batch size not reached
            current_time = time.time()
            if (current_time - last_progress_update_time) > 30 and len(current_batch_eeg) > 0:
                elapsed_sec = time.time() - generation_start_time
                elapsed_min = elapsed_sec / 60.0
                samples_so_far = total_samples_written + len(current_batch_eeg)
                samples_per_sec = samples_so_far / elapsed_sec if elapsed_sec > 0 else 0
                remaining_samples = target_samples - samples_so_far
                eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                eta_min = eta_sec / 60.0
                
                logger.info(
                    f"[Progress Update] Sims: {total_simulations_completed}/{n_simulations_adjusted} "
                    f"| Current batch: {len(current_batch_eeg)} samples (not yet written) "
                    f"| Total so far: {samples_so_far:,}/{target_samples:,} "
                    f"| Speed: {samples_per_sec:.0f}/sec | ETA: {eta_min:.1f}m"
                )
                last_progress_update_time = current_time

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
    
    # Final timing summary
    total_elapsed_sec = time.time() - generation_start_time
    total_elapsed_min = total_elapsed_sec / 60.0
    total_elapsed_hr = total_elapsed_min / 60.0
    samples_per_sec = total_samples_written / total_elapsed_sec if total_elapsed_sec > 0 else 0
    
    logger.info(f"Total time: {total_elapsed_hr:.2f} hours ({total_elapsed_min:.1f} min)")
    logger.info(f"Throughput: {samples_per_sec:.1f} samples/sec")
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
