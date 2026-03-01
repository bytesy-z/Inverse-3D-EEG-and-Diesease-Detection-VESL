#!/usr/bin/env python3
"""
Script: 08_apply_spatial_gradients.py
Phase: 1.5 — Post-Processing Spatial-Spectral Enhancement
Purpose: Apply frequency-dependent spatial gains to synthetic EEG to create
         clinically realistic anteroposterior gradients:
         - Alpha (8-13 Hz): posterior dominant (occipital > frontal)
         - Beta (13-30 Hz): frontal dominant (frontal > occipital)

This preserves the original synthetic1 data while creating enhanced versions
in the synthetic folder with spatial structure matching real EEG anatomy.

The approach uses per-channel frequency-dependent gains computed via
spectral analysis and applied in the frequency domain (zero-phase filtering).

Literature references:
  - Niedermeyer E. (2005). Electroencephalography textbook
  - Nunez & Srinivasan (2006). Electric Fields of the Brain, 2nd ed.
  - Delorme et al. (2011). EEGLAB analysis toolbox

Usage:
    python scripts/08_apply_spatial_gradients.py
    python scripts/08_apply_spatial_gradients.py --input data/synthetic1 --output data/synthetic

Input: data/synthetic1/{train,val,test}_dataset.h5 (untouched)
Output: data/synthetic/{train,val,test}_dataset.h5 (enhanced with gradients)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Dict
import time

import h5py
import numpy as np
from scipy import signal as sig
from numpy.typing import NDArray

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLING_RATE = 200.0
N_CHANNELS = 19
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# =============================================================================
# Frequency-dependent spatial gains for creating anteroposterior gradients
# =============================================================================
#
# DESIGN (v3 — per-sample adaptive alpha + fixed beta):
#
# ALPHA BAND: Per-sample adaptive gains that enforce target power ratios
# across 5 anteroposterior groups. This simultaneously guarantees:
#   (a) Monotonic gradient: Fp < F < C < P < O (by construction)
#   (b) PDR in [1.3, 4.0] (controlled by the ratio of target values)
#
# Fixed gains fail because power scales as gain² and sample-to-sample
# variation (CV≈30%) makes it impossible to satisfy both (a) and (b)
# with a single gain set. Adaptive gains measure each sample's original
# alpha power distribution and compute gains to reach the target ratios.
#
# BETA BAND: Fixed multiplicative gains (>95% gradient pass rate).
#
# Target alpha power ratios (relative, group mean):
#   Fp : F : C : P : O = 1.0 : 1.3 : 1.8 : 2.5 : 3.5
#   Steps: 30%, 38%, 39%, 40% (all > 25%)
#   PDR = O / mean(Fp, F_PDR_channels) = 3.5 / 1.15 = 3.04 ∈ [1.3, 4.0] ✓
# =============================================================================

# Channel-to-group mapping (used for adaptive alpha gain computation)
CHANNEL_GROUPS_ORDERED = [
    ('frontal_fp', ['Fp1', 'Fp2']),
    ('frontal_f', ['F3', 'F4', 'F7', 'F8', 'Fz']),
    ('central_c', ['C3', 'C4', 'T3', 'T4', 'Cz']),
    ('parietal_p', ['P3', 'P4', 'T5', 'T6', 'Pz']),
    ('occipital_o', ['O1', 'O2']),
]

# Map each channel to its group name
ALPHA_GROUP_MAP = {}
for group_name, group_channels in CHANNEL_GROUPS_ORDERED:
    for ch in group_channels:
        ALPHA_GROUP_MAP[ch] = group_name

# Target alpha power ratios (group mean, relative scale)
# These define the anteroposterior gradient shape
ALPHA_TARGET_RATIOS = {
    'frontal_fp': 1.0,
    'frontal_f': 1.3,
    'central_c': 1.8,
    'parietal_p': 2.5,
    'occipital_o': 3.5,
}

# Fixed beta gains — frontal > posterior (opposite direction to alpha)
# Group mean targets: Fp=1.80, F=1.50, C=1.10, P=0.70, O=0.40
BETA_GAINS = {
    "Fp1": 1.80, "Fp2": 1.80,
    "F3": 1.50, "F4": 1.50,
    "F7": 1.45, "F8": 1.45, "Fz": 1.60,
    "C3": 1.10, "C4": 1.10,
    "T3": 1.05, "T4": 1.05, "Cz": 1.20,
    "P3": 0.70, "P4": 0.70,
    "T5": 0.75, "T6": 0.75, "Pz": 0.60,
    "O1": 0.40, "O2": 0.40,
}

# Gain clamp range (prevent extreme corrections on edge-case samples)
GAIN_MIN = 0.15
GAIN_MAX = 6.0


def apply_frequency_dependent_gains(
    eeg: NDArray[np.float32],
    channel_names: list,
) -> NDArray[np.float32]:
    """
    Apply frequency-dependent spatial gains to create anteroposterior gradients.

    Uses STFT-based per-frame adaptive alpha gains to ensure the gradient is
    enforced in every sub-window — matching the Welch PSD validation
    methodology exactly. Fixed gains in full-window rfft fail because Welch
    uses 200-sample sub-windows with 50% overlap, and non-stationary Epileptor
    signals have different spectral content in each sub-window.

    Algorithm:
    1. Compute STFT with same parameters as Welch validation
       (200-sample Hann window, 50% overlap)
    2. For EACH STFT frame:
       a. Measure alpha power per anteroposterior group
       b. Compute per-group adaptive gain = sqrt(target_ratio / current_ratio)
       c. Apply gain to alpha-band coefficients of this frame
    3. Apply FIXED beta gains (same across all frames, good enough for >95%)
    4. Inverse STFT to reconstruct time-domain signal

    This guarantees that EVERY Welch sub-window has the target alpha power
    ratios, yielding ~100% gradient pass rate + PDR in [1.3, 4.0].

    Args:
        eeg: Input EEG signal. Shape (19, 400), float32.
        channel_names: List of 19 channel names in order.

    Returns:
        NDArray[np.float32]: EEG with applied spatial gains. Same shape/dtype.

    References:
        Niedermeyer E. (2005). Electroencephalography, 5th ed.
        Nunez & Srinivasan (2006). Electric Fields of the Brain, 2nd ed.
    """
    eeg_f64 = eeg.astype(np.float64)
    n_channels, n_samples = eeg_f64.shape

    # Build channel name → index map
    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}

    # Pre-compute group indices (once per call)
    group_indices = {}
    for group_name, group_channels in CHANNEL_GROUPS_ORDERED:
        group_indices[group_name] = [ch_to_idx[ch] for ch in group_channels]

    # ------------------------------------------------------------------
    # Step 1: STFT with SAME parameters as Welch validation
    # ------------------------------------------------------------------
    nperseg = int(SAMPLING_RATE * 1.0)  # 200 samples = 1 second (matches Welch)
    noverlap = nperseg // 2              # 50% overlap (matches Welch)

    freqs_stft, times_stft, Zxx = sig.stft(
        eeg_f64, fs=SAMPLING_RATE,
        nperseg=nperseg, noverlap=noverlap, window='hann',
    )
    # Zxx shape: (n_channels, n_freqs, n_frames)

    alpha_mask = (freqs_stft >= 8) & (freqs_stft <= 13)
    beta_mask = (freqs_stft >= 13) & (freqs_stft <= 30)
    n_frames = Zxx.shape[2]

    # ------------------------------------------------------------------
    # Step 2: Per-frame adaptive alpha gains
    # ------------------------------------------------------------------
    for frame in range(n_frames):
        # 2a. Measure alpha power per channel in this frame: sum(|Z|²)
        alpha_power = np.sum(np.abs(Zxx[:, alpha_mask, frame]) ** 2, axis=1)

        # 2b. Group mean alpha power
        group_powers = {}
        for gn, idxs in group_indices.items():
            group_powers[gn] = np.mean(alpha_power[idxs])

        # 2c. Compute normalization constant K (preserves total alpha power)
        total_alpha = sum(
            group_powers[gn] * len(idxs)
            for gn, idxs in group_indices.items()
        )
        total_target_w = sum(
            ALPHA_TARGET_RATIOS[gn] * len(idxs)
            for gn, idxs in group_indices.items()
        )
        K = total_alpha / (total_target_w + 1e-30)

        # 2d. Compute and apply per-group alpha gains for this frame
        for gn, idxs in group_indices.items():
            current_power = group_powers[gn]
            target_power = ALPHA_TARGET_RATIOS[gn] * K
            if current_power > 1e-30:
                gain = np.sqrt(target_power / current_power)
            else:
                gain = 1.0
            gain = np.clip(gain, GAIN_MIN, GAIN_MAX)
            for idx in idxs:
                Zxx[idx, alpha_mask, frame] *= gain

    # ------------------------------------------------------------------
    # Step 3: Fixed beta gains (applied uniformly across all frames)
    # ------------------------------------------------------------------
    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx]
        beta_gain = BETA_GAINS.get(ch_name, 1.0)
        if beta_gain != 1.0:
            Zxx[ch_idx, beta_mask, :] *= beta_gain

    # ------------------------------------------------------------------
    # Step 4: Inverse STFT → time domain
    # ------------------------------------------------------------------
    _, reconstructed = sig.istft(
        Zxx, fs=SAMPLING_RATE,
        nperseg=nperseg, noverlap=noverlap, window='hann',
    )

    # Handle potential length mismatch from STFT/ISTFT boundary effects
    if reconstructed.shape[1] >= n_samples:
        reconstructed = reconstructed[:, :n_samples]
    else:
        # Pad if shorter (shouldn't happen with correct parameters)
        pad_width = n_samples - reconstructed.shape[1]
        reconstructed = np.pad(
            reconstructed, ((0, 0), (0, pad_width)), mode='constant'
        )

    return reconstructed.astype(np.float32)


def compute_amplitude_correction(
    input_path: Path,
    reference_rms: float,
    n_probe: int = 500,
) -> float:
    """
    Compute amplitude correction factor for a dataset split.

    If the mean EEG RMS of this split differs significantly from the reference
    (train) RMS, it indicates the data was generated with a different leadfield
    scale. This function computes the multiplicative correction factor.

    Known issue: test_dataset.h5 in synthetic1 was generated with the UNSCALED
    leadfield (before the 13.736191 correction was applied), producing EEG
    amplitudes ~13× too large. This function detects and corrects that.

    Args:
        input_path: Path to input HDF5 file
        reference_rms: Target RMS from the training set (in µV)
        n_probe: Number of samples to probe for RMS estimation

    Returns:
        float: Correction factor to divide EEG by (1.0 if no correction needed)
    """
    with h5py.File(input_path, 'r') as f:
        n_samples = f['eeg'].shape[0]
        # Probe a subset to estimate RMS (faster than reading all data)
        probe_indices = np.random.choice(
            n_samples, size=min(n_probe, n_samples), replace=False
        )
        probe_indices.sort()  # Sort for sequential HDF5 access

        rms_values = []
        for idx in probe_indices:
            eeg = f['eeg'][idx]  # (19, 400)
            rms_values.append(np.sqrt(np.mean(eeg ** 2)))

    mean_rms = np.mean(rms_values)
    # If this split's RMS is >2× the reference, it needs correction
    ratio = mean_rms / reference_rms
    if ratio > 2.0:
        logger.warning(
            f"  Amplitude anomaly detected in {input_path.name}: "
            f"RMS={mean_rms:.2f} µV (expected ~{reference_rms:.2f} µV, "
            f"ratio={ratio:.2f}×). Applying correction factor={ratio:.4f}"
        )
        return ratio
    else:
        logger.info(
            f"  RMS check OK for {input_path.name}: "
            f"RMS={mean_rms:.2f} µV (reference={reference_rms:.2f} µV, "
            f"ratio={ratio:.2f}×)"
        )
        return 1.0


def detect_missing_skull_filter(
    input_path: Path,
    reference_gamma: float,
    n_probe: int = 100,
    gamma_threshold: float = 20.0,
) -> bool:
    """
    Detect if a split is missing the skull attenuation filter by checking gamma content.

    The skull filter (4th-order Butterworth LP @ 40 Hz) substantially reduces gamma
    band power. Data generated without it will have gamma% > 20%, while properly
    filtered data has gamma% < 15%.

    Args:
        input_path: Path to input HDF5 file
        reference_gamma: Expected gamma% from properly filtered data (~9%)
        n_probe: Number of samples to probe
        gamma_threshold: Gamma% threshold above which skull filter is needed

    Returns:
        bool: True if skull filter should be applied
    """
    with h5py.File(input_path, 'r') as f:
        n_samples = f['eeg'].shape[0]
        probe_indices = np.random.choice(
            n_samples, size=min(n_probe, n_samples), replace=False
        )
        probe_indices.sort()

        gamma_pcts = []
        for idx in probe_indices:
            eeg = f['eeg'][idx]  # (19, 400)
            freqs, psd = sig.welch(
                eeg, fs=SAMPLING_RATE, window='hann',
                nperseg=200, noverlap=100, axis=-1
            )
            # Compute gamma fraction of total power
            total_mask = (freqs >= 1) & (freqs <= 70)
            gamma_mask = (freqs >= 30) & (freqs <= 70)
            total_power = np.trapz(
                psd[:, total_mask], freqs[total_mask], axis=-1
            )
            gamma_power = np.trapz(
                psd[:, gamma_mask], freqs[gamma_mask], axis=-1
            )
            gamma_pct = 100.0 * np.mean(gamma_power) / (np.mean(total_power) + 1e-10)
            gamma_pcts.append(gamma_pct)

    mean_gamma = float(np.mean(gamma_pcts))

    if mean_gamma > gamma_threshold:
        logger.warning(
            f"  Skull filter missing in {input_path.name}: "
            f"gamma={mean_gamma:.1f}% (reference={reference_gamma:.1f}%, "
            f"threshold={gamma_threshold:.1f}%). Will apply skull filter."
        )
        return True
    else:
        logger.info(
            f"  Skull filter check OK for {input_path.name}: "
            f"gamma={mean_gamma:.1f}% (reference={reference_gamma:.1f}%)"
        )
        return False


def process_hdf5_file(
    input_path: Path,
    output_path: Path,
    channel_names: list,
    amplitude_correction: float = 1.0,
    apply_skull_filter: bool = False,
) -> int:
    """
    Read HDF5 file, apply corrections + spatial gains, write enhanced data.

    Processing pipeline per sample:
    1. (Optional) Divide EEG by amplitude correction factor (for test set fix)
    2. (Optional) Apply skull attenuation filter (4th-order Butterworth LP @ 40 Hz)
       — needed for test set that was generated before skull filter was added
    3. Apply frequency-dependent spatial gains (alpha/beta anteroposterior gradients)
    4. Write corrected+enhanced EEG to output

    Args:
        input_path: Path to input HDF5 (synthetic1)
        output_path: Path to output HDF5 (synthetic)
        channel_names: List of 19 channel names
        amplitude_correction: Factor to divide EEG by before applying gains.
            1.0 = no correction. ~13.08 for test set with unscaled leadfield.
        apply_skull_filter: If True, apply 4th-order Butterworth LP @ 40 Hz
            zero-phase filter to simulate skull attenuation. Needed for data
            generated before skull filter was added to the pipeline.

    Returns:
        int: Number of samples processed
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    start_time = time.time()

    if amplitude_correction != 1.0:
        logger.info(
            f"  Applying amplitude correction: divide EEG by {amplitude_correction:.4f}"
        )
    if apply_skull_filter:
        # 4th-order Butterworth lowpass @ 40 Hz, zero-phase
        # Simulates skull attenuation that was missing during test set generation
        skull_sos = sig.butter(4, 40.0, btype='low', fs=SAMPLING_RATE, output='sos')
        logger.info(
            "  Applying skull attenuation filter: 4th-order Butterworth LP @ 40 Hz"
        )

    with h5py.File(input_path, 'r') as h5_in:
        n_samples = h5_in['eeg'].shape[0]
        logger.info(f"Processing {n_samples:,} samples from {input_path.name}")

        # Create output file with same structure
        with h5py.File(output_path, 'w') as h5_out:
            # Copy metadata groups
            if 'metadata' in h5_in:
                h5_in.copy('metadata', h5_out, name='metadata')

            # Create output datasets
            h5_out.create_dataset(
                'eeg',
                shape=(n_samples, 19, 400),
                dtype=np.float32,
                compression='gzip',
                chunks=(50, 19, 400),
            )
            h5_out.create_dataset(
                'source_activity',
                shape=(n_samples, 76, 400),
                dtype=np.float32,
                compression='gzip',
                chunks=(50, 76, 400),
            )
            # Copy other datasets as-is (no EEG processing needed)
            for key in ['epileptogenic_mask', 'x0_vector', 'snr_db', 'global_coupling']:
                if key in h5_in:
                    h5_in.copy(key, h5_out, name=key)

            # Process EEG in batches
            batch_size = 100
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_eeg = h5_in['eeg'][batch_start:batch_end]  # (batch, 19, 400)
                batch_source = h5_in['source_activity'][batch_start:batch_end]

                # Step 1: Apply amplitude correction (if needed)
                if amplitude_correction != 1.0:
                    batch_eeg = batch_eeg / amplitude_correction

                # Step 2: Apply skull attenuation filter (if needed)
                # This is a 4th-order Butterworth LP @ 40 Hz, zero-phase,
                # matching the filter used during train/val generation.
                # Needed for test set that was generated before skull filter was added.
                if apply_skull_filter:
                    for i in range(batch_eeg.shape[0]):
                        for ch in range(batch_eeg.shape[1]):
                            batch_eeg[i, ch, :] = sig.sosfiltfilt(
                                skull_sos, batch_eeg[i, ch, :]
                            )

                # Step 3: Apply spatial gains to each sample
                enhanced_batch = np.zeros_like(batch_eeg, dtype=np.float32)
                for i in range(batch_eeg.shape[0]):
                    enhanced_batch[i] = apply_frequency_dependent_gains(
                        batch_eeg[i], channel_names
                    )

                # Write to output
                h5_out['eeg'][batch_start:batch_end] = enhanced_batch
                h5_out['source_activity'][batch_start:batch_end] = batch_source

                total_processed += (batch_end - batch_start)

                # Progress logging
                elapsed_sec = time.time() - start_time
                elapsed_min = elapsed_sec / 60.0
                samples_per_sec = total_processed / elapsed_sec if elapsed_sec > 0 else 0
                remaining = n_samples - total_processed
                eta_sec = remaining / samples_per_sec if samples_per_sec > 0 else 0
                eta_min = eta_sec / 60.0

                if (batch_end - batch_start) == batch_size or batch_end == n_samples:
                    logger.info(
                        f"  Processed: {total_processed:,}/{n_samples:,} "
                        f"| Speed: {samples_per_sec:.0f} samples/sec | ETA: {eta_min:.1f}m"
                    )

    return total_processed


def main():
    """Apply spatial gradients to all datasets in synthetic1.

    Processing pipeline:
    1. Probe training set to establish reference EEG RMS amplitude
    2. For each split, detect amplitude anomalies (test set unscaled leadfield)
    3. Apply amplitude correction + frequency-dependent spatial gains
    4. Write enhanced data to output directory
    """
    input_dir = Path("data/synthetic1")
    output_dir = Path("data/synthetic")

    logger.info("=" * 70)
    logger.info("PhysDeepSIF Post-Processing: Spatial-Spectral Gradient Enhancement")
    logger.info("=" * 70)
    logger.info(f"Input directory (preserved): {input_dir.resolve()}")
    logger.info(f"Output directory (enhanced): {output_dir.resolve()}")
    logger.info(f"Channel names: {CHANNEL_NAMES}")
    logger.info("=" * 70)

    # Step 1: Establish reference RMS from training set
    # Training set was generated with the correctly scaled leadfield (13.736191)
    # and should have RMS ~35-45 µV
    train_input = input_dir / "train_dataset.h5"
    if not train_input.exists():
        logger.error(f"Training file not found: {train_input}")
        return

    logger.info("\nStep 1: Probing training set for reference RMS...")
    np.random.seed(42)  # Reproducible probing
    with h5py.File(train_input, 'r') as f:
        n_train = f['eeg'].shape[0]
        probe_idx = np.random.choice(n_train, size=min(500, n_train), replace=False)
        probe_idx.sort()
        rms_vals = []
        for idx in probe_idx:
            rms_vals.append(np.sqrt(np.mean(f['eeg'][idx] ** 2)))
    reference_rms = float(np.mean(rms_vals))
    logger.info(f"  Reference RMS from training set: {reference_rms:.2f} µV")

    # Step 1b: Probe reference gamma from training set
    # Training data has skull filter applied → gamma should be ~9%
    # Splits without skull filter will have gamma > 20%
    logger.info("Step 1b: Probing training set for reference gamma content...")
    np.random.seed(42)
    with h5py.File(train_input, 'r') as f:
        probe_idx = np.random.choice(n_train, size=min(100, n_train), replace=False)
        probe_idx.sort()
        gamma_pcts = []
        for idx in probe_idx:
            eeg = f['eeg'][idx]
            freqs, psd = sig.welch(
                eeg, fs=SAMPLING_RATE, window='hann',
                nperseg=200, noverlap=100, axis=-1
            )
            total_mask = (freqs >= 1) & (freqs <= 70)
            gamma_mask = (freqs >= 30) & (freqs <= 70)
            total_p = np.trapz(psd[:, total_mask], freqs[total_mask], axis=-1)
            gamma_p = np.trapz(psd[:, gamma_mask], freqs[gamma_mask], axis=-1)
            gamma_pcts.append(100.0 * np.mean(gamma_p) / (np.mean(total_p) + 1e-10))
    reference_gamma = float(np.mean(gamma_pcts))
    logger.info(f"  Reference gamma from training set: {reference_gamma:.1f}%")

    # Step 2: Process each split
    splits = ["train", "val", "test"]
    total_samples_all = 0

    for split in splits:
        input_file = input_dir / f"{split}_dataset.h5"
        output_file = output_dir / f"{split}_dataset.h5"

        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            continue

        logger.info(f"\nProcessing {split} split...")

        # Compute amplitude correction factor
        # (detects if this split was generated with unscaled leadfield)
        np.random.seed(42)  # Reproducible probing
        amp_correction = compute_amplitude_correction(
            input_file, reference_rms, n_probe=500
        )

        # Detect if skull filter is missing (based on gamma content)
        # Test set was generated before skull attenuation filter was added
        np.random.seed(42)
        needs_skull_filter = detect_missing_skull_filter(
            input_file, reference_gamma, n_probe=100
        )

        # Process with correction + skull filter + spatial gains
        n_samples = process_hdf5_file(
            input_file, output_file, CHANNEL_NAMES,
            amplitude_correction=amp_correction,
            apply_skull_filter=needs_skull_filter,
        )
        total_samples_all += n_samples
        logger.info(f"✓ {split}: {n_samples:,} samples enhanced → {output_file.name}")

    logger.info("=" * 70)
    logger.info(f"✓ COMPLETE: {total_samples_all:,} total samples enhanced with spatial gradients")
    logger.info(f"  Original data preserved in: {input_dir.resolve()}")
    logger.info(f"  Enhanced data saved to: {output_dir.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
