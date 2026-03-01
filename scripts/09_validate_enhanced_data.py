"""
Script: 09_validate_enhanced_data.py
Purpose: Comprehensive validation of enhanced synthetic EEG data

Validates the spatially-enhanced dataset against:
1. Legacy biophysical metrics (Phase 1)
2. New spatial-spectral metrics (PDR, anteroposterior gradients, frequency distribution)
3. Data integrity checks

Output: Generates validation report with pass/fail metrics and summary statistics
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List
import json

import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants from technical specs
N_REGIONS = 76
N_CHANNELS = 19
SAMPLING_RATE = 200.0
WINDOW_LENGTH = 400

# Channel names (10-20 montage)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Channel groupings for spatial analysis
CHANNEL_GROUPS = {
    'frontal_fp': ['Fp1', 'Fp2'],
    'frontal_f': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'central_c': ['C3', 'C4', 'T3', 'T4', 'Cz'],
    'parietal_p': ['P3', 'P4', 'T5', 'T6', 'Pz'],
    'occipital_o': ['O1', 'O2'],
}

# Frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
}


def compute_psd_welch(
    eeg: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Args:
        eeg: EEG array (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz

    Returns:
        Tuple of (frequencies, psd) where psd shape is (n_channels, n_freqs)
    """
    # Welch's method: 1-second windows with 50% overlap, Hann window
    n_samples = eeg.shape[1]
    nperseg = int(sampling_rate * 1.0)  # 1-second window
    noverlap = nperseg // 2

    psd_list = []
    for ch_idx in range(eeg.shape[0]):
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
        )
        psd_list.append(psd)

    psd = np.array(psd_list)  # Shape: (n_channels, n_freqs)
    return freqs, psd


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_bands: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Compute band power for each frequency band.

    Args:
        freqs: Frequency array
        psd: PSD array (n_channels, n_freqs)
        freq_bands: Dict mapping band names to (fmin, fmax) tuples

    Returns:
        Dict mapping band names to power arrays (n_channels,)
    """
    band_power = {}
    for band_name, (fmin, fmax) in freq_bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        # Integrate PSD over frequency band (trapezoid rule)
        band_power[band_name] = np.trapz(psd[:, mask], freqs[mask], axis=1)

    return band_power


def compute_legacy_metrics(eeg: np.ndarray) -> Dict[str, float]:
    """
    Compute legacy biophysical metrics from Phase 1 validation.

    Args:
        eeg: EEG array (n_channels, n_samples)

    Returns:
        Dict with metrics:
        - rms_uv: RMS in microvolts
        - peak_uv: Peak-to-peak amplitude in microvolts
        - gamma_pct: Gamma band (30-80 Hz) percentage of total power
    """
    # RMS amplitude
    rms_uv = np.sqrt(np.mean(eeg ** 2))

    # Peak-to-peak
    peak_uv = np.ptp(eeg)

    # Spectral metrics
    freqs, psd = compute_psd_welch(eeg)
    total_power = np.sum(psd)
    gamma_mask = (freqs >= 30) & (freqs <= 80)
    gamma_power = np.sum(psd[:, gamma_mask])
    gamma_pct = 100.0 * gamma_power / (total_power + 1e-10)

    return {
        'rms_uv': float(rms_uv),
        'peak_uv': float(peak_uv),
        'gamma_pct': float(gamma_pct),
    }


def compute_pdr_ratio(
    eeg: np.ndarray,
    channel_names: List[str] = CHANNEL_NAMES,
) -> Tuple[float, Dict]:
    """
    Compute Posterior Dominant Rhythm (PDR) ratio.

    PDR = occipital alpha power / frontal alpha power
    Target range: [1.3, 4.0] for healthy adults

    Args:
        eeg: EEG array (n_channels, n_samples)
        channel_names: List of channel names

    Returns:
        Tuple of (pdr_ratio, metrics_dict)
    """
    freqs, psd = compute_psd_welch(eeg)
    band_power = compute_band_power(freqs, psd, {'alpha': (8, 13)})
    alpha_power = band_power['alpha']

    # Map channels to groups
    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    frontal_indices = [ch_to_idx[ch] for ch in ['Fp1', 'Fp2', 'F3', 'F4']]
    occipital_indices = [ch_to_idx[ch] for ch in ['O1', 'O2']]

    frontal_alpha = np.mean(alpha_power[frontal_indices])
    occipital_alpha = np.mean(alpha_power[occipital_indices])

    pdr_ratio = occipital_alpha / (frontal_alpha + 1e-10)

    return float(pdr_ratio), {
        'frontal_alpha': float(frontal_alpha),
        'occipital_alpha': float(occipital_alpha),
        'pdr_pass': 1.3 <= pdr_ratio <= 4.0,
    }


def compute_anteroposterior_gradient(
    eeg: np.ndarray,
    channel_names: List[str] = CHANNEL_NAMES,
    freq_band: Tuple[float, float] = (8, 13),
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Compute anteroposterior gradient for a frequency band.

    For healthy EEG:
    - Alpha (8-13 Hz): Anterior < Central < Posterior < Occipital (increases)
    - Beta (13-30 Hz): Anterior > Central > Posterior > Occipital (decreases)

    Args:
        eeg: EEG array (n_channels, n_samples)
        channel_names: List of channel names
        freq_band: (fmin, fmax) tuple for frequency band

    Returns:
        Tuple of (power_per_region, gradient_pass_dict)
    """
    freqs, psd = compute_psd_welch(eeg)
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    band_psd = psd[:, mask]
    band_power = np.trapz(band_psd, freqs[mask], axis=1)

    # Map channels to groups
    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    group_power = {}
    for group_name, group_channels in CHANNEL_GROUPS.items():
        indices = [ch_to_idx[ch] for ch in group_channels if ch in ch_to_idx]
        group_power[group_name] = np.mean(band_power[indices])

    # Check gradients
    # For alpha: Fp < F < C < P < O should all increase
    alpha_gradient = (
        group_power['frontal_fp'] < group_power['frontal_f'] <
        group_power['central_c'] < group_power['parietal_p'] <
        group_power['occipital_o']
    )

    # For beta: Fp > F > C > P > O should all decrease
    beta_gradient = (
        group_power['frontal_fp'] > group_power['frontal_f'] >
        group_power['central_c'] > group_power['parietal_p'] >
        group_power['occipital_o']
    )

    gradient_passes = {
        'alpha_gradient': alpha_gradient if freq_band == (8, 13) else None,
        'beta_gradient': beta_gradient if freq_band == (13, 30) else None,
    }

    return group_power, gradient_passes


def compute_frequency_distribution(
    eeg: np.ndarray,
    channel_names: List[str] = CHANNEL_NAMES,
) -> Dict[str, Dict[str, float]]:
    """
    Compute frequency distribution (band power as % of total) per channel.

    Args:
        eeg: EEG array (n_channels, n_samples)
        channel_names: List of channel names

    Returns:
        Dict mapping channel names to dict of band percentages
    """
    freqs, psd = compute_psd_welch(eeg)
    band_power = compute_band_power(freqs, psd, FREQ_BANDS)

    ch_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    freq_dist = {}

    for band_name, powers in band_power.items():
        total_power = np.sum(powers)
        for ch_name, ch_idx in ch_to_idx.items():
            if ch_name not in freq_dist:
                freq_dist[ch_name] = {}
            freq_dist[ch_name][band_name] = 100.0 * powers[ch_idx] / (total_power + 1e-10)

    return freq_dist


def validate_single_sample(eeg: np.ndarray) -> Dict:
    """
    Validate a single EEG sample against all metrics.

    Args:
        eeg: EEG array (n_channels, n_samples)

    Returns:
        Dict with validation results
    """
    results = {}

    # Legacy metrics
    results['legacy'] = compute_legacy_metrics(eeg)

    # PDR
    pdr_ratio, pdr_metrics = compute_pdr_ratio(eeg)
    pdr_metrics['pdr_ratio'] = pdr_ratio
    results['pdr'] = pdr_metrics

    # Anteroposterior gradients
    alpha_power, alpha_gradient = compute_anteroposterior_gradient(eeg, freq_band=(8, 13))
    results['alpha_gradient'] = alpha_gradient
    results['alpha_power_per_region'] = alpha_power

    beta_power, beta_gradient = compute_anteroposterior_gradient(eeg, freq_band=(13, 30))
    results['beta_gradient'] = beta_gradient
    results['beta_power_per_region'] = beta_power

    # Frequency distribution
    results['freq_distribution'] = compute_frequency_distribution(eeg)

    return results


def validate_dataset_split(
    h5_path: Path,
    n_samples_to_check: int = 100,
) -> Dict:
    """
    Validate a dataset split (train/val/test).

    Args:
        h5_path: Path to HDF5 file
        n_samples_to_check: Number of random samples to validate

    Returns:
        Dict with aggregated validation statistics
    """
    logger.info(f"Validating {h5_path.name}...")

    with h5py.File(h5_path, 'r') as f:
        n_total_samples = f['eeg'].shape[0]
        n_check = min(n_samples_to_check, n_total_samples)

        # Random sample indices
        sample_indices = np.random.choice(n_total_samples, size=n_check, replace=False)

        # Aggregate results
        results_list = []
        for idx in sample_indices:
            eeg = f['eeg'][idx, :, :]  # Shape: (n_channels, n_samples)
            results = validate_single_sample(eeg)
            results_list.append(results)

    # Aggregate across samples
    aggregated = {
        'n_samples_checked': n_check,
        'n_total_samples': n_total_samples,
        'legacy_metrics': {
            'rms_uv_mean': float(np.mean([r['legacy']['rms_uv'] for r in results_list])),
            'rms_uv_std': float(np.std([r['legacy']['rms_uv'] for r in results_list])),
            'peak_uv_mean': float(np.mean([r['legacy']['peak_uv'] for r in results_list])),
            'gamma_pct_mean': float(np.mean([r['legacy']['gamma_pct'] for r in results_list])),
        },
        'pdr_metrics': {
            'pdr_ratio_mean': float(np.mean([r['pdr']['pdr_ratio'] for r in results_list])),
            'pdr_pass_rate': 100.0 * np.mean([r['pdr']['pdr_pass'] for r in results_list]),
        },
        'gradients': {
            'alpha_gradient_pass_rate': 100.0 * np.mean([r['alpha_gradient']['alpha_gradient'] for r in results_list if r['alpha_gradient']['alpha_gradient'] is not None]),
            'beta_gradient_pass_rate': 100.0 * np.mean([r['beta_gradient']['beta_gradient'] for r in results_list if r['beta_gradient']['beta_gradient'] is not None]),
        },
    }

    logger.info(f"✓ Validation complete for {h5_path.name}")
    return aggregated


def main():
    """Run comprehensive validation on enhanced dataset."""
    logger.info("=" * 80)
    logger.info("PhysDeepSIF Enhanced Data Validation")
    logger.info("=" * 80)

    data_root = Path('/data1tb/VESL/fyp-2.0/data')
    enhanced_dir = data_root / 'synthetic'
    
    # Check if enhanced data exists
    if not enhanced_dir.exists():
        logger.error(f"Enhanced data directory not found: {enhanced_dir}")
        return

    results = {}

    # Validate each split
    for split in ['train', 'val', 'test']:
        h5_path = enhanced_dir / f'{split}_dataset.h5'
        if h5_path.exists():
            logger.info(f"\nValidating {split} split...")
            results[split] = validate_dataset_split(h5_path, n_samples_to_check=100)
        else:
            logger.warning(f"File not found: {h5_path}")

    # Print summary report
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY REPORT")
    logger.info("=" * 80)

    for split, metrics in results.items():
        logger.info(f"\n{split.upper()} SPLIT:")
        logger.info(f"  Samples checked: {metrics['n_samples_checked']} / {metrics['n_total_samples']}")
        logger.info(f"\n  Legacy Metrics:")
        logger.info(f"    RMS: {metrics['legacy_metrics']['rms_uv_mean']:.2f} ± {metrics['legacy_metrics']['rms_uv_std']:.2f} µV")
        logger.info(f"    Peak-to-peak: {metrics['legacy_metrics']['peak_uv_mean']:.2f} µV")
        logger.info(f"    Gamma %: {metrics['legacy_metrics']['gamma_pct_mean']:.2f}%")
        logger.info(f"\n  Spatial-Spectral Metrics:")
        logger.info(f"    PDR pass rate: {metrics['pdr_metrics']['pdr_pass_rate']:.1f}%")
        logger.info(f"    Alpha gradient pass rate: {metrics['gradients']['alpha_gradient_pass_rate']:.1f}%")
        logger.info(f"    Beta gradient pass rate: {metrics['gradients']['beta_gradient_pass_rate']:.1f}%")

    # Save detailed results to JSON
    output_file = Path('/data1tb/VESL/fyp-2.0/outputs/validation_report.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Detailed validation report saved to {output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
