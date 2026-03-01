#!/usr/bin/env python3
"""
Quick test: Run 5 TVB simulations through the FULL pipeline (including new
spectral shaping) and analyze the resulting band power distribution per channel.

This validates that the spectral shaping produces clinically realistic EEG.
"""
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import welch

# Suppress TVB warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_forward.source_space import load_source_space_data
from src.phase1_forward.leadfield_builder import load_leadfield
from src.phase1_forward.synthetic_dataset import generate_one_simulation

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load config and data
config = yaml.safe_load(open(PROJECT_ROOT / 'config.yaml'))
data_dir = str(PROJECT_ROOT / 'data')
connectivity, region_centers, region_labels, tract_lengths = load_source_space_data(data_dir)
leadfield = load_leadfield(str(PROJECT_ROOT / 'data/leadfield_19x76.npy'))

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

CHANNEL_GROUPS = {
    'frontal_fp': ['Fp1', 'Fp2'],
    'frontal_f': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'central_c': ['C3', 'C4', 'T3', 'T4', 'Cz'],
    'parietal_p': ['P3', 'P4', 'T5', 'T6', 'Pz'],
    'occipital_o': ['O1', 'O2'],
}


def get_band_power(freqs, psd, freq_range):
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return np.trapz(psd[mask], freqs[mask])


def analyze_eeg(eeg_samples, label):
    """Analyze spectral properties of EEG samples (n_samples, 19, 400)."""
    fs = 200.0
    n_samples = eeg_samples.shape[0]

    # Compute PSD per channel, averaged across samples
    ch_psd = {}  # channel_name -> average PSD
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        psd_list = []
        for s in range(n_samples):
            freqs, psd = welch(eeg_samples[s, ch_idx, :], fs=fs,
                               window='hann', nperseg=200, noverlap=100)
            psd_list.append(psd)
        ch_psd[ch_name] = np.mean(psd_list, axis=0)

    bands = {
        'Delta (1-4)': (1, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-70)': (30, 70),
    }

    print(f"\n{'='*80}")
    print(f"SPECTRAL ANALYSIS: {label} ({n_samples} samples)")
    print(f"{'='*80}")

    # Overall band distribution (average ALL channels)
    all_ch_psd = np.mean([ch_psd[ch] for ch in CHANNEL_NAMES], axis=0)
    total_power = get_band_power(freqs, all_ch_psd, (1, 70))
    print(f"\nOverall band power distribution (avg across all channels):")
    print(f"{'-'*60}")
    for bname, brange in bands.items():
        bp = get_band_power(freqs, all_ch_psd, brange)
        pct = 100 * bp / total_power
        print(f"  {bname:20s}: {pct:6.2f}%")

    # Peak frequency
    peak_idx = np.argmax(all_ch_psd[1:]) + 1  # Skip DC
    print(f"\n  Peak frequency: {freqs[peak_idx]:.1f} Hz")

    # Per-group band distribution
    print(f"\nPer-group band distribution:")
    print(f"{'-'*80}")
    print(f"{'Group':15s} {'Delta':>8s} {'Theta':>8s} {'Alpha':>8s} {'Beta':>8s} {'Gamma':>8s}")
    print(f"{'-'*80}")

    group_alpha = {}
    group_beta = {}
    for gname, gchannels in CHANNEL_GROUPS.items():
        gpsd = np.mean([ch_psd[ch] for ch in gchannels], axis=0)
        gtotal = get_band_power(freqs, gpsd, (1, 70))
        row = []
        for bname, brange in bands.items():
            bp = get_band_power(freqs, gpsd, brange)
            pct = 100 * bp / gtotal
            row.append(pct)
        print(f"{gname:15s} {row[0]:7.1f}% {row[1]:7.1f}% {row[2]:7.1f}% {row[3]:7.1f}% {row[4]:7.1f}%")
        group_alpha[gname] = row[2]
        group_beta[gname] = row[3]

    # Check gradients
    alpha_vals = [group_alpha[g] for g in ['frontal_fp', 'frontal_f', 'central_c', 'parietal_p', 'occipital_o']]
    beta_vals = [group_beta[g] for g in ['frontal_fp', 'frontal_f', 'central_c', 'parietal_p', 'occipital_o']]

    alpha_monotonic = all(a2 > a1 for a1, a2 in zip(alpha_vals[:-1], alpha_vals[1:]))
    beta_monotonic = all(b2 < b1 for b1, b2 in zip(beta_vals[:-1], beta_vals[1:]))

    pdr = alpha_vals[-1] / alpha_vals[0] if alpha_vals[0] > 0 else 0

    print(f"\n{'='*60}")
    print(f"VALIDATION CHECKS:")
    print(f"  Alpha gradient (Fp→O): {' < '.join(f'{v:.1f}' for v in alpha_vals)}")
    print(f"  Alpha monotonic:       {'PASS' if alpha_monotonic else 'FAIL'}")
    print(f"  Beta gradient (Fp→O):  {' > '.join(f'{v:.1f}' for v in beta_vals)}")
    print(f"  Beta monotonic:        {'PASS' if beta_monotonic else 'FAIL'}")
    print(f"  PDR (O/Fp alpha):      {pdr:.2f} (target: 1.3-4.0) {'PASS' if 1.3 <= pdr <= 4.0 else 'FAIL'}")

    # Clinical targets
    clinical = {
        'Delta (1-4)': 10, 'Theta (4-8)': 10, 'Alpha (8-13)': 30,
        'Beta (13-30)': 22, 'Gamma (30-70)': 8,
    }
    all_total = get_band_power(freqs, all_ch_psd, (1, 70))
    print(f"\n  Clinical comparison (overall average):")
    for bname, brange in bands.items():
        bp = get_band_power(freqs, all_ch_psd, brange)
        synth_pct = 100 * bp / all_total
        clin_pct = clinical[bname]
        diff = synth_pct - clin_pct
        status = "OK" if abs(diff) < 15 else "NEEDS TUNING"
        print(f"    {bname:20s}: synth={synth_pct:5.1f}% clin={clin_pct:5.1f}% diff={diff:+5.1f}% {status}")

    # RMS amplitude
    rms_vals = [np.sqrt(np.mean(eeg_samples[s] ** 2)) for s in range(n_samples)]
    mean_rms = np.mean(rms_vals)
    print(f"\n  RMS amplitude: {mean_rms:.2f} µV (target: 5-150 µV) {'PASS' if 5 <= mean_rms <= 150 else 'FAIL'}")

    return group_alpha, group_beta, pdr


# Run 5 simulations
print("Running 5 TVB simulations with spectral shaping...")
all_eeg = []
all_source = []
n_total_windows = 0
n_valid_windows = 0

for sim_idx in range(5):
    result = generate_one_simulation(
        sim_index=sim_idx,
        connectivity=connectivity,
        region_centers=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
        leadfield=leadfield,
        config=config,
        seed=42 + sim_idx,
    )
    if result is not None:
        n_w = result['eeg'].shape[0]
        n_total_windows += 5
        n_valid_windows += n_w
        for w in range(n_w):
            all_eeg.append(result['eeg'][w])
            all_source.append(result['source_activity'][w])
        n_epi = int(result['epileptogenic_mask'][0].sum())
        print(f"  Sim {sim_idx}: {n_w}/5 valid windows, {n_epi} epileptogenic regions")
    else:
        n_total_windows += 5
        print(f"  Sim {sim_idx}: FAILED (all windows rejected or simulation diverged)")

all_eeg = np.array(all_eeg)
all_source = np.array(all_source)

print(f"\nTotal: {n_valid_windows}/{n_total_windows} windows passed validation "
      f"({100*n_valid_windows/max(1,n_total_windows):.0f}%)")
print(f"EEG shape: {all_eeg.shape}")

# Analyze the shaped EEG
if len(all_eeg) > 0:
    group_alpha, group_beta, pdr = analyze_eeg(all_eeg, "Spectrally Shaped Pipeline")
