#!/usr/bin/env python3
"""
Script: 11_plot_fresh_samples.py
Purpose: Generate fresh synthetic EEG samples and plot them.

Generates 3 fresh TVB simulations (mixed healthy and epileptic) through the full
pipeline (TVB → leadfield → noise → skull LP → spectral shaping) and saves
plot visualizations for inspection.

Usage:
    python scripts/11_plot_fresh_samples.py

Outputs:
- fresh_samples_eeg.png: MNE timeseries plot
- fresh_samples_psd.png: Spectral analysis with band power breakdown

Expected runtime: ~30-60 seconds (3 simulations)
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import yaml
import mne
from mne import create_info
from mne.io import RawArray
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import welch

# Suppress TVB warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_forward.source_space import load_source_space_data
from src.phase1_forward.leadfield_builder import load_leadfield
from src.phase1_forward.synthetic_dataset import generate_one_simulation

# Suppress verbose logging except for critical messages
logging.basicConfig(level=logging.WARNING)

# Constants
N_CHANNELS = 19
SAMPLING_RATE = 200.0
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


def main():
    """Generate fresh samples and display interactively."""
    
    print("Loading project configuration and data...")
    config = yaml.safe_load(open(PROJECT_ROOT / 'config.yaml'))
    data_dir = str(PROJECT_ROOT / 'data')
    connectivity, region_centers, region_labels, tract_lengths = (
        load_source_space_data(data_dir)
    )
    leadfield = load_leadfield(
        str(PROJECT_ROOT / 'data/leadfield_19x76.npy')
    )

    print("\nGenerating 3 fresh samples through full pipeline...")
    print("(TVB → decimation → leadfield → noise → skull LP → spectral shaping)")
    
    # Generate samples
    samples_eeg = []
    samples_masks = []
    sample_info = []
    
    for sim_idx in range(3):
        result = generate_one_simulation(
            sim_index=sim_idx,
            connectivity=connectivity,
            region_centers=region_centers,
            region_labels=region_labels,
            tract_lengths=tract_lengths,
            leadfield=leadfield,
            config=config,
            seed=1000 + sim_idx,  # Different seeds for variety
        )
        
        if result is not None:
            # Take first valid window from each simulation
            eeg = result['eeg'][0]  # shape (19, 400)
            mask = result['epileptogenic_mask'][0]  # shape (76,)
            snr = result['snr_db'][0] if 'snr_db' in result else None
            n_epi = int(mask.sum())
            
            samples_eeg.append(eeg)
            samples_masks.append(mask)
            sample_info.append({
                'sim': sim_idx,
                'n_epileptogenic': n_epi,
                'snr_db': snr,
            })
            print(f"  Sim {sim_idx}: ✓ Generated (epileptogenic regions: {n_epi})")
        else:
            print(f"  Sim {sim_idx}: ✗ Failed (diverged or all windows rejected)")
    
    if not samples_eeg:
        print("ERROR: Failed to generate any samples.")
        return 1
    
    print(f"\nSuccessfully generated {len(samples_eeg)} samples.")
    print("\nSample statistics:")
    
    for i, (eeg, info) in enumerate(zip(samples_eeg, sample_info)):
        rms = np.sqrt(np.mean(eeg ** 2))
        print(f"  Sample {i}: RMS={rms:.2f} µV, "
              f"epileptogenic_regions={info['n_epileptogenic']}, "
              f"SNR={info['snr_db']:.1f} dB" if info['snr_db'] else "")
    
    # Create MNE RawArray for plotting
    print("\nCreating MNE interactive viewer...")
    
    # Concatenate all samples into one continuous recording (simulate recording session)
    eeg_concat = np.concatenate(samples_eeg, axis=1)  # (19, 1200) = 3 × 400
    
    # Create MNE info object
    info = create_info(
        ch_names=CHANNEL_NAMES,
        sfreq=SAMPLING_RATE,
        ch_types=['eeg'] * N_CHANNELS
    )
    
    # Set up 10-20 montage for proper electrode positions
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Create RawArray
    raw = RawArray(eeg_concat, info)
    
    # Add annotations marking sample boundaries
    onset_times = [0, 2.0, 4.0]  # 400 samples at 200 Hz = 2 seconds each
    durations = [2.0, 2.0, 2.0]
    descriptions = [
        f"Sample 0 (epileptogenic: {sample_info[0]['n_epileptogenic']})",
        f"Sample 1 (epileptogenic: {sample_info[1]['n_epileptogenic']})",
        f"Sample 2 (epileptogenic: {sample_info[2]['n_epileptogenic']})",
    ]
    
    annotations = mne.Annotations(
        onset=onset_times,
        duration=durations,
        description=descriptions
    )
    raw.set_annotations(annotations)
    
    # Print spectral info
    print("\nSpectral statistics (Welch PSD, 200-sample Hann):")
    
    for i, eeg in enumerate(samples_eeg):
        freqs, psd = welch(eeg[0], fs=SAMPLING_RATE, window='hann', nperseg=200, noverlap=100)
        
        # Band power
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 70),
        }
        total = np.trapz(psd[freqs >= 1], freqs[freqs >= 1])
        
        print(f"  Sample {i}:")
        for bname, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            bp = np.trapz(psd[mask], freqs[mask])
            pct = 100 * bp / total
            print(f"    {bname:8s}: {pct:5.1f}%", end="")
        print()
    
    # Save plot as image
    output_dir = PROJECT_ROOT / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING PLOTS")
    print("="*70)
    
    # Create a matplotlib plot of the raw EEG data
    fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(14, 16))
    fig.suptitle('Fresh Synthetic EEG Samples (0-6 seconds, integrated spectral shaping)', 
                 fontsize=14, fontweight='bold')
    
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        time_axis = np.arange(eeg_concat.shape[1]) / SAMPLING_RATE
        ax.plot(time_axis, eeg_concat[ch_idx], linewidth=0.8, color='steelblue')
        ax.set_ylabel(ch_name, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim([-150, 150])  # Standard ±150 µV display range
        
        # Add sample boundaries as vertical lines
        for t in [2.0, 4.0]:
            ax.axvline(t, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    png_path = output_dir / 'fresh_samples_eeg.png'
    print(f"\nSaving EEG timeseries plot to: {png_path}")
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    plt.close()
    fig.suptitle('Spectral Analysis of Fresh Samples (Welch PSD, 200-sample Hann)', fontsize=14)
    
    bands = {
        'delta': (1, 4, 'red'),
        'theta': (4, 8, 'blue'),
        'alpha': (8, 13, 'green'),
        'beta': (13, 30, 'orange'),
        'gamma': (30, 70, 'purple'),
    }
    
    for idx, eeg in enumerate(samples_eeg):
        ax = axes[idx]
        
        # Compute PSD averaged across all channels
        psd_list = []
        freqs = None
        for ch in range(N_CHANNELS):
            f, p = welch(eeg[ch], fs=SAMPLING_RATE, window='hann', nperseg=200, noverlap=100)
            psd_list.append(p)
            freqs = f
        psd_avg = np.mean(psd_list, axis=0)
        
        # Plot PSD with band colors
        ax.semilogy(freqs, psd_avg, 'k-', linewidth=1, alpha=0.7)
        
        # Shade frequency bands
        for bname, (low, high, color) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            ax.fill_between(freqs[mask], psd_avg[mask], alpha=0.3, color=color, label=bname)
        
        # Compute band percentages for annotation
        total = np.trapz(psd_avg[freqs >= 1], freqs[freqs >= 1])
        
        band_text = f"Sample {idx}\nEpileptic zones: {sample_info[idx]['n_epileptogenic']}\n\n"
        for bname, (low, high, _) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            bp = np.trapz(psd_avg[mask], freqs[mask])
            pct = 100 * bp / total
            band_text += f"{bname:6s}: {pct:5.1f}%\n"
        
        ax.text(0.98, 0.97, band_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_title(f'Channel Fp1 - Sample {idx}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, 70])
    
    axes[1].legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    
    psd_path = output_dir / 'fresh_samples_psd.png'
    print(f"Saving PSD plot to: {psd_path}")
    plt.savefig(psd_path, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {psd_path}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("Visual inspection complete. Plots saved to:")
    print(f"  - {png_path}")
    print(f"  - {psd_path}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
