#!/usr/bin/env python3
"""
Script: 10_final_validation.py
Phase: 1 - Forward Modeling and Synthetic Data Generation
Purpose: Comprehensive final validation of the synthetic EEG generation pipeline.

Runs THREE validation suites on freshly-generated EEG windows:
  1. Legacy biophysical checks (13 metrics: RMS, peak-to-peak, 1/f exponent, SEF95,
     gamma%, AC lag-1, Hjorth mobility/complexity, kurtosis, skewness,
     zero-crossing rate, GFP, envelope mean)
  2. Spatial-spectral gradient checks (group-level alpha/beta monotonic gradients,
     PDR range [1.3, 5.0])
  3. Per-group frequency distribution comparison against clinical norms

Generates 10 TVB simulations (yielding ~36 valid windows at ~72% pass rate),
computes all metrics, and prints a pass/fail summary table.

Usage:
    python scripts/10_final_validation.py

Expected runtime: ~3-5 minutes (10 simulations)

See /docs/02_TECHNICAL_SPECIFICATIONS.md Phase 1.5 and 1.6 for full specifications.
"""

# Standard library imports
import json
import logging
import sys
import warnings
from pathlib import Path

# Third-party imports
import numpy as np
import yaml
from scipy.signal import welch

# Suppress TVB/numba/scipy runtime warnings that clutter output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# Project setup
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (after path setup)
from src.phase1_forward.source_space import load_source_space_data
from src.phase1_forward.leadfield_builder import load_leadfield
from src.phase1_forward.synthetic_dataset import generate_one_simulation

# Configure logging to only show warnings and above (keep output clean)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# Constants
# ============================================================================
N_CHANNELS = 19
N_REGIONS = 76
WINDOW_LENGTH = 400
FS = 200.0  # Sampling rate in Hz

# Standard 10-20 channel names (order must match leadfield columns)
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Anteroposterior channel groups for spatial-spectral validation
# Order: anterior → posterior (matching clinical convention)
CHANNEL_GROUPS = {
    'frontal_fp': ['Fp1', 'Fp2'],
    'frontal_f': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'central_c': ['C3', 'C4', 'T3', 'T4', 'Cz'],
    'parietal_p': ['P3', 'P4', 'T5', 'T6', 'Pz'],
    'occipital_o': ['O1', 'O2'],
}

# Frequency bands (Hz) for spectral analysis
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70),
}

# Clinical target band power percentages (global average)
# Based on Niedermeyer (2005), Tatum (2014), and clinical consensus
CLINICAL_TARGETS = {
    'delta': 10.0,
    'theta': 10.0,
    'alpha': 30.0,
    'beta': 22.0,
    'gamma': 8.0,
}

# Maximum acceptable deviation from clinical targets (percentage points)
MAX_BAND_DEVIATION = 15.0

# Number of simulations to run for validation
N_SIMULATIONS = 10


# ============================================================================
# Helper functions
# ============================================================================
def get_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple
) -> float:
    """
    Compute the integrated power within a frequency range using trapezoidal rule.

    Args:
        freqs: Frequency axis from Welch PSD (Hz)
        psd: Power spectral density values
        freq_range: (low_hz, high_hz) tuple defining the band

    Returns:
        Integrated power in the band (units: µV² if PSD is in µV²/Hz)
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if np.sum(mask) < 2:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_welch_psd(
    signal_1d: np.ndarray,
    fs: float = FS
) -> tuple:
    """
    Compute Welch PSD matching the generation STFT parameters.

    Uses 200-sample Hann window with 50% overlap, matching exactly the
    STFT parameters used in apply_spectral_shaping().

    Args:
        signal_1d: 1D time series (400 samples at 200 Hz)
        fs: Sampling rate

    Returns:
        (freqs, psd) arrays
    """
    freqs, psd = welch(
        signal_1d, fs=fs, window='hann', nperseg=200, noverlap=100
    )
    return freqs, psd


# ============================================================================
# Suite 1: Legacy biophysical checks (13 metrics)
# ============================================================================
def run_legacy_checks(eeg: np.ndarray) -> dict:
    """
    Run the 13 legacy biophysical validation metrics on EEG samples.

    These metrics were established during Phase 1.5 and ensure the synthetic
    EEG has realistic amplitude, temporal structure, and spectral content.

    Args:
        eeg: ndarray (n_samples, 19, 400) — synthetic EEG windows in µV

    Returns:
        dict with metric name → {'value': float, 'range': (min, max), 'passed': bool}
    """
    n_samples = eeg.shape[0]
    results = {}

    # ---- 1. RMS Amplitude ----
    # Root-mean-square voltage averaged across channels and samples
    rms_per_sample = [np.sqrt(np.mean(eeg[s] ** 2)) for s in range(n_samples)]
    mean_rms = float(np.mean(rms_per_sample))
    results['rms_amplitude'] = {
        'value': mean_rms, 'unit': 'µV',
        'range': (5, 150), 'passed': 5 <= mean_rms <= 150
    }

    # ---- 2. Peak-to-Peak ----
    # Average maximum peak-to-peak voltage across channels and samples
    ptp_per_sample = [
        np.mean(np.ptp(eeg[s], axis=1)) for s in range(n_samples)
    ]
    mean_ptp = float(np.mean(ptp_per_sample))
    results['peak_to_peak'] = {
        'value': mean_ptp, 'unit': 'µV',
        'range': (10, 500), 'passed': 10 <= mean_ptp <= 500
    }

    # ---- 3. 1/f Exponent ----
    # Slope of log-log PSD in 2-40 Hz range, averaged across channels/samples
    exponents = []
    for s in range(min(n_samples, 50)):  # Cap at 50 for speed
        for ch in range(N_CHANNELS):
            freqs, psd = compute_welch_psd(eeg[s, ch])
            mask = (freqs >= 2) & (freqs <= 40)
            if np.sum(mask) > 2 and np.all(psd[mask] > 0):
                log_f = np.log10(freqs[mask])
                log_p = np.log10(psd[mask])
                slope = np.polyfit(log_f, log_p, 1)[0]
                exponents.append(-slope)  # 1/f^beta convention: negative slope
    mean_exp = float(np.mean(exponents)) if exponents else 0.0
    results['one_over_f_exponent'] = {
        'value': mean_exp, 'unit': '',
        'range': (0.5, 3.0), 'passed': 0.5 <= mean_exp <= 3.0
    }

    # ---- 4. SEF95 ----
    # Spectral Edge Frequency at 95% of total power (1-100 Hz)
    sef95_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            freqs, psd = compute_welch_psd(eeg[s, ch])
            mask = freqs >= 1
            cumpower = np.cumsum(psd[mask])
            if cumpower[-1] > 0:
                threshold = 0.95 * cumpower[-1]
                idx = np.searchsorted(cumpower, threshold)
                sef95_vals.append(freqs[mask][min(idx, len(freqs[mask]) - 1)])
    mean_sef95 = float(np.mean(sef95_vals)) if sef95_vals else 0.0
    results['sef95'] = {
        'value': mean_sef95, 'unit': 'Hz',
        'range': (5, 60), 'passed': 5 <= mean_sef95 <= 60
    }

    # ---- 5. Gamma Power ----
    # Fraction of total power (1-100 Hz) in gamma band (30-70 Hz)
    gamma_pcts = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            freqs, psd = compute_welch_psd(eeg[s, ch])
            total = get_band_power(freqs, psd, (1, 100))
            gamma = get_band_power(freqs, psd, (30, 70))
            if total > 0:
                gamma_pcts.append(100 * gamma / total)
    mean_gamma = float(np.mean(gamma_pcts)) if gamma_pcts else 0.0
    results['gamma_power'] = {
        'value': mean_gamma, 'unit': '%',
        'range': (0, 15), 'passed': mean_gamma < 15
    }

    # ---- 6. Autocorrelation Lag-1 ----
    # Temporal smoothness indicator; high AC1 means smooth signal (not white noise)
    ac1_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            x = eeg[s, ch]
            ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            ac1_vals.append(ac1)
    mean_ac1 = float(np.mean(ac1_vals))
    results['ac_lag1'] = {
        'value': mean_ac1, 'unit': '',
        'range': (0.5, 1.0), 'passed': mean_ac1 > 0.5
    }

    # ---- 7. Hjorth Mobility ----
    # sqrt(var(dx/dt) / var(x)) — indicates dominant frequency content
    mobility_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            x = eeg[s, ch]
            dx = np.diff(x)
            if np.var(x) > 0:
                mobility_vals.append(np.sqrt(np.var(dx) / np.var(x)))
    mean_mob = float(np.mean(mobility_vals)) if mobility_vals else 0.0
    results['hjorth_mobility'] = {
        'value': mean_mob, 'unit': '',
        'range': (0.01, 0.5), 'passed': 0.01 <= mean_mob <= 0.5
    }

    # ---- 8. Hjorth Complexity ----
    # Mobility(dx/dt) / Mobility(x) — indicates signal complexity
    complexity_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            x = eeg[s, ch]
            dx = np.diff(x)
            ddx = np.diff(dx)
            var_x = np.var(x)
            var_dx = np.var(dx)
            var_ddx = np.var(ddx)
            if var_x > 0 and var_dx > 0:
                mob_x = np.sqrt(var_dx / var_x)
                mob_dx = np.sqrt(var_ddx / var_dx)
                complexity_vals.append(mob_dx / mob_x)
    mean_comp = float(np.mean(complexity_vals)) if complexity_vals else 0.0
    results['hjorth_complexity'] = {
        'value': mean_comp, 'unit': '',
        'range': (0.5, 5.0), 'passed': 0.5 <= mean_comp <= 5.0
    }

    # ---- 9. Kurtosis ----
    # Measures tailedness of amplitude distribution; EEG typically near-Gaussian
    from scipy.stats import kurtosis as scipy_kurtosis
    kurt_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            kurt_vals.append(scipy_kurtosis(eeg[s, ch], fisher=True))
    mean_kurt = float(np.mean(kurt_vals))
    results['kurtosis'] = {
        'value': mean_kurt, 'unit': '',
        'range': (-2, 5), 'passed': -2 <= mean_kurt <= 5
    }

    # ---- 10. Skewness ----
    # Symmetry of amplitude distribution; clinical EEG is roughly symmetric
    from scipy.stats import skew as scipy_skew
    skew_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            skew_vals.append(scipy_skew(eeg[s, ch]))
    mean_skew = float(np.mean(skew_vals))
    results['skewness'] = {
        'value': mean_skew, 'unit': '',
        'range': (-1, 1), 'passed': abs(mean_skew) < 1
    }

    # ---- 11. Zero-Crossing Rate ----
    # Number of sign changes per second; indicates dominant frequency content
    zcr_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            x = eeg[s, ch] - np.mean(eeg[s, ch])
            sign_changes = np.sum(np.diff(np.sign(x)) != 0)
            zcr_hz = sign_changes / 2.0 / (WINDOW_LENGTH / FS)  # crossings / 2 = cycles
            zcr_vals.append(zcr_hz)
    mean_zcr = float(np.mean(zcr_vals))
    results['zero_crossing_rate'] = {
        'value': mean_zcr, 'unit': 'Hz',
        'range': (2, 50), 'passed': 2 <= mean_zcr <= 50
    }

    # ---- 12. Global Field Power (GFP) ----
    # Spatial std across channels at each time point, then averaged
    gfp_vals = []
    for s in range(n_samples):
        gfp = np.std(eeg[s], axis=0)  # std across 19 channels at each time point
        gfp_vals.append(np.mean(gfp))
    mean_gfp = float(np.mean(gfp_vals))
    results['global_field_power'] = {
        'value': mean_gfp, 'unit': 'µV',
        'range': (1, 100), 'passed': 1 <= mean_gfp <= 100
    }

    # ---- 13. Envelope Mean ----
    # Mean of Hilbert envelope (analytic signal) across channels and samples
    from scipy.signal import hilbert
    env_vals = []
    for s in range(min(n_samples, 50)):
        for ch in range(N_CHANNELS):
            analytic = hilbert(eeg[s, ch])
            env_vals.append(np.mean(np.abs(analytic)))
    mean_env = float(np.mean(env_vals))
    results['envelope_mean'] = {
        'value': mean_env, 'unit': 'µV',
        'range': (2, 200), 'passed': 2 <= mean_env <= 200
    }

    return results


# ============================================================================
# Suite 2: Spatial-spectral gradient checks
# ============================================================================
def run_gradient_checks(eeg: np.ndarray) -> dict:
    """
    Validate anteroposterior alpha/beta gradients and PDR.

    Checks that:
    - Alpha power increases monotonically: Fp < F < C < P < O (group-level)
    - Beta power decreases monotonically: Fp > F > C > P > O (group-level)
    - PDR (occipital/frontal alpha ratio) is within [1.3, 5.0]

    Args:
        eeg: ndarray (n_samples, 19, 400) — synthetic EEG windows in µV

    Returns:
        dict with gradient metrics and pass/fail
    """
    n_samples = eeg.shape[0]
    group_order = ['frontal_fp', 'frontal_f', 'central_c', 'parietal_p', 'occipital_o']

    # Compute per-channel PSD, averaged across all samples
    ch_psd = {}
    freqs = None
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        psd_list = []
        for s in range(n_samples):
            f, p = compute_welch_psd(eeg[s, ch_idx])
            psd_list.append(p)
            freqs = f
        ch_psd[ch_name] = np.mean(psd_list, axis=0)

    # Compute per-group band power percentages
    group_alpha = {}
    group_beta = {}
    group_band_pct = {}  # group -> {band: pct}

    for gname in group_order:
        gchannels = CHANNEL_GROUPS[gname]
        gpsd = np.mean([ch_psd[ch] for ch in gchannels], axis=0)
        gtotal = get_band_power(freqs, gpsd, (1, 70))
        if gtotal <= 0:
            gtotal = 1e-10

        band_pcts = {}
        for bname, brange in BANDS.items():
            bp = get_band_power(freqs, gpsd, brange)
            band_pcts[bname] = 100 * bp / gtotal

        group_band_pct[gname] = band_pcts
        group_alpha[gname] = band_pcts['alpha']
        group_beta[gname] = band_pcts['beta']

    # Extract ordered values for gradient checks
    alpha_vals = [group_alpha[g] for g in group_order]
    beta_vals = [group_beta[g] for g in group_order]

    # Check monotonicity
    alpha_monotonic = all(
        a2 > a1 for a1, a2 in zip(alpha_vals[:-1], alpha_vals[1:])
    )
    beta_monotonic = all(
        b2 < b1 for b1, b2 in zip(beta_vals[:-1], beta_vals[1:])
    )

    # PDR: occipital alpha / frontal_fp alpha
    pdr = alpha_vals[-1] / alpha_vals[0] if alpha_vals[0] > 0 else 0
    pdr_pass = 1.3 <= pdr <= 5.0

    results = {
        'alpha_gradient': {
            'values': {g: f"{v:.1f}%" for g, v in zip(group_order, alpha_vals)},
            'monotonic': alpha_monotonic,
            'passed': alpha_monotonic,
        },
        'beta_gradient': {
            'values': {g: f"{v:.1f}%" for g, v in zip(group_order, beta_vals)},
            'monotonic': beta_monotonic,
            'passed': beta_monotonic,
        },
        'pdr': {
            'value': pdr,
            'range': (1.3, 5.0),
            'passed': pdr_pass,
        },
        'group_band_pct': group_band_pct,
    }

    return results


# ============================================================================
# Suite 3: Per-group clinical frequency distribution comparison
# ============================================================================
def run_clinical_comparison(eeg: np.ndarray) -> dict:
    """
    Compare global-average band power distribution against clinical norms.

    Computes the band power distribution averaged across ALL channels and samples,
    then checks that each band is within MAX_BAND_DEVIATION (15 pp) of clinical
    target values.

    Args:
        eeg: ndarray (n_samples, 19, 400)

    Returns:
        dict with per-band comparison metrics
    """
    n_samples = eeg.shape[0]

    # Compute global average PSD across all channels and samples
    all_psd = []
    freqs = None
    for s in range(n_samples):
        for ch in range(N_CHANNELS):
            f, p = compute_welch_psd(eeg[s, ch])
            all_psd.append(p)
            freqs = f
    global_psd = np.mean(all_psd, axis=0)

    total_power = get_band_power(freqs, global_psd, (1, 70))
    if total_power <= 0:
        total_power = 1e-10

    results = {}
    all_pass = True
    for bname, brange in BANDS.items():
        bp = get_band_power(freqs, global_psd, brange)
        pct = 100 * bp / total_power
        target = CLINICAL_TARGETS[bname]
        diff = pct - target
        passed = abs(diff) <= MAX_BAND_DEVIATION
        if not passed:
            all_pass = False
        results[bname] = {
            'synthetic_pct': pct,
            'clinical_pct': target,
            'difference': diff,
            'passed': passed,
        }

    results['overall_pass'] = all_pass
    return results


# ============================================================================
# Main execution
# ============================================================================
def main():
    """Run all three validation suites and print results."""

    # Load project configuration and data
    config = yaml.safe_load(open(PROJECT_ROOT / 'config.yaml'))
    data_dir = str(PROJECT_ROOT / 'data')
    connectivity, region_centers, region_labels, tract_lengths = (
        load_source_space_data(data_dir)
    )
    leadfield = load_leadfield(
        str(PROJECT_ROOT / 'data/leadfield_19x76.npy')
    )

    # ====================================================================
    # Phase 0: Generate EEG samples through the full pipeline
    # ====================================================================
    print(f"Generating {N_SIMULATIONS} TVB simulations through full pipeline...")
    print("(TVB → decimation → leadfield → noise → skull LP → spectral shaping → validation)")
    print()

    all_eeg = []
    n_total_windows = 0
    n_valid_windows = 0

    for sim_idx in range(N_SIMULATIONS):
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
        n_total_windows += 5
        if result is not None:
            n_w = result['eeg'].shape[0]
            n_valid_windows += n_w
            for w in range(n_w):
                all_eeg.append(result['eeg'][w])
            n_epi = int(result['epileptogenic_mask'][0].sum())
            print(f"  Sim {sim_idx:2d}: {n_w}/5 valid windows, "
                  f"{n_epi} epileptogenic regions")
        else:
            print(f"  Sim {sim_idx:2d}: FAILED (diverged or all windows rejected)")

    if not all_eeg:
        print("\nERROR: No valid EEG windows generated. Cannot validate.")
        sys.exit(1)

    eeg = np.array(all_eeg, dtype=np.float32)
    pass_rate = 100 * n_valid_windows / max(1, n_total_windows)

    print(f"\nGenerated: {n_valid_windows}/{n_total_windows} valid windows "
          f"({pass_rate:.0f}% pass rate)")
    print(f"EEG shape: {eeg.shape}")

    # ====================================================================
    # Suite 1: Legacy biophysical checks
    # ====================================================================
    print(f"\n{'='*72}")
    print("SUITE 1: LEGACY BIOPHYSICAL CHECKS (13 metrics)")
    print(f"{'='*72}")

    legacy = run_legacy_checks(eeg)
    n_legacy_pass = 0
    n_legacy_total = len(legacy)

    print(f"\n{'Metric':<25s} {'Value':>12s} {'Range':>20s} {'Status':>8s}")
    print(f"{'-'*65}")
    for name, info in legacy.items():
        val_str = f"{info['value']:.2f} {info.get('unit', '')}"
        range_str = f"[{info['range'][0]}, {info['range'][1]}]"
        status = "PASS" if info['passed'] else "** FAIL **"
        if info['passed']:
            n_legacy_pass += 1
        print(f"  {name:<23s} {val_str:>12s} {range_str:>20s} {status:>8s}")

    print(f"\nLegacy: {n_legacy_pass}/{n_legacy_total} PASS")

    # ====================================================================
    # Suite 2: Spatial-spectral gradient checks
    # ====================================================================
    print(f"\n{'='*72}")
    print("SUITE 2: SPATIAL-SPECTRAL GRADIENT CHECKS")
    print(f"{'='*72}")

    gradients = run_gradient_checks(eeg)

    # Alpha gradient
    alpha_info = gradients['alpha_gradient']
    alpha_vals_str = " < ".join(alpha_info['values'].values())
    print(f"\n  Alpha gradient (Fp→O): {alpha_vals_str}")
    print(f"  Alpha monotonic:       {'PASS' if alpha_info['passed'] else '** FAIL **'}")

    # Beta gradient
    beta_info = gradients['beta_gradient']
    beta_vals_str = " > ".join(beta_info['values'].values())
    print(f"  Beta gradient  (Fp→O): {beta_vals_str}")
    print(f"  Beta monotonic:        {'PASS' if beta_info['passed'] else '** FAIL **'}")

    # PDR
    pdr_info = gradients['pdr']
    print(f"  PDR (O/Fp alpha):      {pdr_info['value']:.2f} "
          f"(target: [{pdr_info['range'][0]}, {pdr_info['range'][1]}]) "
          f"{'PASS' if pdr_info['passed'] else '** FAIL **'}")

    n_gradient_pass = sum([
        alpha_info['passed'], beta_info['passed'], pdr_info['passed']
    ])
    print(f"\nGradients: {n_gradient_pass}/3 PASS")

    # Per-group band table
    print(f"\n  Per-group band distribution:")
    print(f"  {'Group':<15s} {'Delta':>8s} {'Theta':>8s} {'Alpha':>8s} "
          f"{'Beta':>8s} {'Gamma':>8s}")
    print(f"  {'-'*55}")
    for gname in ['frontal_fp', 'frontal_f', 'central_c',
                   'parietal_p', 'occipital_o']:
        bpct = gradients['group_band_pct'][gname]
        print(f"  {gname:<15s} {bpct['delta']:7.1f}% {bpct['theta']:7.1f}% "
              f"{bpct['alpha']:7.1f}% {bpct['beta']:7.1f}% "
              f"{bpct['gamma']:7.1f}%")

    # ====================================================================
    # Suite 3: Clinical frequency distribution comparison
    # ====================================================================
    print(f"\n{'='*72}")
    print("SUITE 3: CLINICAL FREQUENCY DISTRIBUTION COMPARISON")
    print(f"{'='*72}")

    clinical = run_clinical_comparison(eeg)

    print(f"\n  {'Band':<15s} {'Synthetic':>10s} {'Clinical':>10s} "
          f"{'Diff':>10s} {'Status':>10s}")
    print(f"  {'-'*55}")
    n_clinical_pass = 0
    n_clinical_total = 0
    for bname in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        info = clinical[bname]
        status = "PASS" if info['passed'] else "** FAIL **"
        if info['passed']:
            n_clinical_pass += 1
        n_clinical_total += 1
        print(f"  {bname:<15s} {info['synthetic_pct']:9.1f}% "
              f"{info['clinical_pct']:9.1f}% "
              f"{info['difference']:+9.1f}% {status:>10s}")

    print(f"\nClinical: {n_clinical_pass}/{n_clinical_total} PASS")

    # ====================================================================
    # Final Summary
    # ====================================================================
    total_pass = n_legacy_pass + n_gradient_pass + n_clinical_pass
    total_checks = n_legacy_total + 3 + n_clinical_total

    print(f"\n{'='*72}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Pipeline pass rate:     {pass_rate:.0f}% "
          f"({n_valid_windows}/{n_total_windows} windows)")
    print(f"  Legacy biophysical:     {n_legacy_pass}/{n_legacy_total} PASS")
    print(f"  Spatial gradients:      {n_gradient_pass}/3 PASS")
    print(f"  Clinical comparison:    {n_clinical_pass}/{n_clinical_total} PASS")
    print(f"  {'─'*40}")
    print(f"  TOTAL:                  {total_pass}/{total_checks} PASS")

    overall = total_pass == total_checks
    print(f"\n  OVERALL RESULT: {'✅ ALL CHECKS PASSED' if overall else '❌ SOME CHECKS FAILED'}")
    print()

    # Save results to JSON
    output_path = PROJECT_ROOT / 'outputs' / 'final_validation_report.json'
    report = {
        'pipeline_pass_rate': pass_rate,
        'n_samples': int(eeg.shape[0]),
        'n_simulations': N_SIMULATIONS,
        'legacy_checks': {
            k: {'value': v['value'], 'passed': v['passed']}
            for k, v in legacy.items()
        },
        'gradient_checks': {
            'alpha_monotonic': alpha_info['passed'],
            'beta_monotonic': beta_info['passed'],
            'pdr': pdr_info['value'],
            'pdr_passed': pdr_info['passed'],
        },
        'clinical_comparison': {
            k: {
                'synthetic_pct': v['synthetic_pct'],
                'clinical_pct': v['clinical_pct'],
                'passed': v['passed'],
            }
            for k, v in clinical.items() if k != 'overall_pass'
        },
        'total_pass': total_pass,
        'total_checks': total_checks,
        'overall_pass': overall,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {output_path}")

    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
