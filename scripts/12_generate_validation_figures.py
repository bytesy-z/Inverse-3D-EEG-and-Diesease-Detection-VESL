#!/usr/bin/env python3
"""
Script: 12_generate_validation_figures.py
Purpose: Generate 4 publication-quality validation figures for PhysDeepSIF.

Figures:
  1. DLE Histogram — PhysDeepSIF vs eLORETA vs Oracle
  2. AUC vs SNR — PhysDeepSIF performance across noise levels
  3. Top-K Recall — Epileptogenic region detection
  4. Hemisphere Accuracy — Left/right classification

Usage:
    python scripts/12_generate_validation_figures.py [--num-samples 1000] [--device cpu]
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_network.metrics import compute_auc_epileptogenicity
from src.phase2_network.physdeepsif import build_physdeepsif

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

COLOR_PHYSDEEPSIF = '#1f77b4'
COLOR_ELORETA = '#ff7f0e'
COLOR_ORACLE = '#2ca02c'
COLOR_RANDOM = '#d62728'
N_REGIONS = 76
N_CHANNELS = 19
N_TIMES = 400


def setup_plotting():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def preprocess_eeg(eeg, norm_stats):
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    eeg_mean = norm_stats.get('eeg_mean_ac', norm_stats['eeg_mean'])
    eeg_std = norm_stats.get('eeg_std_ac', norm_stats['eeg_std'])
    eeg = (eeg - eeg_mean) / (eeg_std + 1e-7)
    return eeg.astype(np.float32)


def denormalize_sources(sources, norm_stats):
    sources = sources - sources.mean(axis=-1, keepdims=True)
    src_mean = norm_stats.get('src_mean_ac', norm_stats['src_mean'])
    src_std = norm_stats.get('src_std_ac', norm_stats['src_std'])
    sources = sources * (src_std + 1e-7) + src_mean
    return sources.astype(np.float32)


def eloreta_inverse(eeg, leadfield, lambda_reg=0.05):
    L = leadfield.astype(np.float64)
    n_sources = L.shape[1]
    col_norms = np.linalg.norm(L, axis=0)
    col_norms = np.maximum(col_norms, 1e-10)
    D = np.diag(1.0 / col_norms)
    LD = L @ D
    M_reg = LD @ L.T + lambda_reg * np.eye(L.shape[0])
    try:
        M_inv = np.linalg.solve(M_reg, np.eye(M_reg.shape[0]))
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M_reg)
    W = D @ L.T @ M_inv
    eeg = eeg.astype(np.float64)
    if eeg.ndim == 3:
        batch, ch, time = eeg.shape
        src = np.zeros((batch, n_sources, time), dtype=np.float64)
        for b in range(batch):
            src[b] = W @ eeg[b]
        return src.astype(np.float32)
    return (W @ eeg).astype(np.float32)


def compute_per_sample_dle(pred_sources, true_sources, region_centers, epi_mask=None):
    pred_power = np.mean(pred_sources ** 2, axis=-1)
    true_power = np.mean(true_sources ** 2, axis=-1)
    if epi_mask is not None:
        true_power_masked = true_power * epi_mask.astype(np.float32)
        no_epi = ~epi_mask.any(axis=1)
        true_power_masked[no_epi] = true_power[no_epi]
    else:
        true_power_masked = true_power
    total_pred = np.sum(pred_power, axis=1, keepdims=True)
    total_pred = np.maximum(total_pred, 1e-10)
    pred_centroid = (pred_power[:, :, np.newaxis] * region_centers[np.newaxis, :, :]).sum(axis=1) / total_pred
    total_true = np.sum(true_power_masked, axis=1, keepdims=True)
    total_true = np.maximum(total_true, 1e-10)
    true_centroid = (true_power_masked[:, :, np.newaxis] * region_centers[np.newaxis, :, :]).sum(axis=1) / total_true
    return np.linalg.norm(pred_centroid - true_centroid, axis=1)


def compute_topk_recall(sources, epi_mask, k_values):
    recalls = {}
    for k in k_values:
        correct = 0
        n_total = sources.shape[0]
        for i in range(n_total):
            power = np.mean(sources[i] ** 2, axis=-1)
            top_k_idx = np.argsort(power)[::-1][:k]
            true_epi = np.where(epi_mask[i])[0]
            if len(true_epi) > 0 and any(t in top_k_idx for t in true_epi):
                correct += 1
        recalls[k] = correct / max(n_total, 1)
    return recalls


def compute_hemisphere_accuracy(sources, epi_mask, region_labels):
    n_samples = sources.shape[0]
    left_idx = [i for i, label in enumerate(region_labels) if label.startswith('l')]
    right_idx = [i for i, label in enumerate(region_labels) if label.startswith('r')]
    n_left_epi = 0
    n_right_epi = 0
    correct_left = 0
    correct_right = 0
    for i in range(n_samples):
        true_epi_regions = np.where(epi_mask[i])[0]
        if len(true_epi_regions) == 0:
            continue
        true_left = any(r in left_idx for r in true_epi_regions)
        true_right = any(r in right_idx for r in true_epi_regions)
        if true_left and true_right:
            continue
        power = np.mean(sources[i] ** 2, axis=-1)
        max_power_region = int(np.argmax(power))
        pred_hemisphere = 'left' if max_power_region in left_idx else 'right'
        if true_left and not true_right:
            n_left_epi += 1
            if pred_hemisphere == 'left':
                correct_left += 1
        if true_right and not true_left:
            n_right_epi += 1
            if pred_hemisphere == 'right':
                correct_right += 1
    left_acc = correct_left / max(n_left_epi, 1)
    right_acc = correct_right / max(n_right_epi, 1)
    total = correct_left + correct_right
    total_n = n_left_epi + n_right_epi
    overall = total / max(total_n, 1)
    return overall, left_acc, right_acc


def main():
    parser = argparse.ArgumentParser(description='Generate validation figures')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device)
    num_samples = args.num_samples

    checkpoint_path = PROJECT_ROOT / 'outputs' / 'models' / 'checkpoint_best.pt'
    norm_stats_path = PROJECT_ROOT / 'outputs' / 'models' / 'normalization_stats.json'
    test_dataset_path = PROJECT_ROOT / 'data' / 'synthetic3' / 'test_dataset.h5'
    leadfield_path = PROJECT_ROOT / 'data' / 'leadfield_19x76.npy'
    connectivity_path = PROJECT_ROOT / 'data' / 'connectivity_76.npy'
    region_centers_path = PROJECT_ROOT / 'data' / 'region_centers_76.npy'
    region_labels_path = PROJECT_ROOT / 'data' / 'region_labels_76.json'
    figures_dir = PROJECT_ROOT / 'outputs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print('Loading data...')
    with h5py.File(str(test_dataset_path), 'r') as f:
        total = f['eeg'].shape[0]
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(total, size=min(num_samples, total), replace=False))
        eeg_test = f['eeg'][idx].astype(np.float32)
        src_test = f['source_activity'][idx].astype(np.float32)
        epi_mask = f['epileptogenic_mask'][idx]

    leadfield = np.load(str(leadfield_path)).astype(np.float32)
    region_centers = np.load(str(region_centers_path)).astype(np.float32)
    with open(region_labels_path) as f:
        region_labels = json.load(f)
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    actual_n = len(idx)
    print(f'Loaded {actual_n} test samples, EEG: {eeg_test.shape}, sources: {src_test.shape}')

    # ── Build model ────────────────────────────────────────────────────
    print('Building PhysDeepSIF model...')
    model = build_physdeepsif(str(leadfield_path), str(connectivity_path), lstm_hidden_size=76)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint (epoch {checkpoint["epoch"]})')

    # ── Preprocess and run inference ───────────────────────────────────
    print('Running PhysDeepSIF inference...')
    eeg_norm = preprocess_eeg(eeg_test, norm_stats)

    pred_sources_list = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, actual_n, batch_size):
            batch = torch.from_numpy(eeg_norm[i:i + batch_size]).to(device)
            pred_sources_list.append(model(batch).cpu().numpy())
    pred_sources = np.concatenate(pred_sources_list, axis=0)
    pred_sources = denormalize_sources(pred_sources, norm_stats)
    src_test_proc = denormalize_sources(src_test.copy(), norm_stats)

    # ── eLORETA ────────────────────────────────────────────────────────
    print('Running eLORETA...')
    eloreta_sources = eloreta_inverse(eeg_test, leadfield)

    # ── Oracle (true sources as predictions for DLE) ───────────────────
    print('Computing metrics...')

    # ── Figure 1: DLE Histogram ────────────────────────────────────────
    print('Generating Figure 1: DLE Histogram...')
    dle_physdeepsif_per = compute_per_sample_dle(pred_sources, src_test_proc, region_centers, epi_mask)
    dle_eloreta_per = compute_per_sample_dle(eloreta_sources.astype(np.float32), src_test_proc, region_centers, epi_mask)
    dle_oracle_per = compute_per_sample_dle(src_test_proc, src_test_proc, region_centers, epi_mask)

    print(f'  DLE mean±std — PhysDeepSIF: {np.mean(dle_physdeepsif_per):.2f}±{np.std(dle_physdeepsif_per):.2f} mm')
    print(f'  DLE mean±std — eLORETA:     {np.mean(dle_eloreta_per):.2f}±{np.std(dle_eloreta_per):.2f} mm')
    print(f'  DLE mean±std — Oracle:      {np.mean(dle_oracle_per):.2f}±{np.std(dle_oracle_per):.2f} mm')

    setup_plotting()
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    all_dle = np.concatenate([dle_physdeepsif_per, dle_eloreta_per, dle_oracle_per])
    bins = np.linspace(0, np.percentile(all_dle, 99), 50)

    ax1.hist(dle_physdeepsif_per, bins=bins, alpha=0.5, color=COLOR_PHYSDEEPSIF, label='PhysDeepSIF')
    ax1.hist(dle_eloreta_per, bins=bins, alpha=0.5, color=COLOR_ELORETA, label='eLORETA')
    ax1.hist(dle_oracle_per, bins=bins, alpha=0.5, color=COLOR_ORACLE, label='Oracle')

    ylim = ax1.get_ylim()
    for vals, color, name in [
        (dle_physdeepsif_per, COLOR_PHYSDEEPSIF, 'PhysDeepSIF'),
        (dle_eloreta_per, COLOR_ELORETA, 'eLORETA'),
        (dle_oracle_per, COLOR_ORACLE, 'Oracle'),
    ]:
        mean_val = np.mean(vals)
        ax1.axvline(mean_val, color=color, linestyle='--', linewidth=1.5)
        ax1.text(mean_val, ylim[1] * 0.95, f'{name}: {mean_val:.1f}mm',
                 color=color, fontsize=9, rotation=90, va='top', ha='right')

    ax1.set_xlabel('DLE (mm)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Dipole Localization Error Distribution (Test Set, N={actual_n})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(str(figures_dir / 'dle_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print('  Saved dle_histogram.png')

    # ── Figure 2: AUC vs SNR ──────────────────────────────────────────
    print('Generating Figure 2: AUC vs SNR...')
    snr_levels = [5, 10, 15, 20, 30]
    auc_snr = []

    for snr_db in snr_levels:
        eeg_noisy = eeg_test.copy()
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_rng = np.random.RandomState(42 + int(snr_db))
        for b in range(actual_n):
            signal_power = np.mean(eeg_noisy[b] ** 2)
            noise_std = np.sqrt(signal_power / snr_linear)
            eeg_noisy[b] += noise_rng.randn(*eeg_noisy[b].shape).astype(np.float32) * noise_std

        eeg_noisy_norm = preprocess_eeg(eeg_noisy, norm_stats)

        pred_noisy_list = []
        with torch.no_grad():
            for i in range(0, actual_n, batch_size):
                batch = torch.from_numpy(eeg_noisy_norm[i:i + batch_size]).to(device)
                pred_noisy_list.append(model(batch).cpu().numpy())
        pred_noisy = np.concatenate(pred_noisy_list, axis=0)
        pred_noisy = denormalize_sources(pred_noisy, norm_stats)

        auc_val = compute_auc_epileptogenicity(pred_noisy, epi_mask)
        auc_snr.append(auc_val)
        print(f'  SNR={snr_db:2d} dB: AUC={auc_val:.4f}')

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(snr_levels, auc_snr, 'o-', color=COLOR_PHYSDEEPSIF, linewidth=2, markersize=8, label='PhysDeepSIF')
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (AUC=0.5)')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC vs Input SNR (PhysDeepSIF)')
    ax2.set_ylim(0.4, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(str(figures_dir / 'auc_vs_snr.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print('  Saved auc_vs_snr.png')

    # ── Figure 3: Top-K Recall ────────────────────────────────────────
    print('Generating Figure 3: Top-K Recall...')
    k_values = [1, 2, 3, 5, 7, 10]

    recall_physdeepsif = compute_topk_recall(pred_sources, epi_mask, k_values)
    recall_eloreta = compute_topk_recall(eloreta_sources.astype(np.float32), epi_mask, k_values)

    recall_random = {}
    rand_rng = np.random.RandomState(42)
    for k in k_values:
        correct = 0
        for i in range(actual_n):
            shuffled_idx = rand_rng.permutation(N_REGIONS)
            top_k_idx = np.sort(shuffled_idx[:k])
            true_epi = np.where(epi_mask[i])[0]
            if len(true_epi) > 0 and np.any(np.in1d(true_epi, top_k_idx)):
                correct += 1
        recall_random[k] = correct / max(actual_n, 1)

    print('  Recall@K — PhysDeepSIF / eLORETA / Random:')
    for k in k_values:
        print(f'    K={k}: {recall_physdeepsif[k]:.3f} / {recall_eloreta[k]:.3f} / {recall_random[k]:.3f}')

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(k_values, [recall_physdeepsif[k] for k in k_values], 'o-',
             color=COLOR_PHYSDEEPSIF, linewidth=2, markersize=8, label='PhysDeepSIF')
    ax3.plot(k_values, [recall_eloreta[k] for k in k_values], 's--',
             color=COLOR_ELORETA, linewidth=2, markersize=8, label='eLORETA')
    ax3.plot(k_values, [recall_random[k] for k in k_values], '^:',
             color=COLOR_RANDOM, linewidth=2, markersize=8, label='Random')

    ax3.set_xlabel('K')
    ax3.set_ylabel('Recall')
    ax3.set_title('Top-K Recall (Epileptogenic Region Detection)')
    ax3.set_xticks(k_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(str(figures_dir / 'topk_recall.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print('  Saved topk_recall.png')

    # ── Figure 4: Hemisphere Accuracy ─────────────────────────────────
    print('Generating Figure 4: Hemisphere Accuracy...')

    _, left_acc_phys, right_acc_phys = compute_hemisphere_accuracy(
        pred_sources, epi_mask, region_labels)
    _, left_acc_elor, right_acc_elor = compute_hemisphere_accuracy(
        eloreta_sources.astype(np.float32), epi_mask, region_labels)
    random_sources = np.random.RandomState(42).randn(*pred_sources.shape).astype(np.float32)
    _, left_acc_rand, right_acc_rand = compute_hemisphere_accuracy(
        random_sources, epi_mask, region_labels)

    print(f'  Left accuracy  — PhysDeepSIF: {left_acc_phys:.3f}, eLORETA: {left_acc_elor:.3f}, Random: {left_acc_rand:.3f}')
    print(f'  Right accuracy — PhysDeepSIF: {right_acc_phys:.3f}, eLORETA: {right_acc_elor:.3f}, Random: {right_acc_rand:.3f}')

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.3

    left_bars = ax4.bar(x - width / 2,
                        [left_acc_phys, left_acc_elor, left_acc_rand],
                        width, color=[COLOR_PHYSDEEPSIF, COLOR_ELORETA, COLOR_RANDOM],
                        alpha=0.85, label='Left Hemisphere', edgecolor='white', linewidth=0.5)
    right_bars = ax4.bar(x + width / 2,
                         [right_acc_phys, right_acc_elor, right_acc_rand],
                         width, color=[COLOR_PHYSDEEPSIF, COLOR_ELORETA, COLOR_RANDOM],
                         alpha=0.45, label='Right Hemisphere', edgecolor='white', linewidth=0.5, hatch='///')

    ax4.set_xticks(x)
    ax4.set_xticklabels(['PhysDeepSIF', 'eLORETA', 'Random'])
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Hemisphere Classification Accuracy')
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3, axis='y')

    fig4.tight_layout()
    fig4.savefig(str(figures_dir / 'hemisphere_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print('  Saved hemisphere_accuracy.png')

    print(f'\nDone. All 4 figures saved to {figures_dir}/')


if __name__ == '__main__':
    main()
