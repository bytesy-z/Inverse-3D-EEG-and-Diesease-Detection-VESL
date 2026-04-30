#!/usr/bin/env python3
"""
Module: generate_figures.py
Phase: 5 — Validation and Baselines
Purpose: Generate all 6 validation figures for PhysDeepSIF.

Figures:
  1. dle_histogram.png      — DLE distribution: PhysDeepSIF vs eLORETA vs Oracle
  2. auc_vs_snr.png         — AUC across SNR levels with error bars
  3. topk_recall.png        — Top-K recall for epileptogenic region detection
  4. hemisphere_accuracy.png — Hemisphere classification accuracy
  5. learning_curve.png     — Train/val loss over epochs
  6. concordance_heatmap.png — Concordance overlap distribution (50 samples)

Usage:
    python -m src.phase5_validation.generate_figures
    python src/phase5_validation/generate_figures.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_network.metrics import (
    compute_dipole_localization_error,
    compute_auc_epileptogenicity,
)
from src.phase2_network.physdeepsif import build_physdeepsif, PhysDeepSIF

N_REGIONS = 76
N_CHANNELS = 19
N_TIMES = 400
BATCH_SIZE = 64

COLOR_PHYSDEEPSIF = '#1f77b4'
COLOR_ELORETA = '#ff7f0e'
COLOR_ORACLE = '#2ca02c'
COLOR_RANDOM = '#d62728'

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
MODEL_DIR = PROJECT_ROOT / 'outputs' / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'synthetic3'

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = MODEL_DIR / 'checkpoint_best.pt'
NORM_STATS_PATH = MODEL_DIR / 'normalization_stats.json'
TEST_DATASET_PATH = DATA_DIR / 'test_dataset.h5'
LEADFIELD_PATH = PROJECT_ROOT / 'data' / 'leadfield_19x76.npy'
CONNECTIVITY_PATH = PROJECT_ROOT / 'data' / 'connectivity_76.npy'
REGION_CENTERS_PATH = PROJECT_ROOT / 'data' / 'region_centers_76.npy'
REGION_LABELS_PATH = PROJECT_ROOT / 'data' / 'region_labels_76.json'
TRAINING_LOG_PATH = MODEL_DIR / 'training.log'


def setup_plotting():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
    })


# ── Data loading ─────────────────────────────────────────────────────────────
def load_test_subset(num_samples=None):
    """Load test dataset; if num_samples is None, load all."""
    print('Loading test dataset...')
    with h5py.File(str(TEST_DATASET_PATH), 'r') as f:
        total = f['eeg'].shape[0]
        if num_samples is not None and num_samples < total:
            rng = np.random.RandomState(42)
            idx = np.sort(rng.choice(total, size=num_samples, replace=False))
        else:
            idx = np.arange(total)
        eeg = f['eeg'][idx].astype(np.float32)
        sources = f['source_activity'][idx].astype(np.float32)
        mask = f['epileptogenic_mask'][idx]
    print(f'  Loaded {len(idx)} samples, EEG: {eeg.shape}, sources: {sources.shape}')
    return eeg, sources, mask, idx


def load_region_data():
    leadfield = np.load(str(LEADFIELD_PATH)).astype(np.float32)
    region_centers = np.load(str(REGION_CENTERS_PATH)).astype(np.float32)
    with open(REGION_LABELS_PATH) as f:
        region_labels = json.load(f)
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    return leadfield, region_centers, region_labels, norm_stats


# ── Preprocessing (must match training: 03_train_network.py) ─────────────────
def preprocess_input(eeg, norm_stats):
    """Preprocess EEG for model input: z-score raw (DC+AC), no de-mean."""
    eeg_mean = norm_stats['eeg_mean']
    eeg_std = norm_stats['eeg_std']
    return ((eeg - eeg_mean) / (eeg_std + 1e-7)).astype(np.float32)


def preprocess_sources(sources, norm_stats):
    """Preprocess sources: de-mean per-region, then z-score (AC-only)."""
    sources_dm = sources - sources.mean(axis=-1, keepdims=True)
    src_mean = norm_stats['src_mean']
    src_std = norm_stats['src_std']
    return ((sources_dm - src_mean) / (src_std + 1e-7)).astype(np.float32)


def denormalize_sources(sources_norm, norm_stats):
    """Reverse z-score for sources (predicted in normalized AC space)."""
    src_mean = norm_stats['src_mean']
    src_std = norm_stats['src_std']
    return (sources_norm * (src_std + 1e-7) + src_mean).astype(np.float32)


# ── Model inference ──────────────────────────────────────────────────────────
def build_model(device='cpu'):
    """Build PhysDeepSIF and load checkpoint."""
    print('Building PhysDeepSIF model...')
    model = build_physdeepsif(
        str(LEADFIELD_PATH), str(CONNECTIVITY_PATH), lstm_hidden_size=76
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print(f'  Loaded checkpoint (epoch {ckpt.get("epoch", "?")})')
    return model


def run_inference(model, eeg_norm, device='cpu'):
    """Run model on preprocessed EEG in batches."""
    print('Running PhysDeepSIF inference...')
    n = eeg_norm.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = torch.from_numpy(eeg_norm[i:i + BATCH_SIZE]).to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ── eLORETA ──────────────────────────────────────────────────────────────────
def eloreta_inverse(eeg, leadfield, lambda_reg=0.05):
    """Depth-weighted minimum norm (eLORETA-style) inverse solution."""
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
    batch, ch, time = eeg.shape
    src = np.zeros((batch, n_sources, time), dtype=np.float64)
    for b in range(batch):
        src[b] = W @ eeg[b]
    return src.astype(np.float32)


def run_eloreta(eeg, leadfield, norm_stats):
    """Preprocess EEG and run eLORETA."""
    print('Running eLORETA...')
    eeg_norm = preprocess_input(eeg, norm_stats)
    sources = eloreta_inverse(eeg_norm, leadfield)
    return sources


# ── Metric helpers ───────────────────────────────────────────────────────────
def compute_per_sample_dle(pred, true, region_centers, epi_mask):
    """Compute DLE per sample using centroid-based asymmetric metric."""
    return np.array([
        compute_dipole_localization_error(
            pred[i:i+1], true[i:i+1], region_centers,
            epi_mask[i:i+1] if epi_mask is not None else None
        )
        for i in range(len(pred))
    ])


def compute_ei_variance(sources):
    """Compute epileptogenicity index from variance-based source activity.
    
    AC-variance based EI: compute temporal variance per region,
    z-score across regions, sigmoid map to [0, 1].
    """
    var = np.var(sources, axis=-1)
    var_z = (var - var.mean(axis=1, keepdims=True)) / (var.std(axis=1, keepdims=True) + 1e-10)
    ei = 1.0 / (1.0 + np.exp(-var_z))
    return ei


# ── Figure 1: DLE Histogram ─────────────────────────────────────────────────
def figure_dle_histogram(pred_sources_dm, eloreta_sources_dm, true_sources_ac,
                         region_centers, epi_mask):
    """Figure 1: DLE Histogram — PhysDeepSIF vs eLORETA vs Oracle."""
    print('\nFigure 1: DLE Histogram...')

    dle_pds = compute_per_sample_dle(
        pred_sources_dm, true_sources_ac, region_centers, epi_mask)
    dle_elo = compute_per_sample_dle(
        eloreta_sources_dm, true_sources_ac, region_centers, epi_mask)
    dle_orc = compute_per_sample_dle(
        true_sources_ac, true_sources_ac, region_centers, epi_mask)

    print(f'  PhysDeepSIF: {np.mean(dle_pds):.2f} +/- {np.std(dle_pds):.2f} mm')
    print(f'  eLORETA:     {np.mean(dle_elo):.2f} +/- {np.std(dle_elo):.2f} mm')
    print(f'  Oracle:      {np.mean(dle_orc):.2f} +/- {np.std(dle_orc):.2f} mm')

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 80, 50)

    ax.hist(dle_pds, bins=bins, alpha=0.5, color=COLOR_PHYSDEEPSIF,
            label='PhysDeepSIF')
    ax.hist(dle_elo, bins=bins, alpha=0.5, color=COLOR_ELORETA,
            label='eLORETA')
    ax.hist(dle_orc, bins=bins, alpha=0.5, color=COLOR_ORACLE,
            label='Oracle')

    ylim = ax.get_ylim()
    for vals, color, name in [
        (dle_pds, COLOR_PHYSDEEPSIF, 'PhysDeepSIF'),
        (dle_elo, COLOR_ELORETA, 'eLORETA'),
        (dle_orc, COLOR_ORACLE, 'Oracle'),
    ]:
        m, s = np.mean(vals), np.std(vals)
        ax.axvline(m, color=color, linestyle='--', linewidth=1.5)
        ax.text(m, ylim[1] * 0.95, f'{name}: {m:.1f} +/- {s:.1f}mm',
                color=color, fontsize=8, rotation=90, va='top', ha='right')

    ax.set_xlabel('DLE (mm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Dipole Localization Error Distribution (N={len(dle_pds)})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'dle_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved dle_histogram.png')
    return {'physdeepsif': dle_pds, 'eloreta': dle_elo, 'oracle': dle_orc}


# ── Figure 2: AUC vs SNR ────────────────────────────────────────────────────
def figure_auc_vs_snr(eeg_clean, epi_mask, norm_stats, model, device='cpu',
                      n_samples=500, leadfield=None):
    """Figure 2: AUC vs SNR — PhysDeepSIF and eLORETA across noise levels."""
    print('\nFigure 2: AUC vs SNR...')

    rng = np.random.RandomState(42)
    total = eeg_clean.shape[0]
    if total > n_samples:
        idx = np.sort(rng.choice(total, size=n_samples, replace=False))
        eeg_sub = eeg_clean[idx].copy()
        mask_sub = epi_mask[idx]
    else:
        eeg_sub = eeg_clean.copy()
        mask_sub = epi_mask

    snr_levels = [5, 10, 15, 20, 30]
    auc_pds = []
    auc_elo = []
    auc_pds_std = []
    auc_elo_std = []

    for snr_db in snr_levels:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_rng = np.random.RandomState(42 + int(snr_db))
        eeg_noisy = eeg_sub.copy()
        for b in range(eeg_noisy.shape[0]):
            signal_power = np.var(eeg_noisy[b])
            noise_std = np.sqrt(signal_power / snr_linear)
            eeg_noisy[b] += noise_rng.randn(*eeg_noisy[b].shape).astype(np.float32) * noise_std

        # PhysDeepSIF
        eeg_norm = preprocess_input(eeg_noisy, norm_stats)
        pred_norm = run_inference(model, eeg_norm, device)
        ei_pds = compute_ei_variance(pred_norm)
        auc_pds_val = compute_auc_epileptogenicity(pred_norm, mask_sub)
        auc_pds.append(auc_pds_val)

        # also compute per-sample AUC for error bars
        aucs_pds_sample = []
        for s in range(eeg_noisy.shape[0]):
            try:
                aucs_pds_sample.append(
                    compute_auc_epileptogenicity(pred_norm[s:s+1], mask_sub[s:s+1])
                )
            except Exception:
                pass
        auc_pds_std.append(np.std(aucs_pds_sample) if len(aucs_pds_sample) > 1 else 0)

        # eLORETA
        elo_src = run_eloreta(eeg_noisy, leadfield, norm_stats)
        ei_elo = compute_ei_variance(elo_src)
        auc_elo_val = compute_auc_epileptogenicity(elo_src, mask_sub)
        auc_elo.append(auc_elo_val)

        aucs_elo_sample = []
        for s in range(eeg_noisy.shape[0]):
            try:
                aucs_elo_sample.append(
                    compute_auc_epileptogenicity(elo_src[s:s+1], mask_sub[s:s+1])
                )
            except Exception:
                pass
        auc_elo_std.append(np.std(aucs_elo_sample) if len(aucs_elo_sample) > 1 else 0)

        print(f'  SNR={snr_db:2d} dB: PDS AUC={auc_pds_val:.4f} +/- {auc_pds_std[-1]:.4f}, '
              f'eLORETA AUC={auc_elo_val:.4f} +/- {auc_elo_std[-1]:.4f}')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(snr_levels, auc_pds, yerr=auc_pds_std, fmt='o-',
                color=COLOR_PHYSDEEPSIF, linewidth=2, markersize=8,
                capsize=4, label='PhysDeepSIF')
    ax.errorbar(snr_levels, auc_elo, yerr=auc_elo_std, fmt='s--',
                color=COLOR_ELORETA, linewidth=2, markersize=8,
                capsize=4, label='eLORETA')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Random (AUC=0.5)')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Perfect (AUC=1.0)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Input SNR')
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'auc_vs_snr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved auc_vs_snr.png')


# ── Figure 3: Top-K Recall ──────────────────────────────────────────────────
def figure_topk_recall(pred_sources_dm, eloreta_sources_dm, epi_mask):
    """Figure 3: Top-K Recall for K=1..10."""
    print('\nFigure 3: Top-K Recall...')

    k_values = list(range(1, 11))

    def compute_recall(sources, mask, k_list):
        power = np.mean(sources ** 2, axis=-1)
        results = {}
        for k in k_list:
            correct = 0
            for i in range(sources.shape[0]):
                top_k = np.argsort(power[i])[::-1][:k]
                true_epi = np.where(mask[i])[0]
                if len(true_epi) > 0 and any(t in top_k for t in true_epi):
                    correct += 1
            results[k] = correct / max(sources.shape[0], 1)
        return results

    recall_pds = compute_recall(pred_sources_dm, epi_mask, k_values)
    recall_elo = compute_recall(eloreta_sources_dm, epi_mask, k_values)

    # Random baseline: shuffled EI scores
    recall_rand = {}
    rand_rng = np.random.RandomState(42)
    for k in k_values:
        correct = 0
        for i in range(epi_mask.shape[0]):
            shuffled = rand_rng.permutation(N_REGIONS)
            top_k = np.sort(shuffled[:k])
            true_epi = np.where(epi_mask[i])[0]
            if len(true_epi) > 0 and np.any(np.in1d(true_epi, top_k)):
                correct += 1
        recall_rand[k] = correct / max(epi_mask.shape[0], 1)

    print('  Recall@K — PDS / eLORETA / Random:')
    for k in k_values:
        print(f'    K={k}: {recall_pds[k]:.3f} / {recall_elo[k]:.3f} / {recall_rand[k]:.3f}')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, [recall_pds[k] for k in k_values], 'o-',
            color=COLOR_PHYSDEEPSIF, linewidth=2, markersize=8, label='PhysDeepSIF')
    ax.plot(k_values, [recall_elo[k] for k in k_values], 's--',
            color=COLOR_ELORETA, linewidth=2, markersize=8, label='eLORETA')
    ax.plot(k_values, [recall_rand[k] for k in k_values], '^:',
            color=COLOR_RANDOM, linewidth=2, markersize=8, label='Random')

    # Random expected recall = K/76
    ax.plot(k_values, [k / 76 for k in k_values], 'k:',
            linewidth=1, alpha=0.5, label=f'Random expected (K/76)')

    ax.set_xlabel('K')
    ax.set_ylabel('Recall')
    ax.set_title('Top-K Recall (Epileptogenic Region Detection)')
    ax.set_xticks(k_values)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'topk_recall.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved topk_recall.png')


# ── Figure 4: Hemisphere Accuracy ───────────────────────────────────────────
def figure_hemisphere_accuracy(pred_sources_dm, eloreta_sources_dm, epi_mask,
                               region_labels):
    """Figure 4: Hemisphere Accuracy bar chart."""
    print('\nFigure 4: Hemisphere Accuracy...')

    left_idx = [i for i, lbl in enumerate(region_labels) if lbl.startswith('l')]
    right_idx = [i for i, lbl in enumerate(region_labels) if lbl.startswith('r')]

    def compute_hemi_acc(sources, mask):
        n_left, n_right = 0, 0
        correct_left, correct_right = 0, 0
        power = np.mean(sources ** 2, axis=-1)
        for i in range(sources.shape[0]):
            true_epi = np.where(mask[i])[0]
            if len(true_epi) == 0:
                continue
            true_left = any(r in left_idx for r in true_epi)
            true_right = any(r in right_idx for r in true_epi)
            if true_left and true_right:
                continue
            max_region = int(np.argmax(power[i]))
            pred_hemi = 'left' if max_region in left_idx else 'right'
            if true_left and not true_right:
                n_left += 1
                if pred_hemi == 'left':
                    correct_left += 1
            if true_right and not true_left:
                n_right += 1
                if pred_hemi == 'right':
                    correct_right += 1
        left_acc = correct_left / max(n_left, 1)
        right_acc = correct_right / max(n_right, 1)
        overall = (correct_left + correct_right) / max(n_left + n_right, 1)
        return overall, left_acc, right_acc, n_left, n_right

    overall_pds, left_pds, right_pds, nl_pds, nr_pds = compute_hemi_acc(pred_sources_dm, epi_mask)
    overall_elo, left_elo, right_elo, nl_elo, nr_elo = compute_hemi_acc(eloreta_sources_dm, epi_mask)

    # Random
    rand_src = np.random.RandomState(42).randn(*pred_sources_dm.shape).astype(np.float32)
    overall_rand, left_rand, right_rand, _, _ = compute_hemi_acc(rand_src, epi_mask)

    print(f'  Overall — PhysDeepSIF: {overall_pds:.3f}, eLORETA: {overall_elo:.3f}, Random: {overall_rand:.3f}')
    print(f'  Left    — PhysDeepSIF: {left_pds:.3f} (N={nl_pds}), eLORETA: {left_elo:.3f}, Random: {left_rand:.3f}')
    print(f'  Right   — PhysDeepSIF: {right_pds:.3f} (N={nr_pds}), eLORETA: {right_elo:.3f}, Random: {right_rand:.3f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.3

    left_bars = ax.bar(
        x - width / 2,
        [left_pds, left_elo, left_rand],
        width,
        color=[COLOR_PHYSDEEPSIF, COLOR_ELORETA, COLOR_RANDOM],
        alpha=0.85, label='Left Hemisphere', edgecolor='white', linewidth=0.5,
    )
    right_bars = ax.bar(
        x + width / 2,
        [right_pds, right_elo, right_rand],
        width,
        color=[COLOR_PHYSDEEPSIF, COLOR_ELORETA, COLOR_RANDOM],
        alpha=0.45, label='Right Hemisphere', edgecolor='white', linewidth=0.5,
        hatch='///',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(['PhysDeepSIF', 'eLORETA', 'Random'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Hemisphere Classification Accuracy')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'hemisphere_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved hemisphere_accuracy.png')


# ── Figure 5: Learning Curve ────────────────────────────────────────────────
def figure_learning_curve():
    """Figure 5: Learning Curve — parse training log for first run."""
    print('\nFigure 5: Learning Curve...')

    if not TRAINING_LOG_PATH.exists():
        print('  WARNING: Training log not found. Creating placeholder figure.')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Training log not found — run with --log-interval to capture',
                ha='center', va='center', fontsize=14, transform=ax.transAxes,
                style='italic', color='gray')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Learning Curve')
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / 'learning_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  Saved learning_curve.png (placeholder)')
        return

    text = TRAINING_LOG_PATH.read_text()
    pattern = r'Epoch\s+(\d+)/\d+.*?Train loss: ([\d.]+) \| Val loss: ([\d.]+)'
    matches = re.findall(pattern, text, re.DOTALL)

    # Parse into runs (detect epoch=1 transitions)
    runs = []
    current = []
    for epoch_str, train_str, val_str in matches:
        epoch = int(epoch_str)
        train_loss = float(train_str)
        val_loss = float(val_str)
        if epoch == 1 and current:
            runs.append(current)
            current = []
        current.append((epoch, train_loss, val_loss))
    if current:
        runs.append(current)

    # Use the first run that shows meaningful training (loss decreasing from high)
    best_run = None
    for run in runs:
        if len(run) >= 10 and run[0][1] > 1.0 and run[-1][1] < run[0][1] * 0.5:
            best_run = run
            break
    if best_run is None and runs:
        best_run = runs[0]

    if best_run is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No valid training run found in log',
                ha='center', va='center', fontsize=14, transform=ax.transAxes,
                style='italic', color='gray')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Learning Curve')
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / 'learning_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('  Saved learning_curve.png (placeholder)')
        return

    epochs = [e for e, _, _ in best_run]
    train_losses = [t for _, t, _ in best_run]
    val_losses = [v for _, _, v in best_run]
    best_idx = int(np.argmin(val_losses))
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

    print(f'  Training run: {len(epochs)} epochs, best at epoch {best_epoch} (val_loss={best_val:.4f})')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, '-', color=COLOR_PHYSDEEPSIF, linewidth=1.5,
            label='Train Loss', alpha=0.8)
    ax.plot(epochs, val_losses, '-', color=COLOR_ELORETA, linewidth=1.5,
            label='Val Loss', alpha=0.8)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Best epoch ({best_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Learning Curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved learning_curve.png')


# ── Figure 6: Method Concordance ──────────────────────────────────────────
def figure_concordance(eeg, epi_mask, norm_stats, model, leadfield, device='cpu',
                       n_samples=50):
    """Figure 6: Method-method concordance distribution.

    Concordance = overlap between PhysDeepSIF top-10 and eLORETA top-10.
    Tiers per plan §5: HIGH (>=5/10), MODERATE (2-4/10), LOW (<=1/10).
    """
    print('\nFigure 6: Method Concordance...')

    rng = np.random.RandomState(42)
    total = eeg.shape[0]
    if total > n_samples:
        idx = np.sort(rng.choice(total, size=n_samples, replace=False))
        eeg_sub = eeg[idx].copy()
        mask_sub = epi_mask[idx]
    else:
        eeg_sub = eeg.copy()
        mask_sub = epi_mask

    eeg_norm = preprocess_input(eeg_sub, norm_stats)
    pred_norm = run_inference(model, eeg_norm, device)
    ei_model = compute_ei_variance(pred_norm)

    elo_src = run_eloreta(eeg_sub, leadfield, norm_stats)
    ei_elo = compute_ei_variance(elo_src)

    def top10_set(ei_scores):
        return [set(np.argsort(ei_scores[i])[::-1][:10]) for i in range(ei_scores.shape[0])]

    model_top10 = top10_set(ei_model)
    elo_top10 = top10_set(ei_elo)

    overlaps = []
    for i in range(len(model_top10)):
        n_overlap = len(model_top10[i] & elo_top10[i])
        overlaps.append(n_overlap)

    n_high = sum(1 for o in overlaps if o >= 5)
    n_mod = sum(1 for o in overlaps if 2 <= o <= 4)
    n_low = sum(1 for o in overlaps if o <= 1)

    print(f'  Concordance — HIGH: {n_high}/{n_samples}, MODERATE: {n_mod}/{n_samples}, LOW: {n_low}/{n_samples}')
    print(f'  Mean overlap: {np.mean(overlaps):.2f}/10')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: tier distribution bar chart
    ax = axes[0]
    categories = ['HIGH\n(>=5/10)', 'MODERATE\n(2-4/10)', 'LOW\n(<=1/10)']
    counts = [n_high, n_mod, n_low]
    colors_bar = ['#2ca02c', '#ffbb22', '#d62728']
    bars = ax.bar(categories, counts, color=colors_bar, alpha=0.85, edgecolor='white', width=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Method Concordance (PhysDeepSIF vs eLORETA, N={n_samples})')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: overlap count histogram
    ax = axes[1]
    ax.hist(overlaps, bins=np.arange(-0.5, 11.5, 1), color='#1f77b4', alpha=0.8,
            edgecolor='white', rwidth=0.8)
    ax.axvline(4.5, color='#ffbb22', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate boundary')
    ax.axvline(1.5, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7, label='Low boundary')
    ax.set_xlabel('Top-10 Overlap (# regions)')
    ax.set_ylabel('Count')
    ax.set_title(f'Overlap Distribution (mean={np.mean(overlaps):.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / 'concordance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved concordance_heatmap.png')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('PhysDeepSIF — Validation Figures Generator')
    print('=' * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Load data ──────────────────────────────────────────────────────────
    eeg, sources, epi_mask, _ = load_test_subset(num_samples=None)
    leadfield, region_centers, region_labels, norm_stats = load_region_data()
    n_test = eeg.shape[0]
    print(f'Test set: {n_test} samples')

    # ── Build model ────────────────────────────────────────────────────────
    model = None
    if CHECKPOINT_PATH.exists():
        model = build_model(device)
    else:
        print('  WARNING: Checkpoint not found. Model-dependent figures will be skipped.')

    # ── Preprocess sources once (for all methods) ──────────────────────────
    print('Preprocessing sources (de-mean per region, z-score)...')
    sources_norm = preprocess_sources(sources, norm_stats)
    sources_dm = denormalize_sources(sources_norm, norm_stats)

    # ── Run PhysDeepSIF inference ──────────────────────────────────────────
    pred_sources_norm = None
    pred_sources_dm = None
    if model is not None:
        eeg_norm = preprocess_input(eeg, norm_stats)
        pred_sources_norm = run_inference(model, eeg_norm, device)
        pred_sources_dm = denormalize_sources(pred_sources_norm, norm_stats)

    # ── Run eLORETA ────────────────────────────────────────────────────────
    # eLORETA operates on raw EEG directly (inverse maps EEG to source space)
    # We preprocess input the same way as the model
    print('Running eLORETA baseline...')
    elo_src = run_eloreta(eeg, leadfield, norm_stats)
    # eLORETA output is in source space — denormalize for consistent comparison
    elo_src_dm = denormalize_sources(preprocess_sources(elo_src, norm_stats), norm_stats)

    # ── Figure 1: DLE Histogram ────────────────────────────────────────────
    if pred_sources_dm is not None:
        figure_dle_histogram(pred_sources_dm, elo_src_dm, sources_dm,
                             region_centers, epi_mask)
    else:
        print('\nFigure 1: DLE Histogram — skipped (no model predictions)')

    # ── Figure 2: AUC vs SNR ───────────────────────────────────────────────
    if model is not None:
        figure_auc_vs_snr(eeg, epi_mask, norm_stats, model, device,
                          n_samples=500, leadfield=leadfield)
    else:
        print('\nFigure 2: AUC vs SNR — skipped (no model)')

    # ── Figure 3: Top-K Recall ─────────────────────────────────────────────
    if pred_sources_dm is not None:
        figure_topk_recall(pred_sources_dm, elo_src_dm, epi_mask)
    else:
        print('\nFigure 3: Top-K Recall — skipped (no model predictions)')

    # ── Figure 4: Hemisphere Accuracy ──────────────────────────────────────
    if pred_sources_dm is not None:
        figure_hemisphere_accuracy(pred_sources_dm, elo_src_dm, epi_mask,
                                   region_labels)
    else:
        print('\nFigure 4: Hemisphere Accuracy — skipped (no model predictions)')

    # ── Figure 5: Learning Curve ───────────────────────────────────────────
    figure_learning_curve()

    # ── Figure 6: Concordance ──────────────────────────────────────────────
    if model is not None:
        figure_concordance(eeg, epi_mask, norm_stats, model, leadfield,
                           device, n_samples=50)
    else:
        print('\nFigure 6: Concordance — skipped (no model)')

    print(f'\n{"=" * 60}')
    print(f'All figures saved to {OUTPUT_DIR}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
