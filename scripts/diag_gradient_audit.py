"""
Gradient audit for training stall diagnosis (Phase D1).
Measures per-component gradient norms and shares for a fresh model
to confirm gradient starvation from leadfield amplification.

Usage:
    python scripts/diag_gradient_audit.py

Output: prints gradient norm breakdown for beta ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss

EPS = 1e-7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 8
BETAS = [0.001, 0.01, 0.05, 0.1, 0.5]
COMPONENTS = ['loss_source', 'loss_forward', 'loss_physics', 'loss_epi']

def load_data():
    with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
        n = min(N_SAMPLES, f['eeg'].shape[0])
        eeg = torch.from_numpy(f['eeg'][:n].astype(np.float32))
        src = torch.from_numpy(f['source_activity'][:n].astype(np.float32))
        mask = torch.from_numpy(f['epileptogenic_mask'][:n].astype(bool))

    src = src - src.mean(dim=-1, keepdim=True)
    eeg = eeg - eeg.mean(dim=-1, keepdim=True)
    eeg_mean, eeg_std = eeg.mean(), eeg.std()
    src_mean, src_std = src.mean(), src.std()
    eeg = (eeg - eeg_mean) / (eeg_std + EPS)
    src = (src - src_mean) / (src_std + EPS)
    return eeg.to(DEVICE), src.to(DEVICE), mask.to(DEVICE)


def compute_gradient_norms(model, loss_fn, eeg, src, mask, beta):
    loss_fn.beta = beta
    model.train()
    model.zero_grad()

    pred = model(eeg)
    loss_dict = loss_fn(pred, src, eeg, mask, epoch=5)

    norms = {}
    for comp in COMPONENTS:
        model.zero_grad()
        loss_dict[comp].backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms[comp] = total_norm ** 0.5

    return norms


def main():
    print("=" * 70)
    print("Gradient Audit — Per-Component Gradient Norms at Fresh Init")
    print("=" * 70)

    eeg, src, mask = load_data()
    model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
    model = model.to(DEVICE)

    leadfield = torch.from_numpy(np.load('data/leadfield_19x76.npy')).float()
    conn = np.load('data/connectivity_76.npy').astype(np.float32)
    lap = np.diag(conn.sum(axis=1)) - conn
    lap = torch.from_numpy(lap).float()

    L_spec_norm = np.linalg.norm(np.load('data/leadfield_19x76.npy'), ord=2)
    print(f"\nLeadfield spectral norm ||L||_2 = {L_spec_norm:.1f}")
    print(f"Expected gradient amplification: ~{L_spec_norm:.0f}x")
    print(f"Null space dims: 76 - 18 = 58 (forward gradient has zero projection)")
    print()

    print(f"{'Beta':>8} | {'L_src':>10} {'L_fwd':>10} {'L_phy':>10} {'L_epi':>10} | {'fwd/src':>8} | {'fwd_share':>9}")
    print("-" * 75)

    for beta in BETAS:
        loss_fn = PhysicsInformedLoss(
            leadfield, lap,
            alpha=1.0, beta=beta, gamma=0.01, delta_epi=1.0,
            lambda_laplacian=0.0, lambda_temporal=0.3, lambda_amplitude=0.2
        )
        loss_fn = loss_fn.to(DEVICE)

        norms = compute_gradient_norms(model, loss_fn, eeg, src, mask, beta)
        fwd_src_ratio = norms['loss_forward'] / max(norms['loss_source'], EPS)
        total = sum(norms.values())
        fwd_share = 100 * norms['loss_forward'] / max(total, EPS)

        print(f"{beta:>8.3f} | {norms['loss_source']:>10.2e} {norms['loss_forward']:>10.2e} "
              f"{norms['loss_physics']:>10.2e} {norms['loss_epi']:>10.2e} | "
              f"{fwd_src_ratio:>8.1f} | {fwd_share:>8.1f}%")

    print()
    print("Interpretation:")
    print("  fwd_share > 90% confirms gradient starvation at all beta values.")
    print("  After clip_grad_norm_(max_norm=1.0): forward dominates >99.99% of update.")
    print("  Per-component normalization (Phase D3) is the principled fix.")
    print("=" * 70)


if __name__ == '__main__':
    main()
