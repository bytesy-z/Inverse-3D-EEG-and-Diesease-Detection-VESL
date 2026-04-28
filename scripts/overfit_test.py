"""
Overfit test for PhysDeepSIF training fixes (B1-B4, corrected denominator).
Runs 100 epochs on 80/20 split of test_dataset.h5 (50 samples → 40 train / 10 val).
Success criteria: AUC > 0.6, DLE < 20mm, pred_std approaching true_std.
"""
import torch, numpy as np, h5py, sys, json
sys.path.insert(0, '.')
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss
from src.phase2_network.metrics import (
    compute_dipole_localization_error as compute_dle,
    compute_spatial_dispersion as compute_sd,
    compute_auc_epileptogenicity,
    compute_temporal_correlation,
)
import torch.optim as optim

# Load 100 samples
with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
    n = min(100, f['eeg'].shape[0])
    eeg_raw = torch.from_numpy(f['eeg'][:n].astype(np.float32))
    src_raw = torch.from_numpy(f['source_activity'][:n].astype(np.float32))
    mask = torch.from_numpy(f['epileptogenic_mask'][:n].astype(bool))

# De-mean + compute fresh z-score stats (distribution shift!)
eps = 1e-7
src_raw = src_raw - src_raw.mean(dim=-1, keepdim=True)
eeg_raw = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
eeg_mean, eeg_std = eeg_raw.mean(), eeg_raw.std()
src_mean, src_std = src_raw.mean(), src_raw.std()
eeg = (eeg_raw - eeg_mean) / (eeg_std + eps)
src = (src_raw - src_mean) / (src_std + eps)
print(f'Data: {n} samples, eeg std={eeg.std():.4f}, src std={src.std():.4f}')
print(f'  Fresh stats: eeg_std={eeg_std:.4f}, src_std={src_std:.4f}')

# 80/20 split (dataset has 50, not 100 as plan assumed)
n_train = int(n * 0.8)
eeg_train, src_train, mask_train = eeg[:n_train], src[:n_train], mask[:n_train]
eeg_val, src_val, mask_val = eeg[n_train:], src[n_train:], mask[n_train:]
print(f'Train: {len(eeg_train)}, Val: {len(eeg_val)}')
print(f'Val epi samples: {mask_val.any(dim=1).sum()}/{len(mask_val)}')

# Build model + loss
model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
leadfield = torch.from_numpy(np.load('data/leadfield_19x76.npy')).float()
conn = np.load('data/connectivity_76.npy').astype(np.float32)
lap = np.diag(conn.sum(axis=1)) - conn
loss_fn = PhysicsInformedLoss(
    leadfield, torch.from_numpy(lap).float(),
    alpha=1.0, beta=0.03, gamma=0.01, delta_epi=1.0,
    lambda_laplacian=0.0, lambda_temporal=0.3, lambda_amplitude=0.2
)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
region_centers = np.load('data/region_centers_76.npy').astype(np.float32)

print()
print('Epoch | L_src  | L_fwd  | L_epi  | DLE(mm) | SD(mm) | AUC   | Corr  | Pred_s')
print('-' * 85)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(eeg_train)
    losses = loss_fn(pred, src_train, eeg_train, mask_train, epoch=epoch)
    losses['loss_total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred_val = model(eeg_val)
            pred_np = pred_val.numpy()
            true_np = src_val.numpy()
            mask_np = mask_val.numpy()
            dle = compute_dle(pred_np, true_np, region_centers)
            sd = compute_sd(pred_np, region_centers)
            auc = compute_auc_epileptogenicity(pred_np, mask_np)
            corr = compute_temporal_correlation(pred_np, true_np)
            pred_std = pred_np.std()
        print(f'{epoch:4d}  | {losses["loss_source"].item():.4f} | '
              f'{losses["loss_forward"].item():.4f} | {losses["loss_epi"].item():.4f} | '
              f'{dle:7.2f} | {sd:7.2f} | {auc:.3f} | {corr:.3f} | {pred_std:.4f}')

print()
true_std = src_val.numpy().std()
print(f'True source std: {true_std:.4f}')
print(f'AUC must be > 0.6 (was 0.5), DLE must DECREASE across epochs, pred_std must approach {true_std:.4f}')
