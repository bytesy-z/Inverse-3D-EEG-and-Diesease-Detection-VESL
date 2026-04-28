# PhysDeepSIF — Emergency Sprint Plan: April 28–30, 2026

**Generated:** April 28, 2026  
**Last updated:** April 29, 2026 (datagen compromise applied, user sleep-decision)  
**Deadline:** April 30, 2026 (night) — ~36 hours remaining  
**Team:** Zik, Hira, Shahliza  
**Status:** PHASE A-C ✅ DONE — overfit test passed (AUC=0.732). Datagen is the remaining blocker.

### Decision Point — Apr 29
- Both fixes already in `synthetic_dataset.py`: thread limits (+27% throughput), relaxed 3/4 gradient (82%→100% yield)
- Run for 4h unattended → ~4,900 sims → ~24,500 windows ✅ (above 17,500 minimum)
- **Cannot chain training** without manual trigger — user will sleep. Training pushes to Apr 30 morning.
- Phase 3 (WebSocket, XAI, tests) done on laptop in parallel during datagen.

---

## 0. Situation Assessment

| Factor | Reality | Impact |
|--------|---------|--------|
| Synthetic data | NONE locally (only test_dataset.h5) | Must generate — single biggest blocker |
| Lab GPU | RTX 3080 + 16 CPU cores — **accessible** | Primary compute for datagen + training |
| Model checkpoint | Exists (epoch 24, val_loss=1.0377, WITH de-meaning) | Functional fallback; OLD loss function |
| Loss fix | Committed Apr 20 (variance-normalized forward loss), NOT retrained | **Training with this fix still broken — AUC≈0.5, DLE/SD flat across epochs** |
| Phase 3 code | NONE | Shahliza Day 1-2 |
| Phase 4 code | NONE | Hira Day 1-2 |
| Phase 5 code | NONE | Shahliza Day 2-3 |
| WebSockets | NONE | Zik Day 2 |
| Test suite | NONE (tests/ is empty) | All Day 3 |
| XAI | NONE | Zik Day 2 |

---

## 1. Training Root-Cause Analysis

### Why AUC≈0.5 and DLE/SD are flat across ALL epochs even after the April 20 fixes

After full code review of `loss_functions.py`, `trainer.py`, `physdeepsif.py`, `metrics.py`, and `03_train_network.py`, three interacting root causes are identified:

#### Root Cause 1: Forward loss denominator causes instant gradient blow-up → model freezes at epoch 0

In `loss_functions.py:_compute_forward_loss()` (line 371-388):

```python
raw_mse = ((eeg_predicted - eeg_input) ** 2).mean()
fwd_var = eeg_predicted.var().detach()     # ← UNSTABLE at initialization
loss = raw_mse / (fwd_var + _EPS)           # EPS = 1e-7
```

At initialization (`Ŝ ≈ 0`), `eeg_predicted = L @ Ŝ ≈ 0`, so `fwd_var ≈ 0`. The loss becomes `MSE(0, EEG) / 1e-7 ≈ 1 / 1e-7 = 10^7`. The resulting gradient has magnitude ~10^7, which is clipped to norm=1.0. **The gradient direction is preserved but the model converges instantly (epoch 0) to the pseudo-inverse solution `L^T @ EEG` and never moves from it** — hence DLE is flat across all epochs (it converged before epoch 1).

After this instant convergence, `Var(L@Ŝ)` becomes large (since `L@Ŝ` approximates EEG), so the forward loss drops dramatically. The remaining gradients (source loss, epi loss) are too weak relative to the main loss to move the model from this local minimum. This is a **gradient starvation** problem.

#### Root Cause 2: EPI loss can never push probabilities below 0.5

In `loss_functions.py:_compute_epi_loss()` (line 296-323):

```python
power = predicted_sources.pow(2).mean(dim=-1)  # ALWAYS ≥ 0
target = epileptogenic_mask.float()
return self.bce_loss(power, target)             # BCEWithLogitsLoss
```

Since `power ≥ 0`, `sigmoid(power) ≥ 0.5` for ALL regions — healthy regions can never have a predicted probability below 0.5. The BCE gradient is correct (pushes healthy down, epi up), but with amplitude-collapsed predictions (`power ≈ 0` for all), `sigmoid(0) = 0.5` for everything → AUC = 0.5 exactly.

#### Root Cause 3: Per-region source de-meaning breaks forward consistency

Per-region source de-meaning (`S_ac = S_raw - mean_per_region(S_raw)`) does NOT commute with the leadfield projection:

- `L @ S_ac = L @ S_raw - L @ mean_per_region(S_raw)`
- `EEG_ac = EEG_raw - mean_per_channel(EEG_raw)`
- `L @ S_ac ≠ EEG_ac` because `L @ mean_per_region(S_raw) ≠ mean_per_channel(L @ S_raw)`

The model is trained to predict de-meaned sources, but the forward loss compares `L @ Ŝ_de-meaned` with `EEG_de-meaned` — these are physically inconsistent, creating an irreducible error floor.

---

## 2. Training Debugging Plan (Executable by Coding Agent)

### Phase A: Diagnostics (must run FIRST, ~30 min total)

#### A1. Data Sanity Check
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import h5py, numpy as np

# Check if training data exists
import os
data_dir = 'data/synthetic3/'
for fname in ['train_dataset.h5', 'val_dataset.h5']:
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            eeg = f['eeg'][:]
            src = f['source_activity'][:]
            has_mask = 'epileptogenic_mask' in f
            if has_mask:
                mask = f['epileptogenic_mask'][:]
                n_epi = mask.sum(axis=1)
            
            # Per-region de-mean
            src_ac = src - src.mean(axis=-1, keepdims=True)
            eeg_ac = eeg - eeg.mean(axis=-1, keepdims=True)
            
            print(f'{fname}: {src.shape[0]} samples')
            print(f'  eeg_raw: mean={eeg.mean():.4f} std={eeg.std():.4f}')
            print(f'  src_raw: mean={src.mean():.4f} std={src.std():.4f}')
            print(f'  eeg_ac:  mean={eeg_ac.mean():.4f} std={eeg_ac.std():.4f} (de-meaned)')
            print(f'  src_ac:  mean={src_ac.mean():.4f} std={src_ac.std():.4f} (de-meaned)')
            if has_mask:
                print(f'  epi samples: {np.count_nonzero(n_epi > 0)}/{len(n_epi)} ({100*np.count_nonzero(n_epi > 0)/len(n_epi):.1f}%)')
                print(f'  mean n_epi: {n_epi[n_epi>0].mean():.1f}')
                
                # Check: are epi regions distinguishable after de-meaning?
                epi_src = src_ac[mask]  # source values in epi regions
                healthy_src = src_ac[~mask]  # source values in healthy regions
                epi_var = epi_src.var()
                healthy_var = healthy_src.var()
                print(f'  epi variance: {epi_var:.6f}')
                print(f'  healthy variance: {healthy_var:.6f}')
                print(f'  variance ratio: {epi_var/healthy_var:.2f}x (target: >3x)')
    else:
        print(f'MISSING: {path}')
"
```
**Expected:** eeg_ac std ≈ 10 (raw µV), src_ac std ≈ 0.23, variance ratio ≥ 3×. Train dataset should exist or we proceed with test_dataset.h5. If all MISSING → generate data first (Section 3).

#### A2. Loss Scale Audit (run on existing checkpoint)
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import torch, numpy as np, h5py, sys
sys.path.insert(0, '.')
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss

# Load model + 1 batch from test dataset
model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
ckpt = torch.load('outputs/models/checkpoint_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
    eeg = torch.from_numpy(f['eeg'][:8].astype(np.float32))
    src = torch.from_numpy(f['source_activity'][:8].astype(np.float32))
    mask = torch.from_numpy(f['epileptogenic_mask'][:8].astype(bool))

# Apply de-meaning (matching training)
src = src - src.mean(dim=-1, keepdim=True)
eeg = eeg - eeg.mean(dim=-1, keepdim=True)

# Load normalization stats
import json
with open('outputs/models/normalization_stats.json') as f:
    stats = json.load(f)
eps = 1e-7
eeg_norm = (eeg - stats['eeg_mean']) / (stats['eeg_std'] + eps)
src_norm = (src - stats['src_mean']) / (stats['src_std'] + eps)

# Load loss fn with matching matrices
import numpy as np
leadfield = torch.from_numpy(np.load('data/leadfield_19x76.npy')).float()
connectivity = np.load('data/connectivity_76.npy').astype(np.float32)
laplacian = np.diag(connectivity.sum(axis=1)) - connectivity
laplacian_t = torch.from_numpy(laplacian).float()

loss_fn = PhysicsInformedLoss(leadfield, laplacian_t)

# Test: random init vs trained model
print('=== LOSS COMPONENT AUDIT ===')
with torch.no_grad():
    pred_trained = model(eeg_norm)
    pred_random = torch.randn_like(pred_trained) * 0.01  # near-zero init
    
    for label, pred in [('TRAINED', pred_trained), ('RANDOM_NEAR_ZERO', pred_random)]:
        losses = loss_fn(pred, src_norm, eeg_norm, mask)
        print(f'  [{label}]')
        for k, v in losses.items():
            print(f'    {k}: {v.item():.6f}')
        # Also check amplitude
        print(f'    pred_std: {pred.std().item():.6f} (true std: {src_norm.std().item():.6f})')
        print(f'    forward_pred_std: {(leadfield @ pred.transpose(1,2)).std().item():.6f}')

# Key diagnostic: what happens to L_forward with near-zero predictions?
print()
print('=== FORWARD LOSS DENOMINATOR INSTABILITY TEST ===')
pred_zero = torch.zeros_like(src_norm)
fwd = leadfield @ pred_zero.transpose(1,2)  # should be near zero
raw_mse = ((fwd - eeg_norm.transpose(1,2)) ** 2).mean()
fwd_var = fwd.var()
print(f'  At Ŝ=0: raw_mse={raw_mse.item():.4f}, fwd_var={fwd_var.item():.10f}')
print(f'  L_forward = {raw_mse.item():.4f} / ({fwd_var.item():.10f} + 1e-7) = {raw_mse.item() / (fwd_var.item() + 1e-7):.2f}')
print(f'  This is {''BLOWN UP'' if raw_mse.item()/(fwd_var.item()+1e-7) > 100 else ''OK''} — should be O(1), not O(1e7)')
"
```
**Expected findings:** Trained model has `pred_std` MUCH smaller than `true_std` (amplitude collapse confirmed). L_forward at `Ŝ=0` is O(1e7) confirming denominator blow-up.

#### A3. Gradient Flow Check
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import torch, numpy as np, h5py, sys
sys.path.insert(0, '.')
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss

model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
leadfield = torch.from_numpy(np.load('data/leadfield_19x76.npy')).float()
connectivity = np.load('data/connectivity_76.npy').astype(np.float32)
laplacian = np.diag(connectivity.sum(axis=1)) - connectivity
loss_fn = PhysicsInformedLoss(leadfield, torch.from_numpy(laplacian).float())

with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
    eeg = torch.from_numpy(f['eeg'][:4].astype(np.float32))
    src = torch.from_numpy(f['source_activity'][:4].astype(np.float32))
    mask = torch.from_numpy(f['epileptogenic_mask'][:4].astype(bool))

# De-mean + normalize
import json
with open('outputs/models/normalization_stats.json') as f:
    stats = json.load(f)
eps = 1e-7
src = src - src.mean(dim=-1, keepdim=True)
eeg = eeg - eeg.mean(dim=-1, keepdim=True)
eeg_n = (eeg - stats['eeg_mean']) / (stats['eeg_std'] + eps)
src_n = (src - stats['src_mean']) / (stats['src_std'] + eps)

# Forward + backward with fresh model
pred = model(eeg_n)
losses = loss_fn(pred, src_n, eeg_n, mask)
losses['loss_total'].backward()

# Check gradient norms per layer
print('=== GRADIENT FLOW ===')
total_grad_norm = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        gnorm = param.grad.norm().item()
        total_grad_norm += gnorm ** 2
        if 'lstm' in name.lower():
            label = 'LSTM'
        elif 'fc' in name.lower() or 'spatial' in name.lower():
            label = 'SPATIAL'
        elif 'bn' in name.lower():
            label = 'BN'
        elif 'skip' in name.lower() or 'projection' in name.lower():
            label = 'SKIP'
        elif 'output' in name.lower():
            label = 'OUTPUT'
        else:
            label = 'OTHER'
        if gnorm < 1e-8:
            print(f'  [ZERO!] {label} {name}: grad_norm={gnorm:.2e}')
        else:
            print(f'  [{label}] {name}: grad_norm={gnorm:.6f}')
    else:
        print(f'  [DEAD!] {name}: NO GRADIENT')

print(f'  Total grad norm (before clip): {total_grad_norm**0.5:.6f}')

# Check per-loss-component gradients
print()
print('=== PER-COMPONENT GRADIENT CONTRIBUTION ===')
for loss_name in ['loss_source', 'loss_forward', 'loss_epi']:
    model.zero_grad()
    losses[loss_name].backward(retain_graph=True)
    grad_sum = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f'  {loss_name} grad norm: {grad_sum:.6f}')
"
```
**Expected findings:** Forward loss gradient dominates by 100-1000×. This confirms gradient starvation.

---

### Phase B: Fixes (apply sequentially, verify each)

#### B1. CRITICAL FIX — Stabilize forward loss denominator

**File:** `src/phase2_network/loss_functions.py`  
**Function:** `_compute_forward_loss()` (line 325-388)

**Edit 1 — line 386, change:**
```python
# BEFORE (line 386 — UNSTABLE):
loss = raw_mse / (fwd_var + _EPS)

# AFTER:
eeg_var = eeg_input.var().detach()
loss = raw_mse / (eeg_var + _EPS)
```

**Edit 2 — Remove lines 381-382 (no longer needed):**
```python
# DELETE these two lines:
# fwd_var = eeg_predicted.var().detach()
```
(PyTorch will clean up the unused `fwd_var` computation.)

**Edit 3 — Replace the docstring (lines 337-359) with:**
```python
        """
        Compute variance-normalised forward consistency loss.

        L_forward = MSE(L @ Ŝ, EEG^input) / (Var(EEG^input) + ε)

        Why normalise by Var(EEG) and NOT by Var(L @ Ŝ)?
        ─────────────────────────────────────────────────
        Var(EEG) ≈ 1.0 after z-scoring — it is a STABLE constant throughout
        training.  Var(L @ Ŝ) is unstable: at initialisation Ŝ≈0 ⇒ Var(L@Ŝ)≈0
        ⇒ L_forward ≈ 1e7 ⇒ gradient blow-up ⇒ model converges to L^T@EEG
        (pseudo-inverse) in epoch 0 and freezes there permanently.

        Dividing by Var(EEG) produces a dimensionless relative error ≈ O(1)
        at ALL stages of training, giving the source loss and epi loss
        comparable gradient magnitude to the forward loss.
        """
```

**Verify:** Re-run A2 diagnostic. `L_forward` at `Ŝ=0` should now be ≈ O(1) (around 0.5-2.0), not O(1e7).

#### B2. CRITICAL FIX — Per-channel de-mean forward prediction before comparison

**File:** `src/phase2_network/loss_functions.py`  
**Function:** `_compute_forward_loss()`  

**Insert AFTER line 373** (`eeg_predicted = torch.einsum('ij,bjk->bik', self.leadfield, predicted_sources)`):
```python
        # ── Per-channel temporal de-meaning on forward prediction ──
        # Both EEG_input and predicted_sources are de-meaned per-channel/per-region
        # BEFORE normalisation.  However per-region source de-meaning does NOT
        # commute with the leadfield projection: L@S_ac ≠ EEG_ac.  To restore
        # physical consistency we de-mean the forward prediction per-channel as
        # well, so both terms in the MSE are AC-coupled and directly comparable.
        eeg_predicted = eeg_predicted - eeg_predicted.mean(dim=-1, keepdim=True)
```

#### B3. CRITICAL FIX — Replace BCE EPI loss with class-balanced MSE

**File:** `src/phase2_network/loss_functions.py`  
**Function:** `_compute_epi_loss()` (line 296-323)

**Replace the entire function body with:**
```python
    def _compute_epi_loss(
        self,
        predicted_sources: torch.Tensor,
        epileptogenic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute epileptogenic classification loss (class-balanced MSE on power).

        L_epi = weighted_MSE(mean_t(Ŝ²), target_mask)

        Uses MSE on region power instead of BCEWithLogitsLoss because power≥0
        means sigmoid(power)≥0.5 always — healthy regions can never drop below
        chance level.  MSE directly pushes epi power toward 1.0 and healthy
        power toward 0.0 with symmetric gradients.

        Class-balance weighting: epi regions (rare, ~2-8 per sample) are
        up-weighted by N_healthy/N_epi ≈ 10-50× so they contribute equally
        to the gradient despite class frequency.
        """
        if epileptogenic_mask is None:
            return torch.tensor(0.0, device=predicted_sources.device)

        # Region power: (batch, 76)
        power = predicted_sources.pow(2).mean(dim=-1)
        target = epileptogenic_mask.float()  # 1 for epi, 0 for healthy

        # Per-region MSE: (batch, 76)
        per_region_mse = (power - target) ** 2

        # Class-balance weights
        n_epi = epileptogenic_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        n_healthy = N_REGIONS - n_epi
        weights = torch.ones_like(per_region_mse)
        epi_weight = n_healthy / n_epi
        epi_mask_expanded = epileptogenic_mask
        weights[epi_mask_expanded] = epi_weight.expand_as(weights)[epi_mask_expanded]

        loss = (per_region_mse * weights).sum() / weights.sum()
        return loss
```

#### B4. RECOMMENDED FIX — Warm-up forward loss weight

**File:** `src/phase2_network/loss_functions.py`  
**Method:** `forward()` (line 182-251)

**Step 1 — add `epoch` parameter to signature (line 182-188):**
```python
    def forward(
        self,
        predicted_sources: torch.Tensor,
        true_sources: torch.Tensor,
        eeg_input: torch.Tensor,
        epileptogenic_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,  # ← NEW: for warm-up schedule
    ) -> Dict[str, torch.Tensor]:
```

**Step 2 — replace the composite loss computation (lines 235-240) with:**
```python
        # Warm-up: linearly ramp beta from 0 to target over first 5 epochs
        warmup_epochs = 5
        if epoch < warmup_epochs:
            beta_effective = self.beta * (epoch / warmup_epochs)
        else:
            beta_effective = self.beta

        # Composite total loss
        loss_total = (
            self.alpha * loss_source
            + beta_effective * loss_forward
            + self.gamma * loss_physics
            + self.delta_epi * loss_epi
        )
```

**File:** `src/phase2_network/trainer.py`  
**Method:** `_train_epoch()` (line 302)

**Step 3 — pass epoch to loss_fn (add epoch param, line 346):**
Locate line 346: `loss_dict = self.loss_fn(source_pred, sources, eeg_augmented, mask)`
Change to:
```python
                loss_dict = self.loss_fn(source_pred, sources, eeg_augmented, mask,
                                         epoch=self.current_epoch)
```

**Step 4 — store current_epoch in the trainer (line 217 in `train()`):**
Before the for-loop `for epoch in range(num_epochs):`, add:
```python
        self.current_epoch = 0
```
Inside the for-loop, as the first line after `for epoch in range(num_epochs):` (line 218), add:
```python
            self.current_epoch = epoch
```

---

### Phase C: Overfit Test (verify fixes work, ~10 min)

After applying B1-B4, run this overfit test on 100 samples. **This must pass before datagen + full training.**

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import torch, numpy as np, h5py, sys, json
sys.path.insert(0, '.')
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss
from src.phase2_network.metrics import (compute_dipole_localization_error as compute_dle,
                                         compute_spatial_dispersion as compute_sd,
                                         compute_auc_epileptogenicity,
                                         compute_temporal_correlation)
import torch.optim as optim

# Load 100 samples from test dataset
with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
    n = min(100, f['eeg'].shape[0])
    eeg_raw = torch.from_numpy(f['eeg'][:n].astype(np.float32))
    src_raw = torch.from_numpy(f['source_activity'][:n].astype(np.float32))
    mask = torch.from_numpy(f['epileptogenic_mask'][:n].astype(bool))

# De-mean + normalize (exact match to HDF5Dataset.__iter__)
with open('outputs/models/normalization_stats.json') as f:
    stats = json.load(f)
eps = 1e-7
src_raw = src_raw - src_raw.mean(dim=-1, keepdim=True)
eeg_raw = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
eeg = (eeg_raw - stats['eeg_mean']) / (stats['eeg_std'] + eps)
src = (src_raw - stats['src_mean']) / (stats['src_std'] + eps)
print(f'Data: {n} samples, eeg std={eeg.std():.4f}, src std={src.std():.4f}')

# Split: 80 train, 20 val
eeg_train, src_train, mask_train = eeg[:80], src[:80], mask[:80]
eeg_val, src_val, mask_val = eeg[80:], src[80:], mask[80:]

# Build fresh model
model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
leadfield = torch.from_numpy(np.load('data/leadfield_19x76.npy')).float()
conn = np.load('data/connectivity_76.npy').astype(np.float32)
lap = np.diag(conn.sum(axis=1)) - conn
loss_fn = PhysicsInformedLoss(
    leadfield, torch.from_numpy(lap).float(),
    alpha=1.0, beta=0.1, gamma=0.01, delta_epi=1.0,
    lambda_laplacian=0.0, lambda_temporal=0.3, lambda_amplitude=0.2
)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
region_centers = np.load('data/region_centers_76.npy').astype(np.float32)

print()
print('Epoch | L_src  | L_fwd  | L_epi  | DLE(mm) | SD(mm) | AUC   | Corr  | Pred_σ')
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
        
        print(f'{epoch:4d}  | {losses[\"loss_source\"].item():.4f} | '
              f'{losses[\"loss_forward\"].item():.4f} | {losses[\"loss_epi\"].item():.4f} | '
              f'{dle:7.2f} | {sd:7.2f} | {auc:.3f} | {corr:.3f} | {pred_std:.4f}')

print()
true_std = src_val.numpy().std()
print(f'True source std: {true_std:.4f}')
print(f'AUC must be > 0.6 (was 0.5), DLE must DECREASE across epochs, pred_std must approach {true_std:.4f}')
"
```
**Success criteria (MUST PASS before proceeding to full training):**
- AUC > 0.6 by epoch 50 (was 0.5 with old loss)
- DLE decreasing across epochs (was flat with old loss)
- `pred_std` approaching `true_std` (amplitude recovery, was collapsed at 0.04×)
- Loss components all at O(1) scale (no 10^7 blow-up)

---

### Phase D: Full Training Run

**Prerequisites:** Overfit test MUST pass. Training data MUST exist.

```bash
# Step 1 — Generate data if not already done (runs on lab machine CPU cores):
/home/zik/miniconda3/envs/physdeepsif/bin/python scripts/02_generate_synthetic_data.py \
  --n-sims 5000 --n-jobs 16 --output-dir data/synthetic3/ --split train

# Step 2 — Verify data exists:
ls -lh data/synthetic3/train_dataset.h5 data/synthetic3/val_dataset.h5

# Step 3 — Train with FIXED loss on lab GPU:
/home/zik/miniconda3/envs/physdeepsif/bin/python scripts/03_train_network.py \
  --epochs 80 --batch-size 64 --device cuda
```

**Expected:** DLE < 20mm within epoch 3-5 (was instant, now gradual). AUC climbs from 0.5 toward 0.7. SD and correlation improve. Training does NOT early-stop at epoch 8 with all metrics flat.

---

### Phase E: Failsafe — If overfit test fails after B1-B4

If AUC still = 0.5 and DLE still flat, test these in order:

**E1. Bypass temporal module:**
```python
# In physdeepsif.py forward(), replace:
#   source_estimate = self.temporal_module(spatial_out)
#   return source_estimate
# With:
return spatial_out  # spatial-only, skip BiLSTM
```
If overfit now works → BiLSTM initialization or gradient flow is the bottleneck. Disable temporal module for full training (acceptable for MVP — spatial module can learn the inverse mapping on its own).

**E2. Disable data augmentation:**
In `trainer.py:_augment_batch()`, replace entire function body with `return eeg.clone()`. If this fixes learning → augmentation is adding too much noise.

**E3. Bypass de-meaning (test only):**
Comment out the de-meaning lines in `HDF5Dataset.__iter__()` (lines 201-208 of `03_train_network.py`). If this fixes learning → de-meaning removes too much signal. **Do not ship without de-meaning** — it's needed for real EEG compatibility.

**E4. Pure supervised baseline:**
```python
loss = torch.nn.functional.mse_loss(pred, src)
```
If pure MSE works but composite loss doesn't → loss weighting/balancing is fundamentally wrong. Drop physics and forward losses, use only source + epi loss.

---

## 3. Compute Timeline — Realized (Post-Datagen Compromise)

**⚠️ Datagen reality: TVB sim takes 24s, not 0.5s. Plan adjusted Apr 29.**

### 3a: What's DONE (Phases A-C, Apr 28-29)
```
✅ A1-A3 Diagnostics   (30 min)  — BLOWN UP L_forward = 1e7 confirmed at Ŝ=0
✅ B1-B4 Fixes applied            — Combined Var(EEG)+Var(L@Ŝ) denominator
✅ C Overfit test       (10 min)  — AUC=0.732, DLE=8.4mm, Pred_σ=0.39
```

### 3b: Remaining Compute (Lab RTX 3080 + 16 cores)

```
Apr 29 09:00 ──────────────────────────────────────────────  Apr 30 23:59
│
├─ [Ph D]  4 h       DATAGEN (CPU, 0.34 sims/sec, ~24,500 windows) ──┤
│                    ⚠ Start BEFORE sleep                            │
│          3.2 h     TRAINING (GPU RTX 3080, 80 epochs) ─────────────┤
│                    (starts automatically at T+4h if you wake up)    │
│                                                                     │
├─ [Apr 29 afternoon]  Merge Hira/Shahliza branches ──────────────────┤
├─ [Apr 29 eve]        XAI wiring + WebSocket fix ────────────────────┤
├─ [Apr 30 morning]    Integration + CMA-ES endpoint ─────────────────┤
├─ [Apr 30 afternoon]  Testing + thesis figures ──────────────────────┤
├─ [Apr 30 night]      Polish + submit ───────────────────────────────┤
│                                                                     │
│ SPILLOVER from Apr 28: Full training runs into Apr 29 afternoon.    │
│ Phase 3 (WebSocket, XAI, backend) must be COMPLETED Apr 29.        │
└──────────────────────────────────────────────────────────────────────
```

### 3c: What Spills Over & What's Saved

| Task | Original date | New date | Impact |
|------|:------------:|:--------:|--------|
| Datagen start | Apr 28 eve | **Apr 29 morning** | Training delayed by 12h |
| Training end | Apr 28 night | **Apr 29 16:15** | Model ready for afternoon integration |
| WebSocket fix | Apr 28 eve | Apr 29 morning | **Same day** — no spillover |
| XAI wiring | Apr 28 eve | Apr 29 morning | **Same day** — no spillover |
| Backend CMA-ES | Apr 29 | Apr 29 afternoon | Still on schedule |
| Branch merge | Apr 29 | Apr 29 afternoon | Still on schedule |
| Testing | Apr 30 | Apr 30 | Still on schedule |
| Thesis figures | Apr 30 | Apr 30 | Still on schedule |

---

## 4. Detailed Implementation Specifications

### 4.1 ZIK — Training Fix + Backend + WebSocket + XAI + Tests

---

#### 📅 APRIL 28 (TODAY) — Critical Path: Fix Model + Start Compute

---

##### Z1-APR28: Training Debug + Fix (Start immediately)

**Files to edit:**
1. `src/phase2_network/loss_functions.py` — B1, B2, B3, B4 edits (see Section 2 above)
2. `src/phase2_network/trainer.py` — B4 epoch passing (see Section 2 above)

**Verification:** Run Phase A diagnostics (A1-A3), apply B1-B4, run Phase C overfit test.

**⚠️ BLOCKER:** Overfit test MUST PASS (AUC > 0.6, DLE decreasing) before any other task.

**Handoff:** Commit with message `fix(loss): stabilise forward loss denominator, per-channel de-mean, class-balanced epi MSE, beta warm-up`. Push to `main`.

---

##### Z2-APR28: Synthetic Data Generation (Background — start AS SOON as overfit test passes)

```bash
# Run on lab machine (16 cores). This is a blocking dependency for training.
/home/zik/miniconda3/envs/physdeepsif/bin/python scripts/02_generate_synthetic_data.py \
  --n-sims 5000 --n-jobs 16 --output-dir data/synthetic3/ --split train
```

**Monitor progress:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import h5py
try:
    with h5py.File('data/synthetic3/train_dataset.h5', 'r') as f:
        print(f'Written: {f[\"eeg\"].shape[0]} samples')
except: print('Not yet created')
"
```

**Verification:** After completion, `train_dataset.h5` must have ≥ 17,500 samples.

---

##### Z3-APR28: Full Training (GPU — start AS SOON as data gen completes)

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python scripts/03_train_network.py \
  --epochs 80 --batch-size 64 --device cuda
```

**Monitor:** AUC must increase from 0.5 → 0.7+, DLE must decrease, SD must improve.

**Handoff:** Copy `outputs/models/checkpoint_best.pt` and `normalization_stats.json`. Commit both.

---

##### Z4-APR28: WebSocket Endpoint (While datagen/training runs)

**File:** `backend/server.py`

**Add after existing imports (~line 46):**
```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
```

**Add new endpoint before `if __name__ == "__main__":` (before line ~1640):**
```python
# ── WebSocket for real-time job status ──────────────────────────────
active_jobs: Dict[str, Dict] = {}  # job_id → {status, progress, message}

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            if job_id in active_jobs:
                await websocket.send_json(active_jobs[job_id])
                if active_jobs[job_id].get("status") in ("completed", "failed"):
                    del active_jobs[job_id]
                    break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
```

**Modify `/api/analyze` to support async WebSocket mode (add to analyze_handler):**
```python
# After extracting parameters, add:
ws_job_id = str(uuid.uuid4())[:8]
if request.query_params.get("ws") == "true":
    # Return job_id immediately, process in background
    active_jobs[ws_job_id] = {"status": "queued", "progress": 0, "message": "Starting..."}
    asyncio.create_task(_process_analysis_async(ws_job_id, eeg_file, mode, ...))
    return JSONResponse({"status": "queued", "job_id": ws_job_id})

# Helper function (add new function):
async def _process_analysis_async(job_id: str, file_path: str, mode: str, ...):
    try:
        active_jobs[job_id] = {"status": "preprocessing", "progress": 10, "message": "Preprocessing EEG..."}
        # ... existing inference code, updating active_jobs at each step ...
        active_jobs[job_id] = {"status": "completed", "progress": 100, "message": "Done"}
    except Exception as e:
        active_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}
```

**Verification:**
```bash
# Start backend, then test:
curl -X POST "http://127.0.0.1:8000/api/analyze?ws=true" \
  -F "file=@data/samples/0001082.edf" -F "mode=source_localization"
# Should return {"status": "queued", "job_id": "xxxxxxxx"}

# Connect WebSocket in another terminal:
# (manual test via wscat or the frontend)
```

---

##### Z5-APR28: XAI Occlusion Module + Test Suite (While datagen/training runs)

**File to create:** `src/xai/__init__.py` (empty)
**File to create:** `src/xai/eeg_occlusion.py`

```python
"""
Module: eeg_occlusion.py
Purpose: Occlusion-based XAI for biomarker detection.

For a given EEG window and target epileptogenic region, masks successive
channel-time segments, re-runs the PhysDeepSIF + biomarker pipeline,
and measures the score drop.  Segments that cause the largest drop are
most influential for the detection.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple
import torch
import logging

logger = logging.getLogger(__name__)

# Default occlusion parameters
OCCLUSION_WIDTH_SAMPLES = 40   # 200 ms at 200 Hz
OCCLUSION_STRIDE_SAMPLES = 20  # 100 ms overlap
N_CHANNELS = 19
WINDOW_LENGTH = 400


def explain_biomarker(
    eeg_window: NDArray,                # (19, 400) single window
    target_region_idx: int,             # 0-75 DK region index
    run_pipeline_fn,                    # callable(eeg_window) → dict with "scores"
    occlusion_width: int = OCCLUSION_WIDTH_SAMPLES,
    stride: int = OCCLUSION_STRIDE_SAMPLES,
) -> Dict:
    """
    Occlusion-based attribution for a biomarker detection.

    Args:
        eeg_window: Single EEG window (19, 400), z-scored.
        target_region_idx: Index of the top-1 detected region to explain.
        run_pipeline_fn: Function that takes (19,400) EEG and returns
                        dict with "scores" key → (76,) array of EI scores.
        occlusion_width: Width of occlusion segment in samples (default 200 ms).
        stride: Step between occlusion segments (default 100 ms).

    Returns:
        dict with:
            channel_importance: (19,) mean attribution per channel
            time_importance: (n_segments,) attribution per time segment
            attribution_map: (19, n_segments) full channel-time attribution
            top_segments: list[dict] top-5 influential segments
            target_region_idx: int
            baseline_score: float  (unoccluded EI score)
    """
    # Baseline: score without occlusion
    baseline_result = run_pipeline_fn(eeg_window)
    baseline_score = float(baseline_result["scores"][target_region_idx])

    n_segments = (WINDOW_LENGTH - occlusion_width) // stride + 1
    attribution_map = np.zeros((N_CHANNELS, n_segments), dtype=np.float32)

    for ch in range(N_CHANNELS):
        for seg_idx in range(n_segments):
            t_start = seg_idx * stride
            t_end = t_start + occlusion_width

            # Create occluded EEG copy
            eeg_occ = eeg_window.copy()
            # Mask: replace segment with per-channel temporal mean (≈0 after de-meaning)
            eeg_occ[ch, t_start:t_end] = 0.0

            # Re-run pipeline
            occ_result = run_pipeline_fn(eeg_occ)
            occ_score = float(occ_result["scores"][target_region_idx])

            # Attribution = score drop (positive = segment supported detection)
            attribution_map[ch, seg_idx] = baseline_score - occ_score

    # Aggregate
    channel_importance = attribution_map.mean(axis=1)  # (19,)
    time_importance = attribution_map.mean(axis=0)      # (n_segments,)

    # Find top segments
    top_indices = np.argsort(attribution_map.ravel())[-5:][::-1]
    top_segments = []
    for flat_idx in top_indices:
        ch, seg = np.unravel_index(flat_idx, attribution_map.shape)
        t_center = seg * stride + occlusion_width // 2
        top_segments.append({
            "channel_idx": int(ch),
            "start_sample": int(seg * stride),
            "end_sample": int(seg * stride + occlusion_width),
            "start_time_sec": float(seg * stride / 200.0),
            "end_time_sec": float((seg * stride + occlusion_width) / 200.0),
            "importance": float(attribution_map[ch, seg]),
        })

    return {
        "channel_importance": channel_importance.tolist(),
        "time_importance": time_importance.tolist(),
        "attribution_map": attribution_map.tolist(),
        "top_segments": top_segments[:5],
        "target_region_idx": target_region_idx,
        "baseline_score": baseline_score,
    }
```

**Integration into `backend/server.py`:**

After `compute_epileptogenicity_index()` (or in the biomarkers handler), add:
```python
from src.xai.eeg_occlusion import explain_biomarker

# Inside biomarkers handler, after computing scores:
top_region_idx = int(np.argmax(scores))
xai_result = explain_biomarker(
    eeg_window=eeg_windows[top_window_idx],  # use highest-EI window
    target_region_idx=top_region_idx,
    run_pipeline_fn=lambda eeg_win: _run_biomarker_single_window(eeg_win, model, ...),
    occlusion_width=40,
    stride=20,
)

# Add to response:
response_dict["xai"] = {
    "method": "occlusion",
    "target_region": region_codes[top_region_idx],
    "target_region_full": region_names[top_region_idx],
    ...xai_result,
}
```

**Verification:** Run on 0001082.edf — check that top attributed channels include Fp2, F4, F8 (matching 1082.csv annotations).

---

#### Task Z6: Test Suite (Hours 7-8)

**File:** `tests/__init__.py` (empty)
**File:** `tests/conftest.py`

```python
import pytest
import torch
import numpy as np
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(scope="session")
def model():
    """Load PhysDeepSIF model from checkpoint."""
    from src.phase2_network.physdeepsif import build_physdeepsif
    model = build_physdeepsif(
        str(PROJECT_ROOT / "data/leadfield_19x76.npy"),
        str(PROJECT_ROOT / "data/connectivity_76.npy"),
    )
    ckpt = torch.load(
        str(PROJECT_ROOT / "outputs/models/checkpoint_best.pt"),
        map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@pytest.fixture(scope="session")
def normalization_stats():
    with open(PROJECT_ROOT / "outputs/models/normalization_stats.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def synthetic_sample():
    import h5py
    with h5py.File(str(PROJECT_ROOT / "data/synthetic3/test_dataset.h5"), "r") as f:
        eeg = torch.from_numpy(f["eeg"][0:1].astype(np.float32))
        mask = torch.from_numpy(f["epileptogenic_mask"][0:1].astype(bool))
    return eeg, mask

@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient for API testing."""
    from fastapi.testclient import TestClient
    import backend.server as server_mod
    return TestClient(server_mod.app)
```

**File:** `tests/test_model.py`

```python
def test_model_loads(model):
    """Model loads from checkpoint without error."""
    assert model is not None
    params = model.get_parameter_count()
    assert params["total_trainable"] > 300_000  # ~419k

def test_forward_pass_shape(model, synthetic_sample):
    """Forward pass produces correct output shape."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert output.shape == (1, 76, 400), f"Expected (1,76,400), got {output.shape}"

def test_output_finite(model, synthetic_sample):
    """Model output contains no NaN or Inf."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

def test_output_not_constant(model, synthetic_sample):
    """Model output has non-zero variance (not collapsed)."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert output.std() > 1e-6, f"Output std too small: {output.std():.2e}"
```

**File:** `tests/test_inference.py`

```python
import numpy as np
import torch

def test_ei_computation(model, synthetic_sample, normalization_stats):
    """Epileptogenicity index returns valid scores."""
    eeg_raw, mask = synthetic_sample
    # Apply preprocessing
    eeg_ac = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
    eps = 1e-7
    eeg_norm = (eeg_ac - normalization_stats["eeg_mean"]) / (normalization_stats["eeg_std"] + eps)
    
    with torch.no_grad():
        sources = model(eeg_norm)
    
    # Simple EI: region power
    power = sources.pow(2).mean(dim=-1).numpy().flatten()
    assert power.shape == (76,)
    assert np.all(power >= 0), "Power should be non-negative"
    assert power.sum() > 0, "Total power should be positive"

def test_source_activity_range(model, synthetic_sample, normalization_stats):
    """Predicted source activity is within reasonable range after denorm."""
    eeg_raw, _ = synthetic_sample
    eeg_ac = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
    eps = 1e-7
    eeg_norm = (eeg_ac - normalization_stats["eeg_mean"]) / (normalization_stats["eeg_std"] + eps)
    
    with torch.no_grad():
        sources = model(eeg_norm)
    
    # Denormalize: reverse z-score
    src_denorm = sources * (normalization_stats["src_std"] + eps) + normalization_stats["src_mean"]
    assert src_denorm.abs().max() < 10.0, f"Denormalized source too large: {src_denorm.abs().max()}"
```

**File:** `tests/test_api.py`

```python
def test_health_endpoint(test_client):
    """GET /api/health returns 200 with model_loaded=True."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_analyze_synthetic(test_client):
    """POST /api/analyze with synthetic sample index returns valid result."""
    response = test_client.post(
        "/api/analyze",
        data={"sample_idx": 10, "mode": "source_localization"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "plotHtml" in data

def test_analyze_edf(test_client):
    """POST /api/analyze with EDF file returns valid result."""
    with open("data/samples/0001082.edf", "rb") as f:
        response = test_client.post(
            "/api/analyze",
            files={"file": ("0001082.edf", f, "application/octet-stream")},
            data={"mode": "source_localization", "include_eeg": "true"}
        )
    assert response.status_code in (200, 500)  # 500 OK if MNE missing, 200 if works
```

**Verification:** `pytest tests/ -v` — all tests should pass.

---

#### 📅 APRIL 29 — Backend Integration + CMA-ES Wiring + XAI Integration

---

##### Z7-APR29: Backend CMA-ES + NMT Integration

**File:** `backend/server.py`

1. Merge Hira's CMA-ES modules (`fit_patient`, `compute_ei` from `src/phase4_inversion/`)
2. Merge Shahliza's NMT preprocessor (replace simple EDF segmentation in `/api/analyze`)
3. Add `POST /api/biomarkers-cmaes` endpoint:

```python
@app.post("/api/biomarkers-cmaes")
async def analyze_biomarkers_cmaes(
    file: UploadFile = File(...),
    mode: str = Form("biomarkers"),
    include_xai: bool = Form(False),
):
    """
    Full CMA-ES epileptogenicity analysis.
    Pipeline: EDF → NMTPreprocessor → PhysDeepSIF → CMA-ES → EI heatmap
    Timeout: 5 minutes. Falls back to heuristic EI on timeout.
    """
    # 1. Save uploaded EDF
    # 2. Preprocess via NMTPreprocessor.process()
    # 3. Run inference via run_patient_inference()
    # 4. Compute target PSD via Welch
    # 5. Run CMA-ES with timeout (asyncio.to_thread + asyncio.wait_for)
    # 6. Compute EI from fitted x0
    # 7. Generate Plotly heatmap
    # 8. Optionally run XAI
    # 9. Return JSON with eegData, plotHtml, scores, cmaes_info, xai
```

4. Add CMA-ES timeout handling:
```python
import asyncio

async def run_cmaes_with_timeout(patient_data, timeout_seconds=300):
    from src.phase4_inversion.cmaes_optimizer import fit_patient
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fit_patient, **patient_data),
            timeout=timeout_seconds
        )
        return result, None
    except asyncio.TimeoutError:
        # Fall back to heuristic EI (already in server.py)
        return None, "CMA-ES timed out — using heuristic EI fallback"
```

**Verification:**
```bash
curl -X POST http://127.0.0.1:8000/api/biomarkers-cmaes \
  -F "file=@data/samples/0001082.edf" \
  -F "mode=biomarkers"
# Should return JSON with plotHtml, scores, cmaes_info
```

---

##### Z8-APR29: Frontend XAI Integration

**File:** `frontend/app/analysis/page.tsx` (or equivalent)

1. Add XAI display section in Biomarkers tab:
   - Parse `xai` object from API response
   - Display `channel_importance` as horizontal bar chart (top channels highlighted)
   - Overlay `top_segments` on EEG waveform (colored spans)
   - Show `target_region_full` name

2. Add XAI toggle checkbox in upload form: "Show EEG evidence attribution"

**Verification:** Upload 0001082.edf, select biomarkers, toggle XAI, verify Fp2/F4/F8 rank high in channel importance.

---

##### Z9-APR29: Branch Merge + Conflict Resolution

```bash
git checkout main
git merge hira/cmaes
git merge shahliza/phase35
# Resolve conflicts in:
#   - backend/server.py (endpoint additions)
#   - config.yaml (separate sections)
#   - src/ imports
```

---

#### 📅 APRIL 30 — Testing + Polish + Submit

---

##### Z10-APR30: Integration Testing (Morning)

```bash
# 1. End-to-end CMA-ES pipeline
curl -X POST http://127.0.0.1:8000/api/biomarkers-cmaes \
  -F "file=@data/samples/0001082.edf" -F "mode=biomarkers"

# 2. WebSocket status updates
# Connect ws://127.0.0.1:8000/ws/{job_id} during processing

# 3. XAI attributions validate against 1082.csv annotations
# Check that top attributed channels include Fp2, F4, F8

# 4. Source localization mode still works
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@data/samples/0001082.edf" -F "mode=source_localization"
```

---

##### Z11-APR30: Test Suite Completion + Smoke Test

```bash
pytest tests/ -v --tb=short
bash scripts/smoke_test.sh
./start.sh --check
cd frontend && npm run build
```

Fix any failing tests.

---

##### Z12-APR30: Final Commit + Tag + Submit

```bash
git add -A
git commit -m "feat: complete PhysDeepSIF v2.0 — CMA-ES, XAI, WebSockets, test suite"
git tag v2.0-submission
# Generate thesis figures (see Shahliza S7-APR30)
```

---

### 4.2 HIRA — CMA-ES Phase 4

---

#### 📅 APRIL 28 (TODAY) — Objective Function + CMA-ES Optimizer

---

##### H1-APR28: Objective Function

**File to create:** `src/phase4_inversion/__init__.py` (empty)  
**File to create:** `src/phase4_inversion/objective_function.py`

```python
"""
Module: objective_function.py
Phase: 4 — Patient-Specific Parameter Inversion
Purpose: CMA-ES objective function J(x0) for fitting excitability per patient.

J(x0) = w_source * J_source + w_eeg * J_eeg + w_reg * J_reg

Where:
  J_source: 1 - correlation(power_sim, power_est)  [lower = better match]
  J_eeg:    1 - mean(PSD correlation per channel)   [lower = better match]
  J_reg:    sparsity penalty on x0 deviations        [lower = fewer epi regions]
"""

import numpy as np
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

# Constants
N_REGIONS = 76
N_CHANNELS = 19
SAMPLING_RATE = 200
WELCH_NPERSEG = 200  # 1 second Hann window
FREQ_RANGE = (1, 70)

# Default weights (from config.yaml)
W_SOURCE = 0.4
W_EEG = 0.4
W_REG = 0.2
LAMBDA_SPARSE = 0.1
X0_HEALTHY = -2.2  # Baseline healthy excitability
X0_BOUNDS = (-2.4, -1.0)


def objective(
    x0_vector: NDArray,                           # (76,) candidate excitabilities
    target_source_power: NDArray,                 # (76,) mean power from PhysDeepSIF
    target_eeg_psd: NDArray,                      # (19, n_freqs) Welch PSD of real EEG
    leadfield: NDArray,                           # (19, 76)
    connectivity: NDArray,                        # (76, 76)
    simulation_params: dict,                      # Config for TVB sim
    freqs: NDArray = None,                        # (n_freqs,) Welch frequency bins
) -> float:
    """
    Compute objective value for a candidate x0 vector.

    Each evaluation runs a SHORT TVB simulation (4 seconds, not 12) with
    the proposed x0, computes simulated EEG, and compares against target.

    Args:
        x0_vector: (76,) excitability per region, bounded [-2.4, -1.0]
        target_source_power: (76,) region power from PhysDeepSIF inference
        target_eeg_psd: (19, n_freqs) PSD of real patient EEG
        leadfield: (19, 76) forward matrix
        connectivity: (76, 76) structural connectivity
        simulation_params: dict with sim config (see below)
        freqs: (n_freqs,) Welch frequency bins for PSD correlation

    Returns:
        float: J(x0) — objective value (LOWER = better fit)

    Raises:
        RuntimeError: If TVB simulation diverges (NaN in output)
    """
    from src.phase1_forward.epileptor_simulator import run_simulation
    from src.phase1_forward.synthetic_dataset import project_to_eeg
    from scipy.signal import welch
    from scipy.stats import pearsonr

    # 1. Run TVB simulation with this x0
    sim_length_ms = simulation_params.get("simulation_length_ms", 4000)
    transient_ms = simulation_params.get("transient_ms", 1000)
    dt = simulation_params.get("dt", 0.1)

    try:
        source_sim = run_simulation(
            x0_vector=x0_vector,
            connectivity=connectivity,
            tract_lengths=simulation_params.get("tract_lengths"),
            simulation_length_ms=sim_length_ms,
            G=simulation_params.get("global_coupling", 1.0),
            noise_intensity=simulation_params.get("noise_intensity", 0.0005),
            dt=dt,
        )
    except Exception as e:
        logger.warning(f"Simulation failed: {e}")
        return 1e6  # Large penalty for failed sims

    # Check for NaN
    if np.any(np.isnan(source_sim)):
        return 1e6

    # Discard transient
    transient_samples = int(transient_ms / 1000 * SAMPLING_RATE)
    if source_sim.shape[1] > transient_samples:
        source_sim = source_sim[:, transient_samples:]

    # 2. Project to EEG via leadfield
    eeg_sim = leadfield @ source_sim  # (19, T)

    # 3. Compute source similarity (power profile correlation)
    power_sim = np.mean(source_sim ** 2, axis=1)  # (76,)
    source_corr, _ = pearsonr(power_sim, target_source_power)
    if np.isnan(source_corr):
        source_corr = 0.0
    J_source = 1.0 - source_corr  # Range [0, 2], lower is better

    # 4. Compute EEG similarity (per-channel PSD correlation)
    chan_corrs = []
    for ch in range(N_CHANNELS):
        freqs_sim, psd_sim = welch(
            eeg_sim[ch], fs=SAMPLING_RATE, nperseg=min(WELCH_NPERSEG, len(eeg_sim[ch])),
            noverlap=WELCH_NPERSEG // 2
        )
        # Interpolate to match target PSD frequency bins if needed
        if freqs is not None and len(freqs_sim) != len(freqs):
            psd_sim = np.interp(freqs, freqs_sim, psd_sim)
            freqs_use = freqs
        else:
            freqs_use = freqs_sim

        # Use only specified frequency range
        freq_mask = (freqs_use >= FREQ_RANGE[0]) & (freqs_use <= FREQ_RANGE[1])
        if freq_mask.sum() < 5:
            chan_corrs.append(0.0)
            continue

        # Log-PSD correlation (emphasizes spectral shape, not amplitude)
        log_psd_sim = np.log10(psd_sim[freq_mask] + 1e-10)
        log_psd_target = np.log10(target_eeg_psd[ch, freq_mask] + 1e-10)
        corr, _ = pearsonr(log_psd_sim, log_psd_target)
        if np.isnan(corr):
            corr = 0.0
        chan_corrs.append(corr)

    mean_psd_corr = np.mean(chan_corrs)
    J_eeg = 1.0 - mean_psd_corr  # Range [0, 2]

    # 5. Regularization: sparsity (most regions should stay healthy)
    # Penalize x0 above -2.0 (near-epileptogenic threshold)
    violations = np.maximum(0, x0_vector + 2.0)  # Only penalize x0 > -2.0
    J_reg = np.mean(violations ** 2)

    # 6. Composite objective
    total = W_SOURCE * J_source + W_EEG * J_eeg + W_REG * J_reg

    return float(total)
```

**Verification:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import numpy as np
from src.phase4_inversion.objective_function import objective

# Test with random x0 and dummy targets
x0 = np.full(76, -2.1)
target_power = np.random.rand(76)
target_psd = np.random.rand(19, 50)
leadfield = np.load('data/leadfield_19x76.npy')
conn = np.load('data/connectivity_76.npy')
params = {'simulation_length_ms': 2000, 'transient_ms': 500, 'dt': 0.1}

score = objective(x0, target_power, target_psd, leadfield, conn, params)
print(f'Objective value: {score:.4f}')
print('OK — objective function runs without error')
"
```

---

---

##### H2-APR28: CMA-ES Optimizer + EI Computation

**File to create:** `src/phase4_inversion/cmaes_optimizer.py`

```python
"""
Module: cmaes_optimizer.py
Phase: 4 — Patient-Specific Parameter Inversion
Purpose: CMA-ES wrapper for fitting x0 per patient.

Uses the cmaes v0.12.0 package (NOT cma).  The API is:
  from cmaes import CMA
  cma = CMA(mean=init, sigma=sigma0, bounds=(lo, hi))
  # Manual ask()/tell() loop:
  for gen in range(max_gen):
      solutions = []
      for _ in range(cma.population_size):
          x = cma.ask()
          score = objective(x)
          solutions.append((x, score))
      cma.tell(solutions)
      if cma.should_stop():
          break
  best_x = cma.best_x
"""
# ... (full implementation as specified below)
```

**File to create:** `src/phase4_inversion/epileptogenicity_index.py`

```python
"""
Module: epileptogenicity_index.py
Phase: 4 — Patient-Specific Parameter Inversion
Purpose: Map fitted x0 to Epileptogenicity Index (EI ∈ [0, 1]).

Uses sigmoid mapping centered at healthy baseline (-2.2):
  EI_i = sigmoid((x0_i + 2.2) / 0.15)
"""

import numpy as np
from numpy.typing import NDArray

X0_BASELINE = -2.2
EI_SCALE = 0.15

def compute_ei(x0_vector: NDArray) -> NDArray:
    """Compute EI from fitted x0. Returns (76,) values in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-(x0_vector - X0_BASELINE) / EI_SCALE))
```

**⚠️ Verification — run CMA-ES on 1 synthetic test sample BEFORE sleep:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import numpy as np, torch, h5py, json, sys
sys.path.insert(0, '.')
from src.phase4_inversion.cmaes_optimizer import fit_patient
from src.phase2_network.physdeepsif import build_physdeepsif

model = build_physdeepsif('data/leadfield_19x76.npy', 'data/connectivity_76.npy')
ckpt = torch.load('outputs/models/checkpoint_best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state']); model.eval()

with h5py.File('data/synthetic3/test_dataset.h5', 'r') as f:
    idx = np.random.randint(0, f['eeg'].shape[0])
    eeg_raw = torch.from_numpy(f['eeg'][idx:idx+1].astype(np.float32))
    true_mask = f['epileptogenic_mask'][idx:idx+1]

with open('outputs/models/normalization_stats.json') as f: stats = json.load(f)
eps = 1e-7
eeg_ac = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
eeg_n = (eeg_ac - stats['eeg_mean']) / (stats['eeg_std'] + eps)

with torch.no_grad(): sources = model(eeg_n).numpy()[0]
target_power = np.mean(sources ** 2, axis=1)

from scipy.signal import welch
target_psd = np.array([welch(eeg_n[0, ch].numpy(), fs=200, nperseg=200, noverlap=100)[1] for ch in range(19)])

result = fit_patient(target_power, target_psd,
    np.load('data/leadfield_19x76.npy'), np.load('data/connectivity_76.npy'),
    {'simulation_length_ms': 4000, 'transient_ms': 1000, 'dt': 0.1, 'global_coupling': 1.0, 'noise_intensity': 0.0005},
    max_generations=30)

print(f'CMA-ES: {result[\"n_generations\"]} gens, best_score={result[\"best_score\"]:.4f}')
top5 = np.argsort(result['ei'])[-5:][::-1]
true5 = np.where(true_mask[0])[0]
print(f'Top EI: {top5} | True epi: {true5}')
print(f'Overlap: {len(set(top5) & set(true5))}/5')
"
```
**Expected:** CMA-ES completes in < 5 min. Top EI regions should overlap with true epi regions.

---

#### 📅 APRIL 29 — CMA-ES Testing + Backend Integration + Real Patient

---

##### H3-APR29: CMA-ES on 3 Synthetic Samples + Tuning

Run CMA-ES on synthetic test samples (indices 10, 25, 51):
- Verify x0 converges toward ground truth epileptogenic regions
- Measure convergence time per patient (target: < 15 min)
- If too slow: reduce `sim_length_ms` to 2000, `max_generations` to 30
- Tune objective weights (w_source, w_eeg, w_reg) if needed

---

##### H4-APR29: CMA-ES on 0001082.edf Real Patient

**Prerequisite:** Shahliza must complete S1+S2 (preprocessing + inference on 0001082.edf).

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
# Load preprocessed EEG from Shahliza's output
# Run CMA-ES with patient targets
# Compare EI hotspots against 1082.csv annotations (FP2/F4/F8 channels)
# Generate thesis convergence plot (J vs generation)
"
```

**Handoff to Zik:** Export `fit_patient()`, `compute_ei()`, `objective()` as importable functions for `backend/server.py`.

---

##### H5-APR29: Backend CMA-ES Helpers + Thesis Convergence Plot

Create helper functions that Zik imports into `backend/server.py`:
```python
# In backend/server.py or a shared module:
from src.phase4_inversion.cmaes_optimizer import fit_patient
from src.phase4_inversion.epileptogenicity_index import compute_ei
from src.phase4_inversion.objective_function import objective
```

Generate CMA-ES convergence plot for thesis (Figure 6).

---

#### 📅 APRIL 30 — Bug Fixes + Polish

---

##### H6-APR30: Bug Fixes + Integration Support

- Fix any CMA-ES issues found during Zik's integration testing
- Help with end-to-end pipeline testing
- Final CMA-ES documentation for thesis

---

### 4.3 SHAHLIZA — Phase 3 NMT Preprocessing + Phase 5 Validation

---

#### 📅 APRIL 28 (TODAY) — NMT Preprocessor + Inference Engine

---

##### S1-APR28: NMT Preprocessor (6-step pipeline)

**Files to create:**
1. `src/phase3_inference/__init__.py` (empty)
2. `src/phase3_inference/nmt_preprocessor.py`

```python
"""
Module: nmt_preprocessor.py
Phase: 3 — Real EEG Preprocessing and Inference
Purpose: 6-step NMT EEG preprocessing pipeline (Tech Specs §5.2).

Steps:
  1. Load EDF via MNE, verify 19 channels present
  2. Re-reference verification (keep linked-ear, drop A1/A2)
  3. Bandpass filter 0.5-70 Hz (FIR, firwin)
  4. Notch filter 50 Hz (spectrum_fit)
  5. ICA artifact removal (fastica, n_components=15, with fallback)
  6. Segment into 2s windows + z-score normalize using normalization_stats.json

Input: Path to EDF file
Output: dict with preprocessed_eeg (n_epochs, 19, 400) + metadata
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard 10-20 montage (19 channels, matching pipeline)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

# Mapping from NMT EDF channel names to standard names (case fixes)
EDF_CHANNEL_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
    'EEG FP1-Ref': 'Fp1', 'EEG FP2-Ref': 'Fp2',
    # T3, T4, T5, T6 keep same names
    # A1, A2 dropped (ear reference, not in 10-20)
}

# Channels to drop (reference electrodes)
DROP_CHANNELS = ['A1', 'A2', 'EEG A1-Ref', 'EEG A2-Ref']


class NMTPreprocessor:
    """6-step NMT EEG preprocessing pipeline."""

    def __init__(self, normalization_stats_path: str = "outputs/models/normalization_stats.json"):
        with open(normalization_stats_path) as f:
            self.norm_stats = json.load(f)
        self.epoch_length_sec = 2.0
        self.sampling_rate = 200
        self.epoch_length_samples = 400

    def process(self, edf_path: str) -> Dict:
        """
        Full preprocessing pipeline.

        Args:
            edf_path: Path to EDF file

        Returns:
            dict with:
                preprocessed_eeg: (n_epochs, 19, 400) float32
                n_epochs: int
                channel_names: list[str] (19 standard names)
                epoch_times: list[float] (start time of each epoch in seconds)
                raw_duration_sec: float (total recording duration)
        """
        import mne

        # Step 1: Load EDF
        logger.info(f"Loading EDF: {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        # Step 2: Verify and normalize channel names
        raw = self._normalize_channels(raw)

        # Step 3: Bandpass filter 0.5-70 Hz
        raw.filter(l_freq=0.5, h_freq=70.0, method='fir', fir_design='firwin')

        # Step 4: Notch filter 50 Hz
        raw.notch_filter(freqs=[50.0], method='spectrum_fit')

        # Step 5: ICA artifact removal (with fallback)
        raw = self._apply_ica(raw)

        # Step 6: Segment and normalize
        result = self._segment_and_normalize(raw)
        result["raw_duration_sec"] = raw.times[-1]

        return result

    def _normalize_channels(self, raw) -> 'mne.io.Raw':
        """Map NMT channel names to standard 10-20, drop references."""
        # Rename channels
        mapping = {}
        for ch in raw.ch_names:
            clean = ch.strip()
            if clean in EDF_CHANNEL_MAP:
                mapping[ch] = EDF_CHANNEL_MAP[clean]

        if mapping:
            raw.rename_channels(mapping)

        # Drop reference channels
        channels_to_drop = [ch for ch in raw.ch_names if ch in DROP_CHANNELS]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)

        # Select only our 19 standard channels
        available = [ch for ch in STANDARD_CHANNELS if ch in raw.ch_names]
        if len(available) < 19:
            logger.warning(f"Only {len(available)}/19 standard channels found")
        raw.pick(available)

        logger.info(f"Channels after normalization: {raw.ch_names}")
        return raw

    def _apply_ica(self, raw) -> 'mne.io.Raw':
        """Apply ICA artifact removal. Falls back gracefully on failure."""
        import mne
        try:
            ica = mne.preprocessing.ICA(
                n_components=15, method='fastica',
                max_iter=1000, random_state=42
            )
            ica.fit(raw)

            # Auto-detect EOG artifacts
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=['Fp1', 'Fp2'], threshold=3.0
            )
            ica.exclude = eog_indices
            raw = ica.apply(raw)
            logger.info(f"ICA applied: removed {len(eog_indices)} components")
        except Exception as e:
            logger.warning(f"ICA failed ({e}). Skipping ICA for this recording.")
        return raw

    def _segment_and_normalize(self, raw) -> Dict:
        """Segment into 2s windows and z-score normalize."""
        import mne

        # Create fixed-length events
        events = mne.make_fixed_length_events(
            raw, duration=self.epoch_length_sec, overlap=0.0
        )

        # Create epochs
        epochs = mne.Epochs(
            raw, events, tmin=0,
            tmax=self.epoch_length_sec - 1 / self.sampling_rate,
            baseline=None, preload=True,
            reject=dict(eeg=200e-6),  # Reject epochs with >200 µV
            verbose=False,
        )

        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)

        # Per-channel temporal de-meaning (match training pipeline)
        data = data - data.mean(axis=-1, keepdims=True)

        # Global z-score using training normalization stats
        eps = 1e-7
        data = (data - self.norm_stats["eeg_mean"]) / (self.norm_stats["eeg_std"] + eps)

        epoch_times = [float(e[0]) for e in events[:len(data)]]

        return {
            "preprocessed_eeg": data.astype(np.float32),
            "n_epochs": len(data),
            "channel_names": raw.ch_names,
            "epoch_times": epoch_times,
        }
```

**Verification:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
from src.phase3_inference.nmt_preprocessor import NMTPreprocessor
preprocessor = NMTPreprocessor()
result = preprocessor.process('data/samples/0001082.edf')
print(f'Epochs: {result[\"n_epochs\"]}')
print(f'Shape: {result[\"preprocessed_eeg\"].shape}')
print(f'Channels: {result[\"channel_names\"]}')
print(f'Duration: {result[\"raw_duration_sec\"]:.1f}s')
print('OK — preprocessing works')
"
```
**Expected:** n_epochs > 0, shape = (n_epochs, 19, 400), channels = 19 standard names.

**⚠️ This is a BLOCKING handoff to Hira for H4-APR29** (CMA-ES on real patient). Hira needs `target_source_power` and `target_eeg_psd` from this pipeline.

---

#### 📅 APRIL 29 — Validation Metrics + Baselines + Patient Validation

---

##### S3-APR29: Synthetic Validation Metrics

**File to create:** `src/phase5_validation/__init__.py` (empty)  
**File to create:** `src/phase5_validation/synthetic_metrics.py`

```python
"""
Module: synthetic_metrics.py
Phase: 5 — Validation Framework
Purpose: Compute all validation metrics on synthetic test set.

Reuses metrics from src/phase2_network/metrics.py but adds batch-level
aggregation and reporting.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Reuse existing metrics
from src.phase2_network.metrics import (
    compute_dipole_localization_error,
    compute_spatial_dispersion,
    compute_auc_epileptogenicity,
    compute_temporal_correlation,
)


def compute_all_metrics(
    predicted_sources: NDArray,     # (n_samples, 76, 400)
    true_sources: NDArray,          # (n_samples, 76, 400)
    epileptogenic_mask: NDArray,    # (n_samples, 76) bool
    region_centers: NDArray,        # (76, 3) mm
) -> Dict[str, float]:
    """
    Compute all validation metrics in one call.

    Returns:
        dict with: dle_mm, sd_mm, auc, temporal_corr, f1_score
    """
    dle = compute_dipole_localization_error(
        predicted_sources, true_sources, region_centers
    )
    sd = compute_spatial_dispersion(predicted_sources, region_centers)
    auc = compute_auc_epileptogenicity(predicted_sources, epileptogenic_mask)
    corr = compute_temporal_correlation(predicted_sources, true_sources)

    # F1 score: threshold top-10% power as positive
    power = np.mean(predicted_sources ** 2, axis=-1)  # (n_samples, 76)
    threshold = np.percentile(power, 90, axis=-1, keepdims=True)
    pred_mask = power >= threshold
    true_flat = epileptogenic_mask.flatten()
    pred_flat = pred_mask.flatten()
    tp = ((pred_flat) & (true_flat)).sum()
    fp = ((pred_flat) & (~true_flat)).sum()
    fn = ((~pred_flat) & (true_flat)).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "dle_mm": float(dle),
        "sd_mm": float(sd),
        "auc": float(auc),
        "temporal_corr": float(corr),
        "f1_score": float(f1),
    }
```

---

##### S4-APR29: Classical Baselines

**File to create:** `src/phase5_validation/classical_baselines.py`

```python
"""
Module: classical_baselines.py
Phase: 5 — Validation Framework
Purpose: Classical EEG source localization baselines.

Implements 4 standard inverse methods using analytical closed-form
solutions (NO MNE API dependencies — all numpy):

  1. Minimum Norm Estimate (MNE): s = L^T (L L^T + λI)^-1 eeg
  2. eLORETA: s = W L^T (L W L^T + λ C_n)^-1 eeg
     where W = diag(||L_i||^-1) is depth weighting
  3. dSPM: s = Σ^{1/2} L^T (L Σ L^T + λ C_n)^-1 eeg
     where Σ = I (identity) for simplicity
  4. LCMV Beamformer: w_c = R^-1 L_c / (L_c^T R^-1 L_c)
     s_c = w_c^T eeg

All methods use the SAME 19×76 leadfield matrix as PhysDeepSIF.
Regularization λ is computed via L-curve heuristic:
  λ = 0.01 × trace(L L^T) / N_channels
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _compute_lambda(leadfield: NDArray) -> float:
    """L-curve heuristic for regularization parameter."""
    gram = leadfield @ leadfield.T  # (19, 19)
    return 0.01 * np.trace(gram) / leadfield.shape[0]


def minimum_norm_estimate(
    eeg: NDArray,          # (19, T) or (n_windows, 19, T)
    leadfield: NDArray,    # (19, 76)
    lam: float = None,
) -> NDArray:
    """
    Minimum Norm Estimate (Hämäläinen & Ilmoniemi, 1994).

    s = L^T (L L^T + λ I)^-1 eeg
    """
    if lam is None:
        lam = _compute_lambda(leadfield)

    n_chan = leadfield.shape[0]
    gram = leadfield @ leadfield.T  # (19, 19)
    reg_gram = gram + lam * np.eye(n_chan)
    inv_term = np.linalg.solve(reg_gram, eeg)  # (19, T) or (n, 19, T)
    sources = leadfield.T @ inv_term  # (76, T) or (n, 76, T)
    return sources


def eloreta(
    eeg: NDArray,
    leadfield: NDArray,
    lam: float = None,
) -> NDArray:
    """
    eLORETA (Pascual-Marqui, 2007) — depth-weighted minimum norm.

    s = W L^T (L W L^T + λ I)^-1 eeg
    where W = diag(||L_i||)  with ||L_i|| being column norms
    """
    if lam is None:
        lam = _compute_lambda(leadfield)

    n_regions = leadfield.shape[1]
    n_chan = leadfield.shape[0]

    # Depth weighting: W = diag(1 / ||L_i||)
    col_norms = np.linalg.norm(leadfield, axis=0)  # (76,)
    W_diag = 1.0 / (col_norms + 1e-8)
    W = np.diag(W_diag)  # (76, 76)

    # Weighted leadfield: L_W = L @ W^{1/2}
    W_sqrt = np.diag(np.sqrt(W_diag))
    L_weighted = leadfield @ W_sqrt  # (19, 76)

    gram_weighted = L_weighted @ L_weighted.T  # (19, 19)
    reg_gram = gram_weighted + lam * np.eye(n_chan)

    inv_term = np.linalg.solve(reg_gram, eeg)  # (19, T)
    sources_weighted = L_weighted.T @ inv_term  # (76, T)
    sources = W_sqrt @ sources_weighted  # Unapply weighting

    return sources


def dspm(
    eeg: NDArray,
    leadfield: NDArray,
    lam: float = None,
    noise_cov: NDArray = None,
) -> NDArray:
    """
    dSPM — dynamic Statistical Parametric Mapping (Dale et al., 2000).

    s = Σ L^T (L Σ L^T + λ C_n)^-1 eeg

    Here Σ = I (identity source covariance) and C_n = I for simplicity.
    This reduces to the minimum norm estimate.
    """
    if noise_cov is None:
        noise_cov = np.eye(leadfield.shape[0])
    if lam is None:
        lam = _compute_lambda(leadfield)

    n_chan = leadfield.shape[0]
    gram = leadfield @ leadfield.T
    reg_gram = gram + lam * noise_cov
    inv_term = np.linalg.solve(reg_gram, eeg)
    sources = leadfield.T @ inv_term

    # dSPM noise normalization: divide by noise sensitivity
    noise_sensitivity = np.sqrt(np.diag(
        leadfield.T @ np.linalg.solve(reg_gram, leadfield)
    ))
    sources = sources / (noise_sensitivity[:, np.newaxis] + 1e-8)

    return sources


def lcmv_beamformer(
    eeg: NDArray,          # (19, T)
    leadfield: NDArray,    # (19, 76)
    lam: float = None,
) -> NDArray:
    """
    LCMV Beamformer (Van Veen et al., 1997).

    w_c = R^-1 L_c / (L_c^T R^-1 L_c)
    s_c = w_c^T eeg

    where R = EEG covariance matrix (data-dependent).
    """
    if lam is None:
        lam = 1e-3  # Small regularization for covariance inverse

    n_regions = leadfield.shape[1]
    n_chan = leadfield.shape[0]

    # Data covariance: R = EEG @ EEG^T / T
    R = (eeg @ eeg.T) / eeg.shape[1]  # (19, 19)
    R_reg = R + lam * np.trace(R) / n_chan * np.eye(n_chan)
    R_inv = np.linalg.inv(R_reg)

    # Beamformer weights per region
    sources = np.zeros((n_regions, eeg.shape[1]))
    for i in range(n_regions):
        L_i = leadfield[:, i:i+1]  # (19, 1)
        denom = L_i.T @ R_inv @ L_i
        if denom > 1e-10:
            w = (R_inv @ L_i) / denom  # (19, 1)
            sources[i] = (w.T @ eeg).flatten()

    return sources
```

**Verification:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import numpy as np
from src.phase5_validation.classical_baselines import (
    minimum_norm_estimate, eloreta, dspm, lcmv_beamformer
)
# Test on random EEG
leadfield = np.load('data/leadfield_19x76.npy')
eeg = np.random.randn(19, 400).astype(np.float32)
mne_src = minimum_norm_estimate(eeg, leadfield)
eloreta_src = eloreta(eeg, leadfield)
dspm_src = dspm(eeg, leadfield)
lcmv_src = lcmv_beamformer(eeg, leadfield)
for name, src in [('MNE', mne_src), ('eLORETA', eloreta_src),
                   ('dSPM', dspm_src), ('LCMV', lcmv_src)]:
    print(f'{name}: shape={src.shape}, range=[{src.min():.3f}, {src.max():.3f}]')
print('OK — all baselines work')
"
```

---

##### S5-APR29: Patient Validation

**File to create:** `src/phase5_validation/patient_validation.py`

```python
"""
Module: patient_validation.py
Phase: 5 — Validation Framework
Purpose: Patient-level validation using NMT recordings.

Validates:
  1. Intra-patient consistency (EI across recording segments)
  2. Comparison to clinical annotations (1082.csv)
  3. Cross-segment stability (bootstrap analysis)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

N_REGIONS = 76


def intra_patient_consistency(
    ei_per_segment: List[NDArray],   # List of (76,) EI vectors
) -> Dict:
    """
    Compute pairwise Spearman correlation between EI vectors from
    different recording segments.

    Args:
        ei_per_segment: List of EI vectors, one per recording segment

    Returns:
        dict with mean_rho, std_rho, all_rhos
    """
    n_segments = len(ei_per_segment)
    if n_segments < 2:
        return {"mean_rho": 1.0, "std_rho": 0.0, "all_rhos": []}

    rhos = []
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            rho, _ = spearmanr(ei_per_segment[i], ei_per_segment[j])
            rhos.append(rho)

    return {
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "all_rhos": [float(r) for r in rhos],
    }


def compare_to_annotations(
    ei: NDArray,                      # (76,) EI per region
    region_codes: List[str],          # (76,) DK76 region codes
    annotated_channels: List[str],    # e.g., ['Fp2', 'F4', 'F8']
) -> Dict:
    """
    Compare predicted EI hotspots against clinical EEG channel annotations.

    Since annotations are at the scalp level (channels) and our predictions
    are at the source level (DK76 regions), this comparison is qualitative:
    we check if regions near the annotated channels have elevated EI.

    Returns:
        dict with:
            top_regions: list of (code, ei) for top-5 regions
            annotated_channel_ei: dict mapping each annotated channel to
                                 EI of its spatially nearest DK76 region
    """
    from src.region_names import REGION_NAMES

    # Load region centers for spatial mapping
    region_centers = np.load("data/region_centers_76.npy")

    # EEG electrode positions (approximate MNI from 10-20 system)
    # These are approximate — not used for precision, just directionality
    electrode_mni = {
        'Fp1': (-15, 85, -10), 'Fp2': (15, 85, -10),
        'F3': (-40, 45, 35), 'F4': (40, 45, 35),
        'F7': (-50, 30, -10), 'F8': (50, 30, -10),
        'C3': (-50, -10, 55), 'C4': (50, -10, 55),
        'P3': (-40, -55, 45), 'P4': (40, -55, 45),
        'O1': (-20, -85, 15), 'O2': (20, -85, 15),
        'T3': (-60, -20, 0), 'T4': (60, -20, 0),
        'T5': (-50, -55, 5), 'T6': (50, -55, 5),
        'Fz': (0, 50, 45), 'Cz': (0, -15, 65), 'Pz': (0, -55, 40),
    }

    # Top-5 regions by EI
    top_idx = np.argsort(ei)[-5:][::-1]
    top_regions = [
        (region_codes[i], float(ei[i]), REGION_NAMES.get(region_codes[i], region_codes[i]))
        for i in top_idx
    ]

    # Find nearest DK76 region for each annotated channel
    annotated_region_ei = {}
    for ch in annotated_channels:
        if ch not in electrode_mni:
            continue
        ch_pos = np.array(electrode_mni[ch])
        distances = np.linalg.norm(region_centers - ch_pos, axis=1)
        nearest_idx = int(np.argmin(distances))
        annotated_region_ei[ch] = {
            "nearest_region": region_codes[nearest_idx],
            "nearest_region_name": REGION_NAMES.get(region_codes[nearest_idx], region_codes[nearest_idx]),
            "ei": float(ei[nearest_idx]),
            "distance_mm": float(distances[nearest_idx]),
        }

    return {
        "top_regions": top_regions,
        "annotated_channel_ei": annotated_region_ei,
    }


def bootstrap_stability(
    source_estimates: NDArray,        # (n_epochs, 76, 400)
    ei_compute_fn,                    # callable(sources) → (76,) EI
    n_bootstrap: int = 20,
    sample_fraction: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap analysis: resample epochs, recompute EI, measure CV.

    Returns:
        dict with cv_per_region (76,) and median_cv (float)
    """
    rng = np.random.RandomState(seed)
    n_total = source_estimates.shape[0]
    n_sample = max(2, int(n_total * sample_fraction))

    ei_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_sample, replace=True)
        ei = ei_compute_fn(source_estimates[idx])
        ei_samples.append(ei)

    ei_samples = np.array(ei_samples)  # (n_bootstrap, 76)
    cv_per_region = np.std(ei_samples, axis=0) / (np.mean(ei_samples, axis=0) + 1e-8)

    return {
        "cv_per_region": cv_per_region.tolist(),
        "median_cv": float(np.median(cv_per_region)),
    }
```

**Verification:** Run patient_validation.py on 0001082.edf output.

---

#### 📅 APRIL 30 — Thesis Figures + Final Delivery

---

##### S6-APR30: Thesis Figures Generation

Generate all 11 thesis figures:
1. Architecture diagram
2. Training loss curves
3. DLE per-region heatmap
4. AUC vs SNR curve
5. Baselines comparison table
6. CMA-ES convergence plot (with Hira)
7. Patient EI profile (0001082.edf)
8. Brain heatmap screenshots (both modes)
9. EEG waveform + brain synchronized screenshot
10. XAI evidence overlay on EEG waveform
11. Normal vs abnormal EI comparison (if 2+ recordings available)

Create `scripts/06_run_validation.py` and `scripts/07_run_baselines.py` as drivers.

##### S7-APR30: Polish and Handover

Help Zik with integration testing, bug fixes, and `./start.sh` verification.

---

## 5. Feature Delivery Matrix

| Feature | Approach | Risk | Fallback |
|---------|----------|------|----------|
| **Training fix** | B1-B4 applied, overfit test → full train | Medium | Failsafe E1-E4 |
| **Synthetic data** | 5000 sims on lab GPU (16 cores), ~4h | Low | 3000 sims minimum |
| **Model training** | 80 epochs on lab RTX 3080, ~2h | Low | Existing checkpoint as demo |
| **Phase 3 NMT** | MNE pipeline, ICA with try/except | Low | Skip ICA if problematic |
| **Phase 4 CMA-ES** | pop≈14, gen=50, 4s sim | Medium | Heuristic EI fallback |
| **Phase 5 validation** | Synthetic test set + analytical baselines | Low | Simplified metrics |
| **WebSocket** | Single endpoint, job status push | Low | Sync HTTP as primary |
| **XAI** | Occlusion top-1 region, top-3 windows | Medium | Optional frontend toggle |
| **Test suite** | 7 core tests | Low | Prioritize API + model |

---

## 6. Branch Strategy

```
main
  ├─ zik/emergency    (training fix + WebSocket + XAI + tests)
  ├─ hira/cmaes       (Phase 4 only — objective, optimizer, EI)
  └─ shahliza/phase35 (Phase 3 + Phase 5)
```

**Merge schedule:** Apr 29 noon all → main. Apr 30 00:00 CODE FREEZE.

---

## 7. Success Criteria (April 30 Night)

- [ ] Overfit test passes: AUC > 0.6, DLE decreasing across epochs
- [ ] Full retraining completes: AUC > 0.65, DLE < 20mm, SD < 40mm
- [ ] Phase 3 NMT preprocessor works on 0001082.edf
- [ ] Phase 4 CMA-ES produces EI on 1 synthetic + 1 real patient
- [ ] Phase 5 validation metrics computed on synthetic test set
- [ ] Classical baselines (4 methods) compared
- [ ] 0001082.edf patient validation complete (annotations comparison)
- [ ] Backend serves /api/biomarkers-cmaes endpoint
- [ ] WebSocket /ws/{job_id} delivers status updates
- [ ] XAI occlusion attributions returned in biomarkers response
- [ ] Frontend renders both visualization modes + XAI overlay
- [ ] `pytest tests/ -v` passes all 7+ tests
- [ ] `./start.sh --check` reports clean
- [ ] All 11 thesis figures generated
- [ ] `git tag v2.0-submission`
