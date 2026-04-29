# Training Stall Diagnostics & Fix Protocol

**Date**: 2026-04-29
**Status**: Approved for implementation
**Priority**: Critical — blocks all training progress

---

## Executive Summary

Training stalls after epoch 1 with validation loss increasing monotonically (2.29 → 2.49) and temporal correlation collapsing (0.226 → 0.003). Root cause: **gradient starvation** — the forward loss gradient dominates all other gradients through leadfield amplification (~560× spectral norm), and gradient clipping at `max_norm=1.0` eliminates source/epi loss contributions.

**Solution**: Per-component gradient normalization ensures each loss contributes proportionally to its weight regardless of gradient magnitude.

---

## 1. Mathematical Root Cause Analysis

### 1.1 The Gradient Starvation Mechanism

Training stalls because the forward loss gradient **starves** the source and epi loss gradients through the interaction of three factors: leadfield amplification, gradient clipping, and the underdetermined inverse problem.

**The gradient chain** for L_forward passes through L^T (19×76 → 76×76 via source gradient smoothing). The leadfield matrix `L ∈ ℝ^{19×76}` has:
- Effective rank: 18 (of 19, minus 1 for reference)
- Spectral norm: ||L||₂ ≈ 560
- Frobenius norm: ||L||_F ≈ 862
- Condition number: ~10¹⁶

The forward loss gradient w.r.t. predicted sources `Ŝ ∈ ℝ^{76×400}` is:

```
∂L_fwd/∂Ŝ = L^T · (2/N) · (AC(L@Ŝ) - EEG_ac) / (Var(EEG) + Var(L@Ŝ))
```

The variance normalization makes L_fwd ≈ O(1) in **value**, but the **gradient** magnitude scales with ||L^T||. Quantitative estimate at random init:

- ||∂L_fwd/∂Ŝ|| ≈ ||L||₂ · ||ε|| / denom ≈ 560 · 87 / 2 ≈ 24,360
- ||∂L_src/∂Ŝ|| ≈ (2/N) · ||Ŝ - S_true|| ≈ 0.023

**Ratio: ||∂L_fwd/∂Ŝ|| / ||∂L_src/∂Ŝ|| ≈ 10⁶**

Even with β = 0.1, the effective ratio is β × 10⁶ ≈ 10⁵. After gradient clipping at `max_norm=1.0`, the combined gradient direction is >99.99% from the forward loss. The source and epi losses contribute <0.01% of the update direction — they are **starved**.

### 1.2 Why the Overfit Test Appeared to Validate β=0.1

The overfit test used 40 samples with full-batch GD. With N=40:
- Per-sample source gradient variance is extremely high (∝ 1/√N ≈ 0.16)
- Occasional large source gradient "kicks" can overcome the forward gradient direction
- Overfitting doesn't require generalization — the model memorizes 40 patterns regardless

This does NOT validate that β=0.1 works for stochastic training with 23k samples and batch_size=64.

### 1.3 Evidence from the Training Log

| Metric | Epoch 1 (β≈0) | Epoch 6 (β=0.5) | Epoch 31 | Interpretation |
|--------|-------|---------|-------|---------------|
| L_src | 1.972 | 2.021 | 2.037 | **Increasing** — model gets WORSE at source reconstruction |
| L_fwd | 0.998 | 0.088 | 0.027 | **Collapsing** — model tightly fits forward model |
| L_epi | 0.321 | 0.455 | 0.436 | **Increasing** — epi detection degrades |
| Corr | 0.226 | 0.018 | 0.003 | **Collapsed** — temporal dynamics lost |
| AUC | 0.567 | 0.573 | 0.567 | **Near chance** — epi regions undetected |

The model converges to the **minimum-norm estimate (MNE)** attractor: Ŝ ≈ (L^T L)⁻¹ L^T EEG. This is the well-known generalized inverse that minimizes forward error, but it's spatially smooth and lacks temporal precision — exactly matching the observed symptoms (low correlation, near-chance AUC, low SD).

### 1.4 The 58-Dimensional Null Space Problem

L ∈ ℝ^{19×76} with rank 18 means the null space of L has dimension 76 - 18 = **58 dimensions**. The forward gradient only influences 18 of 76 source dimensions. The remaining 58 must be learned entirely from L_src and L_epi — but gradient clipping eliminates their contribution.

Concretely: the source gradient ∂L_src/∂Ŝ has components in ALL 76 dimensions (since S_true provides full supervision), but after clipping, the 18 dimensions are dominated by L_fwd, and the 58 null-space dimensions receive no gradient at all (zero from L_fwd, near-zero from clipped L_src/L_epi).

### 1.5 Current β=0.5 Analysis

The config comment says β was "increased from 0.05 to 0.1" (validated in overfit test), but the actual value is 0.5. This was an intentional attempt to strengthen forward consistency. Given the gradient ratio of ~10⁶, ANY β > 0 will cause gradient starvation with standard clipping. The difference between β=0.01, 0.05, 0.1, and 0.5 is the **rate** of convergence to the MNE attractor, not whether it converges. At β=0.5 the model reaches MNE by epoch 2; at β=0.1 it takes longer but still converges there.

---

## 2. Diagnosis Protocol

### Phase D1: Gradient Norm Audit (15 min, no GPU training)

**Script**: `scripts/diag_gradient_audit.py`

Run on 8 samples from val set with a fresh model, measuring:

```python
For each loss component L_i and each epoch 0-4:
  1. Compute ∂L_i/∂θ (all model parameter gradients)
  2. Compute ||∂L_i/∂θ||₂ (total gradient norm)
  3. Compute per-layer gradient norms (spatial vs temporal module)
  4. After combined backward + clip, compute effective gradient share:
     share_i = ||α_i · ∂L_i/∂θ||₂ / Σ_j ||α_j · ∂L_j/∂θ||₂
  5. Repeat for β ∈ {0.01, 0.05, 0.1, 0.5}
```

**Expected outcome**: Even at β=0.01, the forward gradient share will be >90% due to ||L||₂ ≈ 560 amplification. This confirms gradient starvation is the root cause regardless of β.

### Phase D2: β Sweep (2 hours, GPU)

Run 5-epoch diagnostic training for β ∈ {0.001, 0.01, 0.05, 0.1, 0.5}:

```
For each β:
  - Train 5 epochs, log L_src, L_fwd, Corr, AUC per epoch
  - Key metric: is L_src DECREASING? (it currently INCREASES)
  - Plot gradient share per component from Phase D1
```

**Expected outcome**: Even at β=0.01, L_src may still increase slightly. At β=0.001, L_src should decrease but L_fwd provides almost no constraint (L_fwd ≈ MSE/Var ≈ 1.0 with minimal gradient).

### Phase D3: Per-Component Gradient Normalization (30 min to implement, 1 hour to test)

**This is the principled fix.** Instead of a single combined backward pass with total gradient clipping, normalize each loss component's gradient independently before combining:

```python
# In trainer.py _train_epoch:
component_losses = {
    'source': alpha * loss_source,
    'forward': beta * loss_forward, 
    'physics': gamma * loss_physics,
    'epi': delta_epi * loss_epi,
}

total_grad_norm = 0.0
for loss_name, loss_val in component_losses.items():
    self.model.zero_grad()
    loss_val.backward(retain_graph=True)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), max_norm=float('inf')
    )
    # Record gradient norm for logging
    # Scale gradient to have unit norm, then re-weight
    scale = 1.0 / max(grad_norm, 1e-8)
    for p in self.model.parameters():
        if p.grad is not None:
            p.grad.data *= scale  # normalize to unit norm
            # Store for later accumulation
    
total_grad = sum of all normalized ∙ weighted gradients
clip total_grad to max_norm
optimizer.step()
```

**Scientific justification**: This ensures each loss component contributes proportionally to its weight α, β, γ, δ regardless of gradient magnitude. The forward gradient still passes through L^T, but after normalization it has the same norm as the source gradient. The null-space dimensions (58 out of 76) receive gradient only from source and epi losses, which are now not starved.

**Alternative (simpler)**: Compute per-component gradient norms and log them. Then dynamically adjust β so that the forward gradient norm matches the source gradient norm:

```python
if epoch > 0:
    beta_effective = beta * target_fwd_src_ratio * src_grad_norm / fwd_grad_norm
```

This is a form of adaptive gradient scaling (similar to GradNorm from Chen et al., 2018).

### Phase D4: Validation of Fix (2-3 hours, GPU)

After implementing the gradient normalization from Phase D3:
- Run full training for 80 epochs with β=0.1 (as originally planned)
- Key success criteria:
  - L_src monotonically decreasing
  - Corr increasing (target > 0.3 by epoch 30)
  - AUC increasing (target > 0.65 by epoch 30)
  - L_fwd stays near O(1) (variance normalization working correctly)

---

## 3. Proposed Code Changes

### Change 1: Implement Per-Component Gradient Normalization

**File**: `src/phase2_network/trainer.py` — `_train_epoch` method

Replace the current combined backward + clip with per-component normalization:

```python
# Current (broken):
loss_dict = self.loss_fn(source_pred, sources, eeg_augmented, mask, epoch=self.current_epoch)
loss = loss_dict['loss_total']
loss.backward()
clip_grad_norm_(self.model.parameters(), max_norm=1.0)
optimizer.step()

# Proposed fix:
loss_dict = self.loss_fn(source_pred, sources, eeg_augmented, mask, epoch=self.current_epoch)
component_weights = {
    'loss_source': self.loss_fn.alpha,
    'loss_forward': beta_effective,  # from warm-up
    'loss_physics': self.loss_fn.gamma,
    'loss_epi': self.loss_fn.delta_epi,
}
accumulated_grads = {}
total_loss = torch.tensor(0.0, device=self.device)
for comp_name, weight in component_weights.items():
    if weight > 0:
        comp_loss = weight * loss_dict[comp_name]
        self.model.zero_grad()
        comp_loss.backward(retain_graph=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float('inf')
        )
        # Normalize and accumulate
        scale = weight / max(grad_norm, 1e-8)
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if name not in accumulated_grads:
                    accumulated_grads[name] = torch.zeros_like(p.data)
                accumulated_grads[name] += scale * p.grad.data.clone()
        total_loss += comp_loss.detach()
# Set accumulated gradient
self.model.zero_grad()
for name, p in self.model.named_parameters():
    if name in accumulated_grads:
        p.grad = accumulated_grads[name]
# Final clip of combined gradient
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
self.optimizer.step()
```

**Note**: The `retain_graph=True` in each backward call is necessary because the computational graph is shared. This adds ~4× backward pass cost per step, which increases training time from ~8s/epoch to ~30s/epoch. This is acceptable for a 410k-parameter model.

### Change 2: Set β=0.1 in config.yaml (per validated overfit test)

**File**: `config.yaml` line 129

While per-component normalization makes β=0.5 workable, β=0.1 is a more conservative starting point that gives more gradient budget to source reconstruction:

```yaml
beta_forward: 0.1
```

### Change 3: Reduce gradient_clip_norm from 1.0 to 5.0

**File**: `config.yaml` and `trainer.py`

With per-component normalization, the combined gradient is already well-balanced. A clip norm of 1.0 is very aggressive for a 410k-parameter model. Increasing to 5.0 allows larger updates while preventing occasional spikes:

```python
# Current:
gradient_clip_norm=1.0

# Proposed:
gradient_clip_norm=5.0
```

### Change 4: Increase learning rate warm-up (conditional on D2 results)

**File**: `src/phase2_network/trainer.py`

Add a 5-epoch linear LR warm-up from 1e-5 to 1e-3 to prevent early instability:

```python
# In _train_epoch, before optimizer.step():
if self.current_epoch < 5:
    lr_scale = (self.current_epoch + 1) / 5
    for pg in self.optimizer.param_groups:
        pg['lr'] = self.base_lr * lr_scale
```

### Change 5: Fix LR scheduler patience mismatch

**File**: `src/phase2_network/trainer.py` line 175

Change hardcoded `patience=10` to match config value of 5:

```python
self.scheduler = ReduceLROnPlateau(
    self.optimizer,
    mode='min',
    factor=0.5,
    patience=5,  # Match config.yaml value
    threshold=1e-4,
    min_lr=1e-6,
)
```

---

## 4. Execution Order

1. **Phase D1** — Gradient audit (15 min, CPU/GPU, no training)
2. **Change 1** — Implement per-component gradient normalization (~1 hour coding)
3. **Change 2** — Fix beta_forward to 0.1 in config.yaml (1 line)
4. **Change 3** — Increase gradient_clip_norm to 5.0 (1 line)
5. **Change 5** — Fix LR scheduler patience (1 line)
6. **Phase D2** — Run 5-epoch diagnostic training with β=0.1 + per-component normalization (30 min)
7. Evaluate: if L_src is decreasing and Corr > 0.05, proceed to full training
8. **Phase D4** — Full training run (80-200 epochs, 2-6 hours on RTX 3080)

**Change 4 (LR warm-up)** is conditional on Phase D2 results. Only add if the diagnostic run shows LR instability.

---

## Appendix A: Training Log Excerpt (Current Broken Behavior)

```
Epoch   1/80
  Train loss: 2.3224 | Val loss: 2.2931 | DLE: 33.81mm | SD: 68.34mm | AUC: 0.567 | Corr: 0.226 | L_src=1.972 L_fwd=0.998 L_epi=0.321
  ✓ Best model saved (val_loss: 2.2931)

Epoch   2/80
  Train loss: 2.5234 | Val loss: 2.5347 | DLE: 36.13mm | SD: 64.97mm | AUC: 0.566 | Corr: 0.011 | L_src=2.019 L_fwd=0.591 L_epi=0.456

...

Epoch  31/80
  Train loss: 2.4498 | Val loss: 2.4864 | DLE: 35.29mm | SD: 63.92mm | AUC: 0.567 | Corr: 0.003 | L_src=2.037 L_fwd=0.027 L_epi=0.436
  ⚠ No improvement for 30 epochs
Early stopping: 30 epochs without improvement
```

**Key observations**:
- Train loss INCREASES from epoch 1→2 (2.32→2.52) — optimization is destructive
- L_src INCREASES (1.97→2.04) — model gets worse at source reconstruction
- L_fwd DECREASES (0.998→0.027) — forward fit improves but at cost of everything else
- Corr COLLAPSES (0.226→0.003) — temporal dynamics lost
- AUC flat at ~0.57 — no epileptogenicity learning

---

## Appendix B: References

1. **Grech et al. (2008)** — Review on EEG source imaging, discusses MNE attractor and ill-posedness
2. **Hansen (1998)** — "Rank-Deficient and Discrete Ill-Posed Problems", SIAM
3. **Chen et al. (2018)** — "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML
4. **Sun et al. (2022)** — DeepSIF architecture (PNAS)
5. **Jirsa et al. (2014)** — Epileptor model specification
6. **Technical Specs §4.4.5** — Amplitude collapse diagnosis in this repo
