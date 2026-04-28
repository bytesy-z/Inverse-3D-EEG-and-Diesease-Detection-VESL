# ZIK Day 2 — Mon Apr 28: Implementation Plan (Data Generation + Retraining)

## Status After Audit (Sun Apr 27)

| Item | Status |
|------|--------|
| All Phase 1 imports | ✅ Verified working |
| All Phase 2 imports | ✅ Verified working |
| Model build (PhysDeepSIF) | ✅ 419k trainable params |
| `optuna` installed | ✅ v4.8.0 |
| `cmaes` installed | ✅ v0.12.0 |
| `data/synthetic3/test_dataset.h5` | ✅ 23 samples (will be regenerated with train/val) |
| Disk space | ✅ ~25 GB free |
| Lab GPU (RTX 3080) | ✅ Available |
| Laptop GPU (RTX 3050 Ti) | ⚠️ KWS training running until Apr 28 evening — not an issue since lab GPU is available |

---

## Phase 1: Pre-flight Validation (0.5h, before leaving for lab)

Run these locally to verify everything works before committing lab time.

### 1.1: Verify all imports

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
from src.phase1_forward.synthetic_dataset import generate_dataset, generate_all_splits
from src.phase1_forward.epileptor_simulator import run_simulation
from src.phase1_forward.parameter_sampler import sample_simulation_parameters
from src.phase2_network.physdeepsif import build_physdeepsif
from src.phase2_network.loss_functions import PhysicsInformedLoss
from src.phase2_network.trainer import PhysDeepSIFTrainer
print('All imports OK')
"
```

### 1.2: Small generation test (10 simulations locally)

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python scripts/02_generate_synthetic_data.py --split test --n-sims 10 --n-jobs 4
```

Expected output:
- Runs 10 TVB simulations in parallel (4 workers)
- Each generates ~5 windows, ~70% pass spatial-spectral validation
- Saves ~30-40 samples to `data/synthetic3/test_dataset.h5`
- If this fails, fix before going to lab

### 1.3: Verify training script imports

```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
from src.phase2_network.physdeepsif import build_physdeepsif
print('Training import OK')
"
```

---

## Phase 2: Lab Execution (RTX 3080, ~1-2h total)

### Step 2.1: Verify lab environment

```bash
# 1. Check GPU
nvidia-smi
# Confirm RTX 3080 visible, no other process using it

# 2. Verify conda env exists (or install full requirements)
conda activate physdeepsif

# 3. Verify repo is up to date
cd /path/to/fyp-2.0
git status
git pull origin main

# 4. Verify Python path
/path/to/lab/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2.2: Check for existing synthetic data on lab PC

```bash
ls -lh data/synthetic*/train_dataset.h5 data/synthetic*/val_dataset.h5 data/synthetic*/test_dataset.h5
```

**If found** (train >1 GB, val/test >100 MB): Copy to laptop and skip to Step 2.4.
```bash
# From laptop, after getting back:
scp lab:/path/to/data/synthetic3/*.h5 zik-laptop:~/UniStuff/FYP/fyp-2.0/data/synthetic3/
```

**If not found**, proceed to Step 2.3.

### Step 2.3: Generate full synthetic dataset on RTX 3080

```bash
/path/to/lab/python scripts/02_generate_synthetic_data.py --n-sims 23000 --n-jobs 16
```

What this does:
- Calls `generate_all_splits()` which generates train/val/test sequentially
- 23000 train sims + 2900 val + 2900 test = **28800 total sims**
- With 16 RTX 3080 cores × ~0.5s per sim ≈ **~15 min total**
- Expected output: ~80k train + 10k val + 10k test samples (after ~70% validation pass rate)
- Seed offsets prevent data leakage: train=0, val=100000, test=200000
- HDF5 incremental writing every 500 samples (fault tolerant)

**Monitor progress:**
```bash
# In another terminal, watch the log file:
tail -f outputs/generation.log
```

### Step 2.4: Start retraining with variance-normalized loss

Wait for data generation to finish first, then:

```bash
# Verify data exists
ls -lh data/synthetic3/train_dataset.h5 data/synthetic3/val_dataset.h5

# Launch training (RTX 3080)
/path/to/lab/python scripts/03_train_network.py --epochs 50 --batch-size 64 --device cuda
```

**Expected performance:**
- Model: 419k params (tiny — ~1.6 MB)
- Batch 64: ~30s per epoch on RTX 3080
- 50 epochs: **~25 min total**
- VRAM: ~60 MB (fits easily in 10 GB RTX 3080)

**Key training parameters (from config.yaml):**
- Loss: Variance-normalized forward (committed Apr 20)
- Preprocessing: de-mean → z-score (matches inference)
- Optimizer: AdamW, lr=0.001, weight_decay=0.0001
- LR schedule: CosineAnnealingWarmRestarts (T_0=10)
- Early stopping: patience=15 on val_loss

**Watch these metrics as training progresses:**
- `val_loss` — target < **1.0377** (epoch 24 baseline from old model)
- `DLE` — target < **20 mm**
- `AUC` — target > **0.85**
- `Corr` — target > **0.7**

### Step 2.5: Copy results to laptop

After training completes, transfer everything back:

```bash
# From lab machine:
scp data/synthetic3/train_dataset.h5       laptop:~/UniStuff/FYP/fyp-2.0/data/synthetic3/
scp data/synthetic3/val_dataset.h5         laptop:~/UniStuff/FYP/fyp-2.0/data/synthetic3/
scp data/synthetic3/test_dataset.h5        laptop:~/UniStuff/FYP/fyp-2.0/data/synthetic3/
scp outputs/models/checkpoint_best.pt      laptop:~/UniStuff/FYP/fyp-2.0/outputs/models/
scp outputs/models/normalization_stats.json laptop:~/UniStuff/FYP/fyp-2.0/outputs/models/
scp outputs/models/training.log            laptop:~/UniStuff/FYP/fyp-2.0/outputs/models/
```

---

## Phase 3: Post-Retraining Validation (on laptop, 1h evening)

### 3.1: Compare old vs new checkpoint

```python
# File: compare_checkpoints.py (run locally)
import torch

old = torch.load('outputs/models/checkpoint_best.pt', map_location='cpu')  # Backup old first!
# Back up old checkpoint before overwriting:
# cp outputs/models/checkpoint_best.pt outputs/models/checkpoint_best.pt.epoch24

new_checkpoint = torch.load('outputs/models/checkpoint_best.pt', map_location='cpu')

new_loss = new_checkpoint['val_loss']
old_loss = 1.0377  # epoch 24

print(f"Old val_loss: {old_loss:.4f}")
print(f"New val_loss: {new_loss:.4f}")

if new_loss < old_loss * 0.95:
    print("✅ New model significantly better — use new checkpoint")
elif new_loss > old_loss * 1.05:
    print("⚠️ New model worse — keep epoch 24, document in thesis")
else:
    print("✅ Marginal (within 5%) — use new model (variance-normalized loss)")
```

### 3.2: Verify normalization stats

```python
import json
import numpy as np

with open('outputs/models/normalization_stats.json') as f:
    stats = json.load(f)

print(f"eeg_mean = {stats['eeg_mean']:.6e}")
print(f"eeg_std  = {stats['eeg_std']:.4f}")
print(f"src_mean = {stats['src_mean']:.6e}")
print(f"src_std  = {stats['src_std']:.4f}")

# Sanity checks
assert abs(stats['eeg_mean']) < 1e-5, f"eeg_mean should be ~0 after de-meaning, got {stats['eeg_mean']}"
assert abs(stats['src_mean']) < 1e-5, f"src_mean should be ~0 after de-meaning, got {stats['src_mean']}"
assert stats['eeg_std'] > 0.1, f"eeg_std too small: {stats['eeg_std']}"
assert stats['src_std'] > 0.001, f"src_std too small (amplitude collapse?): {stats['src_std']}"

print("✅ Normalization stats look sane")
```

### 3.3: Run smoke tests

```bash
# Kill any existing servers first
./start.sh --kill

# Run checks
./start.sh --check

# Start backend
./start.sh --backend

# Test health endpoint
curl -sS http://127.0.0.1:8000/api/health

# Test inference on a sample
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" -F "sample_idx=0" -F "mode=source_localization" | python -m json.tool | head -20
```

### 3.4: Verify training log (check model convergence)

```bash
grep "Best model" outputs/models/training.log
grep "Val loss" outputs/models/training.log | tail -5
```

---

## Scientific Accuracy — Code Audit Results

All existing code verified correct during pre-plan audit:

| Aspect | File:Line | Status |
|--------|-----------|--------|
| De-meaning BEFORE z-score (training) | `scripts/03_train_network.py:192-213` | ✅ |
| De-meaning BEFORE z-score (inference) | `backend/server.py:301-306` | ✅ |
| Variance-normalized forward loss | `loss_functions.py:260-323` | ✅ |
| Denominator detached (gradient stop) | `loss_functions.py:318` | ✅ |
| Vectorized Laplacian (einsum, no loop) | `loss_functions.py:350-365` | ✅ |
| TVB dt = 0.1ms (≤0.1 for stability) | `epileptor_simulator.py:71` | ✅ |
| Noise only on fast subsystem (x1,y1,x2,y2) | `epileptor_simulator.py:403-409` | ✅ |
| Anti-aliased decimation (FIR multi-stage) | `epileptor_simulator.py:499-522` | ✅ |
| Epileptor param mapping (tau0→r, tau2→tau) | `epileptor_simulator.py:256-281` | ✅ |
| x0 ranges (Healthy [-2.2,-2.05], Epi [-1.8,-1.2]) | `parameter_sampler.py:46-47` | ✅ |
| Spectral shaping (STFT adaptive alpha) | `synthetic_dataset.py:289-440` | ✅ |
| Skull LP filter (40 Hz, 4th-order Butterworth) | `synthetic_dataset.py:112-120` | ✅ |
| Spatial-spectral validation (PDR, gradients) | `synthetic_dataset.py:683-779` | ✅ |
| Forward model (L @ S, shape verified) | `synthetic_dataset.py:443-484` | ✅ |
| Config loss weights (α=1.0, β=0.5, γ=0.1) | `config.yaml:117-120` → `loss_functions.py:159-163` | ✅ |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| TVB import error on lab machine | Install: `conda install -c conda-forge tvb-library` |
| `h5py` not found | `pip install h5py` |
| Generation too slow | Check `n_jobs` in config, try `--n-jobs 16` |
| Training OOM on RTX 3080 | Reduce `--batch-size 32` (config default is 64) |
| NaN in loss | Training script handles this via gradient clipping (max_norm=1.0) |
| Model not converging | Check normalization stats are sane, check de-meaning is applied |

---

## Success Checklist

- [ ] Phase 1.2: 10-sim test generation succeeds locally
- [ ] Phase 2.3: Synthetic data generated on lab RTX 3080
  - `train_dataset.h5` ≥50k samples
  - `val_dataset.h5` ≥5k samples
  - `test_dataset.h5` ≥5k samples
- [ ] Phase 2.4: Training launched and completes ≥10 epochs
- [ ] New `checkpoint_best.pt` saved
- [ ] New `normalization_stats.json` saved (eeg_mean ≈ 0, src_mean ≈ 0)
- [ ] Phase 2.5: All results copied back to laptop
- [ ] Phase 3.3: Smoke tests pass (`./start.sh --check` clean)
