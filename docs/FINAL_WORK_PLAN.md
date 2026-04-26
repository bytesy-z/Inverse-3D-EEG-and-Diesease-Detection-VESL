# PhysDeepSIF — Final Sprint Work Plan: April 27 – May 3, 2026

**Generated:** April 26, 2026  
**Deadline:** May 3, 2026 (Saturday)  
**Calendar days:** 7 | **Team:** Zik, Hira, Shahliza  
**Status:** AMBER — significant work remaining, but achievable with focused parallel execution

---

## 0. Verified Technical Context

This plan has been verified against the actual codebase on April 26, 2026. Key facts:

| Item | Verified Value | Implication |
|------|---------------|-------------|
| Model params | 419,004 trainable + 7,220 buffers | Tiny model — fits any GPU |
| Checkpoint | epoch 24, val_loss 1.0377 | Trained WITH de-meaning |
| Loss functions | Variance-normalized forward loss + vectorized Laplacian | Committed Apr 20 (commit eca8c3c) |
| Disk space | 14 GB free | Enough for data (~3 GB) + models |
| GPU (laptop) | RTX 3050 Ti, 3.68 GB VRAM | Model+bs64 ≈ 100 MB — fits easily |
| GPU (laptop) | Another training running until Apr 28 | Can't use laptop GPU until then |
| GPU (lab) | RTX 3080 + 16 cores | **Primary compute target** |
| Python env | `physdeepsif` conda env at `/home/zik/miniconda3/envs/physdeepsif` | Has tvb-library 2.10.0, torch 2.10.0+cu128, cmaes 0.12.0, MNE 1.11.0 |
| Missing packages | `optuna` (not installed), `cma` (use `cmaes` instead) | Install `optuna` in conda env; use `cmaes.CMA` API |
| Sample data | `data/samples/0001082.edf` (14 MB, 21 ch, 200 Hz, 30 min) + `1082.csv` (annotations) | **Abnormal recording** — has sharp waves in FP2/F4/F8 |
| EDF channels | 21 ch: FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T3,T4,T5,T6,FZ,PZ,CZ,A1,A2 | **Must map to 19 ch** (drop A1/A2, rename FP→Fp, FZ→Fz, etc.) |
| Synthetic data | NONE on this machine (no HDF5 files found) | **Must be generated or copied from lab** |
| `cmaes` API | `CMA.tell(list[tuple[array, float]])`, NOT `cma.fmin()` | Different from `cma` package — see verified example below |
| Frontend | Complete (EEG waveform, unified dashboard, Plotly brain) | Only backend integration needed |
| Backend | 4 endpoints: `/api/health`, `/api/analyze`, `/api/biomarkers`, `/api/eeg_waveform` | Need CMA-ES and NMT endpoints |

---

## 1. What IS Done (Verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| Phase 1: Forward modeling | ✅ Complete | `src/phase1_forward/` fully implemented |
| Phase 1: Spectral shaping | ✅ Complete | STFT-based in `synthetic_dataset.py` |
| Phase 1: Biophysical validation | ✅ Complete | 13/13 metrics PASS |
| Phase 2: PhysDeepSIF architecture | ✅ Complete | 419k params, SpatialMLP + BiLSTM |
| Phase 2: Loss functions | ✅ Complete + Fixed | Variance-normalized forward loss, vectorized Laplacian |
| Phase 2: DC offset de-meaning | ✅ Complete | In training script + backend inference |
| Phase 2: Training (epoch 24) | ✅ Complete | `checkpoint_best.pt`, val_loss=1.0377 |
| Web app v3 (full) | ✅ Complete | Backend 4 endpoints, frontend EEG waveform + brain viz |
| Loss fix (W1.5 + W8) | ✅ Committed | April 20 commit, not yet retrained |

## 2. What Remains (Critical Path)

| Task | Owner | Blocker | Est. Time |
|------|-------|---------|-----------|
| W1: Regenerate synthetic data | Zik (GPU) | Need lab GPU or wait until Apr 28 | 4-6h compute |
| W3: Retrain with fixed loss | Zik (GPU) | W1 must complete first | 2-4h compute |
| W5: Phase 3 NMT preprocessing | Shahliza | None — CPU only | 8-10h coding |
| W6: Phase 4 CMA-ES MVP | Hira | None — CPU only | 12-14h coding |
| W7: Phase 5 validation + baselines | Shahliza | Need retrained model (or use epoch 24) | 10-12h coding |
| Backend integration | Zik | W5 + W6 must be code-complete | 4-6h |
| Patient discrimination | Shahliza | Need Phase 3 + Phase 4 | 4h |
| End-to-end testing | Zik | Everything above | 4h |

---

## 3. Per-Person Task Breakdown

### 🟦 HIRA — CMA-ES Phase 4 (Branch: `hira/model-optimization`)

**Environment:** Uses `physdeepsif` conda env. Has `cmaes` 0.12.0 (NOT `cma`). CPU-only.

#### Day 1 — Sun Apr 27 (6-8h)

- [ ] Create branch `hira/model-optimization` from `main`
- [ ] Create `src/phase4_inversion/__init__.py`
- [ ] Implement `src/phase4_inversion/objective_function.py`:

```python
"""
CMA-ES objective function for fitting x0 (excitability) per patient.

J(x0) = w_source * J_source(x0) + w_eeg * J_eeg(x0) + w_reg * J_reg(x0)

Where:
  J_source: MSE between model-predicted source activity and PhysDeepSIF output
  J_eeg:    PSD-based error between simulated EEG and real EEG
  J_reg:    Sparsity regularizer ||x0 - x0_healthy||^2 (push toward healthy baseline)

The CMA-ES optimizer proposes candidate x0 vectors (76-dim), each requiring
a TVB simulation (4s sim = ~1s compute). Reduce sim to 4s to keep generation time
manageable: pop=20 × 4s sim = ~20s per generation, 50 generations ≈ 15 min total.
"""
import numpy as np
from numpy.typing import NDArray
from ..phase1_forward.epileptor_simulator import run_simulation, segment_source_activity
from ..phase1_forward.parameter_sampler import sample_simulation_parameters
from ..phase1_forward.synthetic_dataset import project_to_eeg, apply_skull_attenuation_filter, apply_spectral_shaping, add_measurement_noise

def objective(x0: NDArray, patient_eeg_psd, leadfield, connectivity,
              region_centers, region_labels, tract_lengths, config,
              w_source=0.4, w_eeg=0.4, w_reg=0.2) -> float:
    ...
```

**Key implementation detail:** Each CMA-ES evaluation requires running a TVB simulation with a proposed x0 vector. The simulator takes 76-dim x0 where:
- Healthy regions: x0 ∈ [-2.2, -2.05] (from `config.yaml`)
- Epileptogenic regions: x0 ∈ [-1.8, -1.2]
- The objective compares simulated EEG PSD with patient EEG PSD

- [ ] Implement `src/phase4_inversion/cmaes_optimizer.py`:

**CRITICAL: Use `cmaes.CMA` API, NOT `cma.CMAEvolutionStrategy`!**

The `cmaes` package (installed v0.12.0) uses a different API than the `cma` package:

```python
from cmaes import CMA
import numpy as np

def fit_patient(patient_inference, leadfield, connectivity, config):
    """CMA-ES optimization for patient-specific epileptogenicity mapping."""
    x0_init = np.full(76, config['parameter_inversion']['initial_x0'])  # -2.1
    sigma0 = config['parameter_inversion']['initial_sigma']              # 0.3
    bounds = (np.full(76, config['parameter_inversion']['bounds'][0]),  # -2.4
              np.full(76, config['parameter_inversion']['bounds'][1]))   # -1.0
    
    cma = CMA(mean=x0_init, sigma=sigma0, bounds=bounds, seed=42)
    
    # MVP settings: smaller population for speed
    # pop_size = cma.population_size  # automatically ~14 for dim=76
    max_generations = config['parameter_inversion'].get('max_generations', 50)
    
    convergence_history = []
    
    for gen in range(max_generations):
        solutions = []
        while len(solutions) < cma.population_size:
            x = cma.ask()
            score = objective(x, ...)
            solutions.append((x, score))
        
        cma.tell(solutions)  # NOTE: list of (x, score) tuples
        convergence_history.append(min(s for _, s in solutions))
        
        if cma.should_stop():
            break
    
    best_x0 = min(solutions, key=lambda s: s[1])[0]
    ...
```

**Verified gotcha:** `CMA.tell()` takes a `list[tuple[array, float]]`, NOT separate `(solutions, values)`. This is different from the `cma` package's `cma.fmin()` API.

#### Day 2 — Mon Apr 28 (6-8h)

- [ ] Implement `src/phase4_inversion/epileptogenicity_index.py`:

```python
def compute_ei(x0_fitted: NDArray) -> NDArray:
    """
    Map fitted x0 to epileptogenicity index EI ∈ [0, 1].
    
    Uses sigmoid mapping centered at healthy baseline (-2.2):
      EI_i = sigmoid((x0_i + 2.2) / 0.15)
    
    Interpretation:
      x0 = -2.2 (healthy baseline) → EI ≈ 0.5
      x0 = -1.8 (mildly epileptogenic) → EI ≈ 0.92
      x0 = -1.2 (highly epileptogenic) → EI ≈ 1.0
      x0 = -2.4 (strongly suppressed) → EI ≈ 0.13
    
    The scale parameter (0.15) controls the transition sharpness:
      - Smaller = sharper transition (more binary classification)
      - Larger = smoother (more continuous gradient)
    
    Note: This replaces the heuristic inverted-range EI in the backend.
    The heuristic (compute_epileptogenicity_index in server.py) uses peak-to-peak
    amplitude inversion. CMA-ES EI uses the physics-based x0 parameter directly.
    """
    x0_baseline = -2.2  # Healthy threshold from Epileptor literature
    scale = 0.15        # Transition sharpness
    ei = 1.0 / (1.0 + np.exp(-(x0_fitted - x0_baseline) / scale))
    return ei
```

- [ ] Unit-test CMA-ES on 1 synthetic test patient:
  - Use existing `checkpoint_best.pt` to get PhysDeepSIF predictions as "patient" input
  - Run CMA-ES with pop=14 (automatic for 76-dim), gen=20 (quick test)
  - Verify: x0 converges toward ground truth epileptogenic regions

#### Day 3 — Tue Apr 29 (6-8h)

- [ ] CMA-ES integration testing on 2-3 synthetic patients
- [ ] Measure actual per-evaluation time: each TVB sim ≈ 1s on CPU
  - pop=20 × 1s = 20s per generation
  - 50 generations × 20s = ~17 min total per patient
  - This is acceptable for a thesis demo
- [ ] If too slow: reduce sim to 2s (from 4s) by changing `SIMULATION_LENGTH_MS` to ~4000ms in `objective_function.py`
- [ ] Create `scripts/08_run_cmaes.py` — standalone driver script

#### Day 4 — Wed Apr 30 (4-6h)

- [ ] If retrained model available from Zik: re-test CMA-ES with new checkpoint
- [ ] Document CMA-ES convergence behavior (x0 trajectories, EI accuracy)
- [ ] Help Zik with backend CMA-ES endpoint design

#### Day 5 — Thu May 1 (4-6h)

- [ ] CMA-ES on 1 NMT patient (once Shahliza provides preprocessed data)
- [ ] Write tech specs §6.3 with actual implementation details
- [ ] Generate CMA-ES convergence plot for thesis

#### Day 6-7 — Fri-Sat May 2-3

- [ ] Bug fixes, polishing, thesis contribution text

---

### 🟠 SHAHLIZA — Phase 3 NMT + Phase 5 Validation (Branch: `shahliza/validation-pipeline`)

**Environment:** CPU-only (MNE, scipy). All packages already installed in `physdeepsif` env.

#### Day 1 — Sun Apr 27 (6-8h)

- [ ] Create branch `shahliza/validation-pipeline` from `main`
- [ ] Create `src/phase3_inference/__init__.py`
- [ ] Implement `src/phase3_inference/nmt_preprocessor.py`:

```python
"""
NMT (NeuroMorphic Therapy) EEG preprocessing pipeline.
Implements the 6-step pipeline per Tech Specs §5.2.

Steps:
  1. Load EDF via MNE, verify 19 channels present
  2. Ensure linked-ear reference (EEG standard for 10-20 montage)
  3. Bandpass filter 0.5-70 Hz (Butterworth, 4th order)
  4. Notch filter 50 Hz (EEG line noise removal)
  5. ICA artifact removal (fastica, n_components=15)
  6. Segment into 2s windows (400 samples @ 200 Hz) and z-score normalize

Input: Path to EDF file
Output: Processed EEG windows (n_epochs, 19, 400) + channel names + epoch times
"""
import mne

NMTPreprocessor:
    def load_edf(edf_path: str) -> mne.io.Raw:
        # Load with MNE, verify 19 channels
        # Expected channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
        #                    F7, F8, T3, T4, T5, T6, Fz, Cz, Pz
    
    def verify_channels(raw: mne.io.Raw) -> mne.io.Raw:
        # If channel names use 'EEG XXX' format, strip prefix
        # If missing channels, raise error
        # If extra channels, pick only the 19 standard ones
    
    def apply_linked_ear_reference(raw: mne.io.Raw) -> mne.io.Raw:
        # MNE set_eeg_reference('average') for linked-ear equivalent
        # Note: NMT data is already linked-ear ref per tech specs
    
    def bandpass_filter(raw: mne.io.Raw) -> mne.io.Raw:
        # raw.filter(l_freq=0.5, h_freq=70.0, method='fir', fir_design='firwin')
    
    def notch_filter(raw: mne.io.Raw) -> mne.io.Raw:
        # raw.notch_filter(freqs=50.0)  # 50 Hz line noise
    
    def remove_ica_artifacts(raw: mne.io.Raw) -> mne.io.Raw:
        # Fit ICA with n_components=15
        # Auto-detect EOG/ECG components
        # Remove artifact components
    
    def segment_and_normalize(raw: mne.io.Raw) -> dict:
        # Create 2s epochs (400 samples @ 200 Hz)
        # epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None)
        # Z-score normalize per channel using training stats from normalization_stats.json
        # Return: preprocessed_eeg (n_epochs, 19, 400), channel_names, epoch_times
```

**Key gotcha for MNE EDF loading:**
- NMT EDF files may have channel names like `'EEG Fp1-Ref'` or `'EEG 001'`
- Must map to standard 10-20 names using `mne.rename_channels()`
- The test file `data/samples/0001082.edf` (14 MB) is the first test case
- Read `data/samples/1082.csv` for channel mapping information

- [ ] Test on `data/samples/0001082.edf`:
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import mne
raw = mne.io.read_raw_edf('data/samples/0001082.edf', preload=True)
print('Channels:', raw.ch_names)
print('N channels:', len(raw.ch_names))
print('Sampling rate:', raw.info['sfreq'])
print('Duration:', raw.times[-1], 's')
"
```
Run this first to understand the channel naming convention before implementing the preprocessor.

#### Day 2 — Mon Apr 28 (6-8h)

- [ ] Implement `src/phase3_inference/inference_engine.py`:

```python
def run_patient_inference(
    edf_path: str,
    model,               # PhysDeepSIF model
    norm_stats: dict,     # From normalization_stats.json
    device: str = 'cpu'
) -> dict:
    """
    Full patient inference pipeline:
      1. Preprocess EDF via NMTPreprocessor
      2. Run PhysDeepSIF on each epoch
      3. Aggregate results per-region
    
    Returns:
        {
            "source_estimates": ndarray (n_epochs, 76, 400),
            "mean_source_power": ndarray (76,),
            "activation_consistency": ndarray (76,),
            "preprocessed_eeg": ndarray (n_epochs, 19, 400),
            "n_epochs": int,
            "epoch_times": list[float],
            "channel_names": list[str],
            "edf_path": str,
        }
    """
```

**Important:** The inference must apply the SAME preprocessing as training:
1. Per-channel temporal de-meaning (`eeg_window - eeg_window.mean(axis=-1, keepdims=True)`)
2. Global z-score using training stats (`(eeg - eeg_mean) / (eeg_std + eps)`)

This is already implemented in `backend/server.py:run_inference()`. The new `inference_engine.py` should reuse the same normalization logic but support batch processing.

- [ ] Implement `src/phase3_inference/source_aggregator.py`:

```python
def aggregate_sources(source_estimates, region_labels_path='data/region_labels_76.json'):
    """
    Aggregate window-level source estimates into patient-level profile.
    
    For each region i:
      - mean_power[i] = mean over (epochs, time) of |source[i, t]|
      - variance[i] = var over (epochs, time) of source[i, t]
      - peak_to_peak[i] = max(source[i, :]) - min(source[i, :])
      - activation_consistency[i] = fraction of epochs where region is in top-10%
    
    Returns:
        dict with per-region metrics and overall statistics
    """
```

#### Day 3 — Tue Apr 29 (6-8h)

- [ ] Create `src/phase5_validation/__init__.py`
- [ ] Implement `src/phase5_validation/synthetic_metrics.py`:

```python
def compute_dle(predicted_centers, true_centers):
    """Distance Localization Error (mm) — mean Euclidean distance between
    predicted and true epileptogenic region centers."""

def compute_sd(predicted_sources, region_centers, top_k=5):
    """Spatial Dispersion (mm) — std of distances of top-k regions from centroid."""

def compute_auc(true_mask, predicted_scores):
    """Area Under ROC Curve — discriminating epileptogenic from healthy regions."""

def compute_temporal_correlation(predicted, true):
    """Mean Pearson correlation across regions between predicted and true time courses."""

def compute_all_metrics(predicted, true, true_mask, region_centers):
    """Run all 4 metrics and return dict."""
```

**Existing code reference:** `src/phase2_network/metrics.py` already has DLE, SD, AUC, and temporal correlation implementations. Check those first and reuse/extend.

- [ ] Implement `src/phase5_validation/classical_baselines.py`:

```python
"""
Classical EEG source localization baselines for comparison with PhysDeepSIF.

Implements 4 standard methods using MNE-Python:
  1. eLORETA — exact low-resolution brain electromagnetic tomography
  2. MNE — minimum norm estimate (dSPM variant)
  3. dSPM — dynamic statistical parametric mapping
  4. LCMV — linearly constrained minimum variance beamformer

Each baseline maps 19-channel EEG (19, 400) → 76-region source activity (76, 400)
using the same 19×76 leadfield matrix that PhysDeepSIF uses.

Implementation notes:
  - MNE baselines work with MNE's Forward model, NOT with our custom leadfield
  - Must create a minimal MNE Forward object from our leadfield + region centers
  - Use mne.make_forward_solution() with custom source space (76 discrete sources)
  - Inverse methods then operate on this forward model
  - Expected: these baselines should perform WORSE than PhysDeepSIF on our metrics,
    validating that the deep learning approach adds value beyond classical methods
  
Key gotcha: MNE expects specific data structures (Info, Forward, Inverse).
The 76 discrete sources must be registered as a DiscreteSourceSpace.
Use mne.SourceEstimate for output alignment.
"""
```

**This is harder than it sounds.** Creating a valid MNE Forward model from our 19×76 leadfield requires:
1. An MNE `Info` object with our 19 channels and 200 Hz sampling rate
2. A `SourceSpaces` object with 76 discrete sources at our region_centers
3. A `Forward` object constructed via `mne.make_forward_solution()`
4. Each inverse method then creates an `InverseOperator`

**Simpler alternative if MNE integration proves too complex:**
- Implement eLORETA manually: it's just a weighted minimum-norm inverse
- The eLORETA inverse for our setup reduces to: `s = (L^T L + λI)^-1 L^T eeg`
- MNE/dSPM/LCMV have similar closed-form expressions
- This avoids MNE's complex API and works directly with numpy

#### Day 4 — Wed Apr 30 (6-8h)

- [ ] Run baselines on test subset
- [ ] Generate comparison table: PhysDeepSIF vs eLORETA vs MNE vs dSPM vs LCMV
- [ ] Start patient validation (`src/phase5_validation/patient_validation.py`):

```python
def validate_patient(
    edf_paths: list[str],
    model,
    norm_stats: dict,
    leadfield: ndarray,
    connectivity: ndarray,
    config: dict,
    device: str = 'cpu'
) -> dict:
    """
    Patient-level validation using NMT recordings.
    
    For each patient:
      1. Preprocess EDF via NMTPreprocessor
      2. Run PhysDeepSIF inference
      3. Compute EI per region (using CMA-ES or heuristic)
      4. Report: max EI, top-3 regions, spatial distribution
    
    Discrimination metric:
      - Compare max(EI) across patients
      - Abnormal patients should have higher max EI and clearer focal patterns
      - Normal patients should have uniform EI ≈ 0.5
    """
```

#### Day 5-7 — Thu-Sat May 1-3

- [ ] Complete patient validation (at least 2 NMT recordings)
- [ ] If only 1 recording type available: intra-patient consistency analysis
  - Split recording into segments (first half vs second half)
  - Compute EI per segment
  - Check: do the same regions come up consistently across segments?
  - This is a valid validation approach (test-retest reliability)
- [ ] Generate ALL thesis tables and figures
- [ ] Create `scripts/06_run_validation.py` and `scripts/07_run_baselines.py`
- [ ] Help Zik with merge and integration testing

---

### 🟦 ZIK — GPU Operations, Backend Integration, Orchestration

#### Day 1 — Sun Apr 27 (6-8h)

- [ ] Install missing packages in conda env:
  ```bash
  /home/zik/miniconda3/envs/physdeepsif/bin/pip install optuna
  ```
- [ ] Verify `scripts/02_generate_synthetic_data.py` runs without import errors:
  ```bash
  /home/zik/miniconda3/envs/physdeepsif/bin/python scripts/02_generate_synthetic_data.py --n-sims 10 --split test
  ```
  If this fails, fix import errors before going to the lab.
- [ ] Verify `scripts/03_train_network.py` runs without import errors:
  ```bash
  /home/zik/miniconda3/envs/physdeepsif/bin/python -c "from src.phase2_network.physdeepsif import build_physdeepsif; print('OK')"
  ```
- [ ] Create skeleton scripts:
  - `scripts/08_run_cmaes.py` (calls Hira's CMA-ES)
  - `scripts/06_run_validation.py` (calls Shahliza's validation)
- [ ] Review and understand the CMA-ES `cmaes` API (verified working):
  ```python
  from cmaes import CMA
  cma = CMA(mean=np.zeros(76), sigma=0.3, bounds=(lo, hi))
  # ask() returns one candidate; call in loop to get population
  # tell() takes list of (x_array, fitness_float) tuples
  ```

#### Day 2 — Mon Apr 28: LAB DAY (Critical Path)

- [ ] Go to lab. Verify RTX 3080 is operational.
- [ ] **If synthetic data exists on lab PC**: copy `data/synthetic3/` or `data/synthetic4/` (train.h5, val.h5, test.h5) to this machine.
- [ ] **If no synthetic data on lab PC**: generate on RTX 3080:
  ```bash
  /path/to/lab/python scripts/02_generate_synthetic_data.py --n-sims 5000 --n-jobs 16
  ```
  With RTX 3080 + 16 cores: 5000 sims × ~0.5s each ≈ 25 min (parallel)
  ~5 windows per sim × 70% pass rate ≈ 17,500 train samples
  
- [ ] **Start retraining** immediately if data available:
  ```bash
  /path/to/lab/python scripts/03_train_network.py --epochs 50 --batch-size 64
  ```
  RTX 3080: 410k param model, batch_size=64 → ~30s per epoch
  50 epochs with early stopping ≈ 25 min total
  
- [ ] **If lab GPU not available**: Generate data on this machine starting Apr 28 evening:
  ```bash
  /home/zik/miniconda3/envs/physdeepsif/bin/python scripts/02_generate_synthetic_data.py --n-sims 3000 --n-jobs 12
  ```
  3000 sims × ~1-2s each × 12 parallel ≈ 4-6h total
  ~5 windows × 70% pass ≈ 10,500 train samples
  
  Start training after kws job finishes (Apr 28 evening):
  ```bash
  /home/zik/miniconda3/envs/physdeepsif/bin/python scripts/03_train_network.py --epochs 50 --batch-size 32
  ```
  RTX 3050 Ti: batch_size=32 (smaller to fit VRAM), ~60s per epoch
  50 epochs with early stopping ≈ 50 min total

**Estimated VRAM usage:** PhysDeepSIF is only 419k params (~1.6 MB). Even with batch_size=64:
- Model: 1.6 MB
- Optimizer: 3.2 MB  
- Batch data: 3.9 MB
- Activations: ~50 MB
- **Total: ~60 MB** — fits easily on both GPUs

**Disk space concern:** 14 GB free. Synthetic data ≈ 2-3 GB. Model checkpoints < 10 MB. Should be fine.

#### Day 3 — Tue Apr 29

- [ ] Training should be done. Evaluate results:
  ```python
  # Compare old vs new checkpoint
  old_loss = 1.0377  # epoch 24
  
  # If new checkpoint val_loss < old_loss AND metrics improve:
  #   → Use new checkpoint
  # If new checkpoint val_loss > old_loss:
  #   → Keep epoch 24 checkpoint, document in thesis that retraining
  #      was attempted and the existing model performed better
  # If results are marginal (within 5%):
  #   → Go with the new model (it has variance-normalized loss)
  ```
  
- [ ] Pull Hira's CMA-ES branch and Shahliza's Phase 3/5 branches
- [ ] Merge all branches into `main`:
  ```
  git checkout main
  git merge hira/model-optimization
  git merge shahliza/validation-pipeline
  # Resolve conflicts (most likely in config.yaml, backend/server.py)
  ```
- [ ] Run Shahliza's Phase 3 on `data/samples/0001082.edf`
- [ ] Run Hira's CMA-ES on 1 synthetic sample

#### Day 4 — Wed Apr 30: Backend Integration (Full Day)

- [ ] Add NMT preprocessing to backend:
  - Option A: Modify `/api/analyze` to use NMT preprocessor when EDF is uploaded
  - Option B: Add separate `POST /api/preprocess-nmt` endpoint
  - Recommended: Option A (simpler for frontend)
  
  Changes needed in `backend/server.py`:
  ```python
  # Add import at top
  from src.phase3_inference.nmt_preprocessor import NMTPreprocessor
  from src.phase3_inference.inference_engine import run_patient_inference
  
  # In run_inference() or analyze_handler():
  # After loading EDF, use NMTPreprocessor instead of simple segmentation
  # NMTPreprocessor handles: verify channels, filter, ICA, segment, z-score
  ```

- [ ] Add CMA-ES biomarker endpoint:
  ```python
  # Add new endpoint
  @app.post("/api/biomarkers-cmaes")
  async def analyze_biomarkers_cmaes(file: UploadFile = File(...)):
      """
      Full CMA-ES epileptogenicity analysis.
      
      Pipeline: EDF → NMT preprocess → PhysDeepSIF → CMA-ES → heatmap
      
      Timeout: 5 minutes. If CMA-ES doesn't converge, fall back to heuristic EI.
      """
      # 1. Preprocess EDF via NMTPreprocessor
      # 2. Run PhysDeepSIF inference
      # 3. Run CMA-ES with patient_inference dict
      # 4. Compute epileptogenicity_index from fitted x0
      # 5. Generate Plotly heatmap
      # 6. Return JSON with eegData, plotHtml, scores, cmaes_info
  ```

- [ ] Add CMA-ES timeout handling:
  ```python
  import asyncio
  
  async def run_cmaes_with_timeout(patient_data, timeout_seconds=300):
      try:
          result = await asyncio.wait_for(
              asyncio.to_thread(fit_patient, patient_data),
              timeout=timeout_seconds
          )
          return result
      except asyncio.TimeoutError:
          # Fall back to heuristic EI
          return compute_heuristic_ei(patient_data)
  ```

#### Day 5 — Thu May 1: End-to-End Testing

- [ ] Test full pipeline: EDF upload → preprocess → PhysDeepSIF → CMA-ES → heatmap
- [ ] Test both modes: source localization + biomarkers (CMA-ES)
- [ ] Test with `data/samples/0001082.edf` via curl:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/biomarkers-cmaes \
    -F "file=@data/samples/0001082.edf" \
    -F "mode=biomarkers"
  ```
- [ ] Run Shahliza's full validation suite
- [ ] Fix integration bugs (there will be some)

#### Day 6-7 — Fri-Sat May 2-3: Polish + Submit

- [ ] Merge final fixes
- [ ] Run `./start.sh --check`
- [ ] Run `scripts/smoke_test.sh`
- [ ] Generate thesis figures (see below)
- [ ] `git tag v2.0-submission`

---

## 4. Known Issues and Mitigations

### Issue 1: `cmaes` API differs from `cma` package

**Risk:** The installed package is `cmaes` v0.12.0 (not `cma`). The API is fundamentally different.

| Feature | `cma` package | `cmaes` package (INSTALLED) |
|---------|-------------|---------------------------|
| Entry point | `cma.fmin(objective, x0, sigma0)` | `CMA(mean, sigma, bounds)` class |
| Evaluation | Automatic loop | Manual `ask()` + `tell()` loop |
| `tell()` signature | `tell(xs, values)` separate arrays | `tell(list[tuple[array, float]])` |
| Bounds | Via `bounds` option dict | Via `bounds` constructor arg |
| Best solution | `result.xbest` | `cma.best_x` property |

**Mitigation:** Use the verified `cmaes` API pattern from the code snippet above. Do NOT try to use `cma.fmin()`.

### Issue 2: No synthetic data on this machine

**Risk:** `data/synthetic3/` doesn't exist locally. Data generation requires TVB simulator (installed in conda env).

**Mitigation two paths:**
1. **Lab PC has the data:** Copy it on Monday. Truest path.
2. **Lab PC doesn't have data:** Generate locally. Use `scripts/02_generate_synthetic_data.py` with `--n-sims 3000`. Allow 4-6h.

### Issue 3: MNE Forward model construction is non-trivial

**Risk:** Creating a valid MNE Forward model from our 19×76 leadfield + 76 region centers requires careful setup of SourceSpaces, Info, and transformation matrices. MNE expects specific formats.

**Mitigation:** If `mne.make_forward_solution()` proves too complex, implement baselines analytically:
- **eLORETA:** `s = (L^T D^{-1}) eeg` where `D = (L^T L + λI)` — reduced to matrix operations
- **dSPM:** `s = Σ^{1/2} L^T (L Σ L^T + λ C_n)^{-1} eeg` — noise covariance estimation needed
- **Minimum norm:** `s = (L^T L + λI)^{-1} L^T eeg`
- **LCMV:** `w = (R^{-1} L) / (L^T R^{-1} L)` where R is data covariance

### Issue 4: CMA-ES performance on 76-dimensional problem

**Risk:** CMA-ES with 76 parameters (one x0 per region) may need many evaluations to converge. Population for 76-dim ≈ 14-20.

**Mitigation:**
- Start with CMA default population size (~14 for 76-dim)
- Reduce TVB simulation to 4s (from 12s) in CMA-ES objective
- Use bounding box [-2.4, -1.0] per region (from config.yaml `parameter_inversion.bounds`)
- Set max_generations=50 for MVP (config.yaml says 200, but 50 is sufficient for thesis)
- Expected runtime: 50 gens × 14 pop × 1s sim = ~12 min per patient

### Issue 5: RTX 3050 Ti VRAM is only 3.68 GB

**Risk:** GPU shared with another process (kws training until Apr 28).

**Mitigation:**
- PhysDeepSIF is tiny (419k params, ~1.6 MB state)
- Training at batch_size=32 uses ~30-40 MB VRAM — no issue
- Data generation is CPU-only (TVB uses CPU, not GPU)
- The lab RTX 3080 (16 GB) is the primary training target

### Issue 6: ICA artifact removal may fail on some NMT recordings

**Risk:** ICA with n_components=15 may not converge or may not find 15 components if the recording is short.

**Mitigation in NMTPreprocessor:**
```python
try:
    ica = mne.preprocessing.ICA(n_components=15, method='fastica', max_iter=1000)
    ica.fit(raw)
    # Auto-detect artifact components
    ica.exclude = find_eog_ecg_artifacts(raw, ica)
    raw = ica.apply(raw)
except Exception as e:
    logger.warning(f"ICA failed: {e}. Skipping ICA for this recording.")
    # Continue without ICA — the network was trained on non-ICA data anyway
```

### Issue 7: Disk space (14 GB free)

**Risk:** Synthetic data generation creates large HDF5 files.

**Mitigation:**
- Train HDF5: ~1.5 GB, Val: ~0.2 GB, Test: ~0.2 GB
- Keep only necessary splits (can delete val after training)
- Delete `data/synthetic3/` temporary files after training completes
- Model checkpoints: < 10 MB each

### Issue 8: EDF channel name mismatches (VERIFIED)

**Risk:** NMT EDF files use different naming conventions than our pipeline expects.

**Verified:** `data/samples/0001082.edf` has 21 channels with these names:
```
['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'PZ', 'CZ', 'A1', 'A2']
```

**Required mapping:** Our pipeline uses 19 channels with these names:
```
['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
```

**Channel transformations needed:**
1. `FP1` → `Fp1`, `FP2` → `Fp2` (case fix)
2. `FZ` → `Fz`, `CZ` → `Cz`, `PZ` → `Pz` (case fix)
3. Drop `A1` and `A2` (ear reference channels, not in 10-20 montage)
4. Keep `T3`/`T4` as-is (NOT renamed to T7/T8 — uses older 10-20 naming)

**Bonus:** `1082.csv` is an annotations file marking epileptiform events (sharp waves, spike-and-wave) at channels FP2, F4, F8. This confirms the recording is from an **abnormal** patient with right frontal/temporal epileptiform activity — ideal for validating that our EI correctly identifies these regions.

**Mitigation in NMTPreprocessor:**
```python
STANDARD_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                     'Fz', 'Cz', 'Pz']

EDF_CHANNEL_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
    # T3, T4, T5, T6 remain the same
    # A1, A2 are dropped (not in standard 10-20)
}

def normalize_channel_names(raw):
    mapping = {}
    for ch in raw.ch_names:
        clean = ch.strip()
        if clean in EDF_CHANNEL_MAP:
            mapping[ch] = EDF_CHANNEL_MAP[clean]
        # else: keep name as-is for standard channels
    
    # Drop reference channels
    channels_to_drop = [ch for ch in raw.ch_names 
                       if ch not in STANDARD_CHANNELS 
                       and EDF_CHANNEL_MAP.get(ch, ch) not in STANDARD_CHANNELS]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    
    raw.rename_channels(mapping)
    raw.pick(STANDARD_CHANNELS)  # Select only our 19 channels
```

### Issue 9: Training data mismatch after variance-normalized loss fix

**Risk:** The variance-normalized forward loss (committed Apr 20) changes the loss landscape. We're now retraining on the same data but with a different loss function. The old normalization_stats.json was computed WITHOUT variance normalization.

**Mitigation:** The normalization stats (eeg_mean, eeg_std, src_mean, src_std) are computed from the de-meaned training data, not from the loss function. They remain valid regardless of loss changes. Only the loss computation and gradient updates change. **No need to regenerate data or recompute normalization stats.**

### Issue 10: `data/samples/1082.csv` is an epileptiform event annotation file (VERIFIED)

**This is actually great news for validation!** The CSV contains timestamps of sharp waves, spike-and-wave complexes, and other epileptiform events at specific channels. Summary:
- **Patient type:** Abnormal (15M, has frequent epileptiform activity)
- **Epileptiform channels:** Primarily FP2, F4, F8 (right frontal/temporal)
- **Event types:** Sharp wave, spike wave, sharp and slow wave, spike and wave
- **Expected EI:** Our CMA-ES should identify right frontal/temporal regions (rFP, rF8, rAMYG, rHIP) as highly epileptogenic (EI > 0.7)

**Use in validation:**
- This provides ground truth for patient discrimination: abnormal recording with known epileptogenic zone
- Compare our predicted EI hotspots against the annotated channels
- If we also have a normal recording, we can show clear EI separation: abnormal max(EI) >> normal max(EI)

**Before implementing NMTPreprocessor:**
```bash
/home/zik/miniconda3/envs/physdeepsif/bin/python -c "
import mne
raw = mne.io.read_raw_edf('data/samples/0001082.edf', preload=True)
print('Channels:', raw.ch_names)
print('N channels:', len(raw.ch_names))
print('Sampling rate:', raw.info['sfreq'])
print('Duration:', raw.times[-1], 's')
"
```
Already verified: 21 channels, 200 Hz, ~1806s (30 min).

---

## 5. Expected Results (Minimum Viable Thesis)

By May 3, we should have:

### Must-Have (Critical Path)
- [ ] Model retrained with variance-normalized loss (or documented why epoch 24 suffices)
- [ ] Synthetic data regenerated (or restored from lab PC)
- [ ] Phase 3 NMT preprocessing working on at least 1 EDF recording
- [ ] Phase 4 CMA-ES producing plausible EI on at least 1 synthetic patient
- [ ] Backend serving CMA-ES biomarker results
- [ ] End-to-end EDF upload → heatmap demonstrated in browser
- [ ] Phase 5 validation metrics (DLE, AUC, temporal correlation) on synthetic test set

### Should-Have (Strong Value Add)
- [ ] Classical baselines (eLORETA, MNE, dSPM) computed
- [ ] Patient validation showing normal vs. abnormal discrimination
- [ ] CMA-ES convergence plots and EI heatmaps
- [ ] Training loss curves comparing old vs. new loss function

### Nice-to-Have (If Time Permits)
- [ ] Optuna hyperparameter search (design only, not execution)
- [ ] Multiple NMT patient recordings analyzed
- [ ] Variance-matching loss term implemented
- [ ] Updated frontend README

---

## 6. Thesis Figures Checklist

Figures to generate by end of sprint:

1. **Architecture diagram** — PhysDeepSIF Spatial + Temporal modules
2. **Training loss curves** — Source loss, forward loss, physics loss, total (epoch 24, or new)
3. **DLE heatmap** — Per-region localization error on test set
4. **AUC per noise level** — 5, 10, 15, 20, 30 dB SNR
5. **Baselines comparison table** — PhysDeepSIF vs. eLORETA vs. MNE vs. dSPM vs. LCMV
6. **CMA-ES convergence plot** — Objective function J over generations
7. **Patient EI profile** — Bar chart of epileptogenicity index per region
8. **Brain heatmap screenshots** — Source localization (inferno) + biomarker (warm gradient)
9. **EEG waveform + brain synchronization** — Side-by-side screenshot
10. **Normal vs. abnormal EI comparison** — If 2+ NMT recordings available

---

## 7. Git Branch Strategy (Revised)

```
main (production)
  ├─ hira/model-optimization      → CMA-ES + training config
  ├─ shahliza/validation-pipeline → Phase 3 + Phase 5
  └─ zik/integration              → backend + frontend (or work on main)
```

**Merge schedule:**
- **Tue Apr 29**: Hira merges CMA-ES. Shahliza merges Phase 3. Zik resolves conflicts.
- **Thu May 1**: Shahliza merges Phase 5. Zik merges backend integration.
- **Fri May 2**: CODE FREEZE. Bug fixes only.
- **Sat May 3**: Final tag + submit.

**Conflict avoidance rules:**
- `backend/server.py`: Zik adds CMA-ES endpoint ONLY (no modifications to existing routes)
- `config.yaml`: Each person adds separate sections (preprocessing, cmaes, validation)
- No one touches another person's `src/phase*_/` directories

---

## 8. Cut Items (and Thesis Framing)

| Cut Item | Thesis framing |
|----------|---------------|
| Optuna hyperparameter search (W2) | "Search space defined in config.yaml §training; TPE optimization designed but not executed due to compute constraints during sprint" |
| Full CMA-ES (pop=50, gen=200) | "MVP implemented with pop≈14, gen=50; demonstrates feasibility on synthetic patient; full-scale optimization gated on GPU server availability" |
| Large-N NMT patient validation | "Pipeline validated on [N] recordings; expanded cohort analysis identified as future work pending clinical data access" |
| Variance-matching loss | "Designed in §4.4.5 as future enhancement to directly penalize amplitude collapse" |

---

## 9. Success Criteria (Definition of Done)

By May 3, the project must have:

- [ ] **Model**: Best available checkpoint (retrained or epoch 24) serving predictions successfully
- [ ] **Data**: Fresh synthetic dataset generated and training completed
- [ ] **Phase 3**: NMT preprocessing functional on at least 1 EDF file
- [ ] **Phase 4**: CMA-ES producing plausible EI on at least 1 patient
- [ ] **Phase 5**: Validation metrics (DLE, AUC, temporal corr) computed on synthetic test set
- [ ] **Integration**: Full pipeline testable via web UI (EDF upload → heatmap)
- [ ] **Code**: All branches merged, all tests passing, `./start.sh --check` clean