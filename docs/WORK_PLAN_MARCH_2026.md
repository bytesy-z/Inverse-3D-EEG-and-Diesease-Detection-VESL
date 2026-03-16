# PhysDeepSIF — Work Plan: March 17 – April 13, 2026

**Generated:** March 16, 2026  
**Timeline Updated:** March 16, 2026 (shifted 11 days forward from original Mar 6 start)
**Team:** Zik (Muhammad Zikrullah Rehman), Shahliza (Shahliza Ahmad), Hira (Hira Sardar)
**Deadline:** April 13, 2026 (Sunday, end of 4-week sprint)
**Calendar days:** 28 | **Working days (estimate):** ~20

---

## 1. Current State — Sanity Check Summary

### What IS implemented and working

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 1 — Forward modeling | ✅ Complete | Source space, Epileptor, leadfield, synthetic dataset generation |
| Phase 1 — Spectral shaping | ✅ Complete | STFT-based alpha/beta processing integrated into generator |
| Phase 1 — Biophysical validation | ✅ Complete | 13/13 metrics PASS |
| Phase 2 — Network architecture | ✅ Complete | PhysDeepSIF: 410,244 params (spatial MLP + biLSTM) |
| Phase 2 — Loss functions | ✅ Complete | 5 sub-losses: source MSE, forward, Laplacian, temporal, amplitude |
| Phase 2 — Training loop | ✅ Complete | Trainer with augmentation, early stopping, all 4 metrics |
| Phase 2 — DC offset de-meaning | ✅ Complete | Implemented in training script (HDF5Dataset + in-memory + stats), backend, and biomarker script |
| Phase 2 — Initial training | ✅ Complete | Epoch 24, val_loss=1.0377, trained WITH de-meaning |
| Web app v3 — Backend | ✅ Complete | FastAPI, /api/analyze, /api/biomarkers, Plotly 3D brain |
| Web app v3 — Frontend | ✅ Complete | Next.js 15, unified /analysis dashboard, dark theme |
| Data artifacts | ✅ Present | leadfield_19x76.npy, connectivity_76.npy, region_centers, labels |

### What is NOT implemented

| Component | Status | Tech Specs Section |
|-----------|--------|-------------------|
| Hyperparameter optimization (Optuna) | ❌ Not started | §4.5 |
| Forward loss normalization fix | ❌ Not started | §4.4.5 |
| Synthetic data regeneration | ❌ Not started | Copilot instructions TODO #1 |
| Phase 3 — NMT preprocessing | ❌ Not started | §5 |
| Phase 4 — CMA-ES inversion | ❌ Not started | §6 |
| Phase 5 — Validation suite | ❌ Not started | §7 |
| Phase 5 — Classical baselines | ❌ Not started | §7.1.2 |
| Biomarker module (full CMA-ES-based) | ❌ Not started | §6.3 |
| EEG waveform comparison in UI | ❌ Not specified, not built | Feedback |
| `scripts/05_hyperparam_search.py` | ❌ Referenced but missing | §12 |

### Discrepancies Found Between Docs and Reality

| Issue | In Docs | Reality | Action Needed |
|-------|---------|---------|---------------|
| Best checkpoint | §12 says "epoch 19, val_loss=1.0141" | Actual is epoch 24, val_loss=1.0377 | Update docs |
| Phase 2 status | §12 says "de-meaning not yet implemented" | De-meaning IS implemented and trained | Update §12 |
| Training data location | Instructions reference `data/synthetic3/` | Directory does not exist on this machine | Data must be regenerated or restored |
| Conda env name | Instructions say "deepsif" | Actual env is `physdeepsif` | Update instructions |
| Normalization stats | §4.4.3 says EEG std=559.2, src_std=0.2561 | Current: EEG std=10.09, src_std=0.234 (de-meaned) | Stats from different training run |
| EEG waveform feature | Not in tech specs at all | Demo feedback requires it | Must add to §9 |

---

## 2. Remaining Work Items — Prioritized

### CRITICAL PATH (model quality — blocks everything downstream)

**W1. Regenerate Synthetic Data** (~16–24h compute)
- Current `data/synthetic3/` is missing and was generated with older code
- Regenerate using current `src/phase1_forward/synthetic_dataset.py` with:
  - Integrated spectral shaping (STFT-based)
  - Skull attenuation filter
  - All fixes since original generation
- Output: `data/synthetic4/` — train.h5 (80k), val.h5 (10k), test.h5 (10k)
- **Can run unattended on GPU machine**

**W1.5. DC Offset & Forward Loss Fix** (~1–2 days code)

The DC offset problem has two parts. Here is the exact status:

| Component | Status | Detail |
|-----------|--------|--------|
| Root cause analysis | ✅ Done | Epileptor x2-x1 DC offset dominates 98.1% of power; varies with x0 |
| Per-region temporal de-meaning (training) | ✅ Done | Implemented in `HDF5Dataset.__iter__()` and `normalize_data()` |
| Per-channel de-meaning (inference) | ✅ Done | Implemented in `backend/server.py` and `demo_biomarker_detection.py` |
| Current model trained with de-meaning | ✅ Done | Checkpoint epoch 24, normalization stats show near-zero means |
| **Forward loss scale mismatch** | ❌ NOT DONE | Forward loss is 41,456× larger than source loss (§4.4.5) |
| **Forward loss normalization** | ❌ NOT DONE | Need to normalize by EEG variance or leadfield column norms |
| **Amplitude collapse still present** | ❌ NOT FIXED | Model outputs σ≈0.04 instead of σ≈0.2; near-zero temporal dynamics |
| **Validation of AC dynamics** | ❌ NOT DONE | Must verify retrained model produces real temporal variation |

**What remains to be done (all assigned to Zik):**

1. **Forward loss normalization** in `src/phase2_network/loss_functions.py`:
   - Normalize forward consistency loss by EEG variance: $\mathcal{L}_{fwd} = \frac{\|\mathbf{L}\hat{S} - \text{EEG}\|^2}{\text{Var}(\text{EEG})}$
   - Or scale leadfield to unit-norm columns before computing forward loss
   - Or simply reduce β to ~0.001 (found via Optuna)
   - **Goal:** source loss and forward loss should be on comparable scales

2. **Optional: variance-matching loss term** in `loss_functions.py`:
   - $\mathcal{L}_{var} = \sum_i (\text{Var}(\hat{S}_i) - \text{Var}(S_i))^2$
   - Directly penalizes the model for producing flat/DC output
   - Epileptogenic regions have 3.9× higher variance — this makes the useful signal an explicit target

3. **Post-retraining verification** (part of W3):
   - Confirm predicted source σ ≈ 0.2 (not 0.04)
   - Confirm temporal variance > 0.01 (not 0.00002)
   - Confirm spatial CV > 0.1 (not 0.002)
   - Confirm AUC improves from 0.495 baseline
   - Confirm variance-based scoring (not inverted-range) now works for biomarker detection

**W2. Hyperparameter Optimization** (~25–50h compute)
- Create `scripts/05_hyperparam_search.py` using Optuna TPE
- Search space per §4.5: α, β, γ, LR, weight_decay, batch_size, T₀, LSTM dropout
- Key fix: β (forward loss weight) must drop drastically — forward loss is 41,456× too large (§4.4.5)
- Include forward loss normalization as a categorical Optuna choice (raw vs EEG-variance-normalized vs leadfield-normalized)
- Optionally include variance-matching loss weight λ_var ∈ [0, 1.0] in search space
- Objective: maximize temporal correlation
- 50–100 trials, 30–45 min each
- Output: best hyperparameters stored as YAML/JSON

**W3. Full Retraining With Optimal Hyperparameters** (~2–4h)
- Train on fresh synthetic4 data with best Optuna hyperparams
- Validate against all 4 metrics: DLE < 20mm, SD < 30mm, AUC > 0.85, temporal corr > 0.7
- **DC offset verification**: confirm model no longer outputs 100% DC (see W1.5 step 3)
- Save final checkpoint to `outputs/models/`

### HIGH PRIORITY (feature completeness)

**W4. EEG Waveform Comparison in UI** (~3–4 days)
- **Not in tech specs** — new requirement from demo feedback
- Backend: return EEG channel data (19×400 per window) alongside brain plot
- Frontend: side-by-side layout — EEG montage plot (left) + brain viz (right)
- Synchronized: selecting a frame in the brain animation highlights the corresponding 2s EEG window
- Use Plotly line traces for EEG (standard montage display with channel offsets)
- Must work for both source localization and biomarker views

**W5. Phase 3 — NMT Preprocessing Pipeline** (~3–4 days code)
- Create `src/phase3_inference/nmt_preprocessor.py`
- Implement 6-step pipeline per §5.2: load EDF → verify channels → keep linked-ear ref → bandpass 0.5–70 Hz → 50 Hz notch → ICA artifact removal → 2s segmentation → z-score
- Create `src/phase3_inference/inference_engine.py`: batch inference wrapper
- Create `src/phase3_inference/source_aggregator.py`: aggregate epoch estimates → per-patient source power profile

**W6. Full Biomarker Module** (~3–4 days code)
- Currently: demo script using inverted-range scoring (heuristic)
- Target: proper CMA-ES-based epileptogenicity index per §6
- Create `src/phase4_inversion/objective_function.py`: J(x0) = w1·J_source + w2·J_eeg + w3·J_reg
- Create `src/phase4_inversion/cmaes_optimizer.py`: CMA-ES wrapper (pop=50, σ=0.3, max_gen=200)
- Create `src/phase4_inversion/epileptogenicity_index.py`: EI computation from fitted x0

**W7. Phase 5 — Validation Suite** (~4–5 days code)
- Create `src/phase5_validation/synthetic_metrics.py`: full metric suite across noise levels (5, 10, 15, 20, 30 dB)
- Create `src/phase5_validation/classical_baselines.py`: eLORETA, MNE, dSPM, LCMV
- Create `src/phase5_validation/patient_validation.py`: intra-patient consistency, cross-segment stability, normal vs abnormal disc
- Output: tables and plots for thesis/paper

### MEDIUM PRIORITY (polish)

**W8. Vectorize Laplacian Loss** (~2h)
- Replace Python for-loop in `loss_functions.py` with `torch.einsum` or batch matmul
- Performance improvement only, no functional change

**W9. Update Documentation** (~1 day)
- Fix all discrepancies listed in Section 1 above
- Add EEG waveform feature to §9
- Update §12 completion status
- Update copilot-instructions with current state

**W10. Integration Testing & Final Demo** (~2 days)
- End-to-end test: EDF upload → preprocessing → inference → CMA-ES → heatmap → EEG comparison
- Performance/load testing
- Final bug fixes

---

## 3. Dependency Graph

```
W1 (regen data) ──→ W1.5 (fwd loss fix) ──→ W2 (Optuna search) ──→ W3 (retrain + DC verify)
                                                                 ╰──→ W6 (biomarker CMA-ES)
                                                                 ╰──→ W7 (validation suite)

W4 (EEG UI) ──────→ (independent, needs only current backend)

W5 (NMT preproc) ──→ W6 (biomarker needs real EEG preprocessed)
                  ╰──→ W7 (validation needs NMT data)

W8 (vectorize loss) ─→ W1.5 (makes loss changes cleaner)
W9 (docs update) ─────→ (independent, done anytime)
```

**Key insight:** W1→W1.5→W2→W3 is the critical path. W1.5 (forward loss fix) can be coded while W1 runs overnight. W4, W5, W8 are fully independent and can start immediately in parallel.

---

## 4. Work Assignment — Three-Person Parallel Split

### Guiding Principles
1. **Minimize merge conflicts**: each person owns separate directories/files
2. **Shared interfaces defined up front**: agree on function signatures and data formats before branching
3. **GPU-bound work (W1, W2, W3) must be serialized** on the GPU machine — assigned to one person
4. **CPU-only work can be truly parallel across machines**

---

### 🟢 ZIK — Model & Training Pipeline (Branch: `zik/model-optimization`)

**Rationale:** Zik has the deepest context on the model, training script, loss functions, and data generation pipeline. GPU access is required.

| Week | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Mar 17–23 | **W1**: Regenerate synthetic data using current code | `data/synthetic4/{train,val,test}.h5` |
| | | **W8**: Vectorize Laplacian loss (while data generates) | Updated `loss_functions.py` |
| | | **W1.5**: Implement forward loss normalization + optional variance loss | Updated `loss_functions.py` with normalized forward loss |
| | | **W2**: Create and start Optuna hyperparam search script | `scripts/05_hyperparam_search.py`, initial trials running |
| **Week 2** | Mar 24–30 | **W2**: Complete Optuna search (50–100 trials) | Best hyperparams in `outputs/optuna_best.json` |
| | | **W3**: Full retraining with best hyperparams | New `outputs/models/checkpoint_best.pt` with improved metrics |
| | | Validate model meets targets (DLE<20, AUC>0.85, corr>0.7) | Validation results log |
| | | **DC offset verification**: confirm model outputs real dynamics (σ≈0.2, temporal var>0.01) | DC verification report |
| **Week 3** | Mar 31–Apr 6 | **W6**: CMA-ES objective function + optimizer | `src/phase4_inversion/{objective_function,cmaes_optimizer,epileptogenicity_index}.py` |
| | | Integrate CMA-ES biomarker into backend API | Updated `backend/server.py` with CMA-ES endpoint |
| **Week 4** | Apr 7–13 | **W10**: Integration testing of full pipeline | End-to-end test results |
| | | **W9**: Update technical specs and copilot instructions | Updated docs |
| | | Bug fixes, merge conflict resolution | Clean merge to main |

**Files Zik owns exclusively:**
- `scripts/02_generate_synthetic_data.py`
- `scripts/03_train_network.py`
- `scripts/05_hyperparam_search.py` (new)
- `src/phase2_network/loss_functions.py`
- `src/phase2_network/trainer.py`
- `src/phase4_inversion/` (entire new directory)
- `data/synthetic4/` (new)
- `outputs/models/`
- `config.yaml` (training-related sections only)

---

### 🔵 HIRA — Frontend & Visualization (Branch: `hira/eeg-visualization`)

**Rationale:** Hira has worked on the frontend (credited in AppFooter). The EEG waveform feature is self-contained in the frontend + one backend endpoint.

| Week | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Mar 17–23 | **W4a**: Design EEG waveform component | Figma/mockup + component spec |
| | | **W4b**: Create `EEGWaveformPlot` React component | `frontend/components/eeg-waveform-plot.tsx` |
| | | Plotly line traces with 19-channel montage display, channel offsets, amplitude scale | Working standalone component with mock data |
| **Week 2** | Mar 24–30 | **W4c**: Backend — add EEG data to API response | Modified response schema in `backend/server.py` (coordinate with Zik) |
| | | **W4d**: Side-by-side layout in analysis page | Updated `frontend/app/analysis/page.tsx` |
| | | Frame synchronization: brain animation frame ↔ EEG window highlight | Synced interaction working |
| **Week 3** | Mar 31–Apr 6 | **W4e**: Polish EEG display | Axis labels, time stamps, channel names, clinical styling |
| | | Responsive layout (fullscreen EEG, fullscreen brain, side-by-side) | All viewport sizes working |
| | | Add support for both source localization and biomarker views | Both tabs show relevant EEG |
| **Week 4** | Apr 7–13 | **W10**: UI integration testing with retrained model | All routes working end-to-end |
| | | **W9**: Update §9 of tech specs with EEG waveform feature | Documentation |
| | | Visual polish, accessibility check, loading states | Production-ready UI |

**Files Hira owns exclusively:**
- `frontend/app/analysis/page.tsx`
- `frontend/components/eeg-waveform-plot.tsx` (new)
- `frontend/components/brain-visualization.tsx`
- `frontend/components/results-summary.tsx`
- `frontend/app/globals.css` / `frontend/styles/`
- `frontend/lib/` (except `job-store.ts` types — coordinate with Zik)

**Shared files (coordinate changes):**
- `backend/server.py` — Hira adds EEG data fields to API response (small, isolated change)
- `frontend/lib/job-store.ts` — type definitions may need updating

---

### 🟠 SHAHLIZA — Preprocessing, Validation & Baselines (Branch: `shahliza/validation-pipeline`)

**Rationale:** Phase 3, Phase 5, and classical baselines are independent of the model training pipeline and the frontend. They depend only on the existing trained model and data formats.

| Week | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Mar 17–23 | **W5a**: Create `src/phase3_inference/__init__.py` | Package structure |
| | | **W5b**: Implement `nmt_preprocessor.py` per §5.2 | 6-step preprocessing pipeline |
| | | Test with sample EDF (`data/samples/0001082.edf`) | Preprocessed output (n_epochs, 19, 400) |
| **Week 2** | Mar 24–30 | **W5c**: Implement `inference_engine.py` | Batch inference wrapper |
| | | **W5d**: Implement `source_aggregator.py` | Patient-level source power profile |
| | | **W7a**: Create `src/phase5_validation/__init__.py` | Package structure |
| | | **W7b**: Implement `synthetic_metrics.py` | Full metric suite across 5 noise levels |
| **Week 3** | Mar 31–Apr 6 | **W7c**: Implement `classical_baselines.py` | eLORETA, MNE, dSPM, LCMV on same test set |
| | | **W7d**: Implement `patient_validation.py` | Intra-patient consistency, cross-segment stability |
| | | Run baselines on test set (once Zik provides retrained model) | Comparison tables |
| **Week 4** | Apr 7–13 | **W7e**: Normal vs abnormal NMT discrimination analysis | AUROC for max(EI) classifier |
| | | Generate thesis/paper validation figures and tables | Plots in `outputs/validation/` |
| | | **W10**: End-to-end integration testing | Full pipeline validation |

**Files Shahliza owns exclusively:**
- `src/phase3_inference/` (entire new directory)
- `src/phase5_validation/` (entire new directory)
- `scripts/06_run_validation.py` (new — validation driver script)
- `scripts/07_run_baselines.py` (new — classical baselines script)
- `outputs/validation/` (new — results directory)

**Dependencies on Zik:**
- Week 3+: needs the retrained model checkpoint from W3 to run final validation
- Can use current checkpoint (epoch 24) for development and testing during weeks 1–2

---

## 5. Interface Contracts (Agree Before Branching)

These interfaces must be agreed upon **before** anyone starts coding to prevent merge conflicts.

### 5.1 Backend API — EEG Data in Response (Hira ↔ Zik)

```python
# Addition to /api/analyze and /api/biomarkers response
{
    # ... existing fields ...
    "eegData": {
        "channels": ["Fp1", "Fp2", ...],       # 19 channel names
        "samplingRate": 200,                     # Hz
        "windowLength": 400,                     # samples per window
        "windows": [                             # list of n_windows
            {
                "startTime": 0.0,                # seconds from recording start
                "endTime": 2.0,
                "data": [[...], [...], ...]      # (19, 400) nested list, µV scale
            },
            ...
        ]
    }
}
```

**Owner:** Hira adds this to the backend response. Zik reviews.

### 5.2 Phase 3 Output → Phase 4/5 Input (Shahliza ↔ Zik)

```python
# src/phase3_inference/inference_engine.py
def run_patient_inference(
    edf_path: Path,
    model: PhysDeepSIF,
    norm_stats: dict,
    device: torch.device
) -> dict:
    """
    Returns:
        {
            "source_estimates": ndarray (n_epochs, 76, 400) float32,
            "mean_source_power": ndarray (76,) float64,
            "activation_consistency": ndarray (76,) float64,
            "preprocessed_eeg": ndarray (n_epochs, 19, 400) float32,
            "n_epochs": int,
            "epoch_times": list[float]  # start time of each epoch in seconds
        }
    """
```

**Owner:** Shahliza implements. Zik consumes in CMA-ES module.

### 5.3 CMA-ES Output → Validation Input (Zik ↔ Shahliza)

```python
# src/phase4_inversion/cmaes_optimizer.py
def fit_patient(
    patient_inference: dict,  # output of run_patient_inference
    leadfield: ndarray,
    connectivity: ndarray,
    config: dict
) -> dict:
    """
    Returns:
        {
            "x0_fitted": ndarray (76,) float64,  # fitted excitability per region
            "epileptogenicity_index": ndarray (76,) float64,  # EI ∈ [0, 1]
            "convergence_history": list[float],  # J per generation
            "n_generations": int,
            "final_objective": float
        }
    """
```

**Owner:** Zik implements. Shahliza consumes in validation pipeline.

---

## 6. Git Workflow & Merge Strategy

### Branch Structure

```
main (production)
  ├── zik/model-optimization
  ├── hira/eeg-visualization
  └── shahliza/validation-pipeline
```

### Merge Schedule

| Date | Action | Who |
|------|--------|-----|
| **Mar 17** | All branch from `main` at the same commit | Everyone |
| **Mar 23** (end of Week 1) | **Sync checkpoint**: Zik pushes updated `config.yaml` and any shared type changes to `main`. Hira and Shahliza rebase. | Zik merges first |
| **Mar 30** (end of Week 2) | **Major integration**: Zik merges retrained model + data path changes. Shahliza merges Phase 3. Hira merges EEG component. Order: Zik → Shahliza → Hira (resolve any conflicts at each step). | All three |
| **Apr 6** (end of Week 3) | **Feature freeze**: Zik merges CMA-ES. Shahliza merges validation. Hira merges synced EEG display. | All three |
| **Apr 10** | **Code freeze**: only bug fixes after this point | Everyone |
| **Apr 13** | Final merge + tag v2.0 release | Zik |

### Conflict Minimization Rules

1. **`backend/server.py` is the only shared file with real conflict risk.**
   - Hira: only modify response schemas (add `eegData` field) — touch only `run_inference()` return dict and response models
   - Zik: only modify CMA-ES endpoint (add new `/api/biomarkers-cmaes` route) — do NOT modify existing routes
   - Shahliza: does NOT touch backend at all

2. **No one touches another person's directories.** Ownership is exclusive per the file lists above.

3. **`config.yaml` changes:**
   - Zik owns training/model sections
   - Shahliza may add `preprocessing` and `validation` sections
   - Use YAML anchors or separate sections to avoid line-level conflicts

4. **Shared data files** (`data/*.npy`, `data/*.json`): read-only, no one modifies these.

5. **When in doubt, communicate** before touching a shared file.

---

## 7. Timeline — Gantt View

```
Week 1: Mar 17–23
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZIK:      [████ W1: Regen data ████][█ W8+W1.5: Loss fixes █][ W2: Start Optuna ]
HIRA:     [██ W4a: Design ██][████████ W4b: EEG component ████████]
SHAHLIZA: [██████████████ W5a+b: NMT preprocessor █████████████████]

Week 2: Mar 24–30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZIK:      [████████████ W2: Optuna search (50-100 trials) █████████████]
          [███ W3: Retrain + DC verify ███]
HIRA:     [██ W4c: Backend EEG ██][████ W4d: Side-by-side layout ████]
SHAHLIZA: [██ W5c+d: Inference engine ██][██ W7a+b: Synthetic metrics ██]

          ─── Mar 30: MAJOR INTEGRATION MERGE ───

Week 3: Mar 31–Apr 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZIK:      [████████████ W6: CMA-ES inversion module █████████████████]
HIRA:     [████ W4e: EEG polish ████][██ Responsive/a11y ██]
SHAHLIZA: [██ W7c: Classical baselines ██][██ W7d: Patient validation ██]

          ─── Apr 6: FEATURE FREEZE MERGE ───

Week 4: Apr 7–13
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZIK:      [██ W10: Integration ██][██ W9: Docs ██][█ Final merge █]
HIRA:     [██ W10: UI testing ████][█ W9: §9 docs █][█ Polish █]
SHAHLIZA: [██ W7e: NMT discrimination ██][██ Figures/tables ██][█ W10 █]

          ─── Apr 10: CODE FREEZE (bug fixes only) ───
          ─── Apr 13: FINAL RELEASE v2.0 ───
```

---

## 8. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data regeneration takes >24h or fails | Medium | CRITICAL — blocks W2, W3 | Start immediately (Mar 17); run overnight; have fallback to find synthetic3 backup on TUKL PC |
| Optuna search doesn't find good hyperparams in 100 trials | Medium | High — model metrics stay poor | Start with manual β reduction first (set β=0.001); if AUC improves, narrow Optuna range around that; consider variance-matching loss as backup |
| CMA-ES too slow (3h/patient too long for demo) | Medium | Medium — can use heuristic fallback | Keep current inverted-range scoring as fast fallback; CMA-ES can run async/batch |
| Training data not available (synthetic3 deleted) | HIGH | CRITICAL | W1 regenerates fresh data — this is already the plan |
| ICA in NMT preprocessing flaky across patients | Low | Medium — some patients fail | Add robust fallback: skip ICA if <5 components found, flag patients with high artifact residual |
| Merge conflicts in server.py | Medium | Low — localized | Follow strict ownership rules above; use separate endpoints for new features |
| Model doesn't reach AUC > 0.85 target | Medium | High | Accept AUC > 0.7 as minimum; document in thesis as "promising, needs further optimization"; variance loss term as escalation |

---

## 9. Success Criteria (Definition of Done)

By April 13, the project must have:

- [ ] **Model**: Retrained PhysDeepSIF with AUC ≥ 0.70, temporal correlation ≥ 0.30, DLE < 20mm
- [ ] **Data**: Fresh synthetic4 dataset generated with current code
- [ ] **UI**: EEG waveform displayed alongside brain visualization, synchronized with animation frames
- [ ] **Phase 3**: NMT preprocessing pipeline functional, tested on sample EDF
- [ ] **Phase 4**: CMA-ES optimizer functional (even if slow), produces EI per region
- [ ] **Phase 5**: Validation suite runs on synthetic test set; classical baselines computed
- [ ] **Docs**: Tech specs §12 updated, all discrepancies fixed
- [ ] **Integration**: Full pipeline tested end-to-end (EDF → preprocessing → inference → visualization)

### Stretch Goals (if time permits)
- [ ] AUC > 0.85 and temporal correlation > 0.70
- [ ] NMT normal vs abnormal discrimination AUROC computed
- [ ] Variance-matching loss term implemented and tested
- [ ] Patient consistency analysis on multiple NMT recordings

---

## 10. Immediate Next Actions (Starting March 17)

| Person | Action |
|--------|--------|
| **Zik** | 1. Create branches. 2. Start data regeneration (W1) — kick off `scripts/02_generate_synthetic_data.py` overnight. 3. While waiting, vectorize Laplacian loss (W8). 4. Implement forward loss normalization (W1.5) — the core DC offset fix for training. |
| **Hira** | 1. Pull latest main, create branch. 2. Review current `/analysis` page and `BrainVisualization` component. 3. Start designing EEG waveform component (mock data first, no backend dependency). |
| **Shahliza** | 1. Pull latest main, create branch. 2. Read §5.2 of tech specs thoroughly. 3. Start implementing `src/phase3_inference/nmt_preprocessor.py` — test with `data/samples/0001082.edf`. |
| **All** | By Mar 17: Review and agree on interface contracts (Section 5 of this document) before branching and starting code. |

---

## Appendix A: File Ownership Matrix

| File/Directory | Zik | Hira | Shahliza |
|---------------|-----|------|----------|
| `src/phase1_forward/` | ✏️ own | — | — |
| `src/phase2_network/` | ✏️ own | — | — |
| `src/phase3_inference/` (new) | — | — | ✏️ own |
| `src/phase4_inversion/` (new) | ✏️ own | — | — |
| `src/phase5_validation/` (new) | — | — | ✏️ own |
| `scripts/02_*`, `03_*`, `05_*` | ✏️ own | — | — |
| `scripts/06_*`, `07_*` (new) | — | — | ✏️ own |
| `backend/server.py` | ✏️ own | ✏️ EEG response only | 👀 read |
| `frontend/app/analysis/` | 👀 read | ✏️ own | — |
| `frontend/components/` | 👀 read | ✏️ own | — |
| `frontend/lib/` | ✏️ types only | ✏️ own | — |
| `config.yaml` | ✏️ training sections | — | ✏️ preprocessing/validation sections |
| `data/*.npy`, `data/*.json` | 👀 read | 👀 read | 👀 read |
| `data/synthetic4/` (new) | ✏️ own | — | 👀 read |
| `outputs/models/` | ✏️ own | — | 👀 read |
| `outputs/validation/` (new) | — | — | ✏️ own |
| `docs/02_TECHNICAL_SPECIFICATIONS.md` | ✏️ own | ✏️ §9 only | ✏️ §5, §7 only |
| `.github/copilot-instructions.md` | ✏️ own | — | — |

Legend: ✏️ = can modify, 👀 = read only, — = no access needed

## Appendix B: Compute Resource Allocation

| Task | Resource | Estimated Time | Machine |
|------|----------|---------------|---------|
| W1: Data generation | GPU + 8 CPU cores | 16–24h | TUKL GPU server |
| W1.5: Forward loss fix | CPU only (code changes) | 4–8h | Any machine |
| W2: Optuna search | GPU + 8 CPU cores | 25–50h | TUKL GPU server |
| W3: Retraining | GPU | 2–4h | TUKL GPU server |
| W5: NMT preprocessing | CPU only | Development time | Any machine |
| W6: CMA-ES development | CPU only (TVB sims) | Development time | Any machine |
| W7: Validation/baselines | CPU + 1 GPU (inference) | ~4h compute | Any machine with GPU |

**GPU bottleneck:** W1, W2, W3 must be serialized on the GPU machine. Total GPU time: ~45–78h. This is the critical path — start W1 immediately.
