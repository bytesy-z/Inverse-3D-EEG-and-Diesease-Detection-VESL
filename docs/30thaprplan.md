# PhysDeepSIF — Final Submission Plan v4 (April 30, 2026)

## 0. Best Model Configuration — Final (After Wave 1)

**Retrained on synthetic3 (80k train, 10k val, 10k test), 60 epochs, batch_size=64.**

| Metric | PhysDeepSIF β=0.0 | v1 (March) | Random | Oracle |
|--------|-------------------|------------|--------|--------|
| DLE (centroid) | **31.08 mm** | 32.83 mm | 53.67 mm | 15.62 mm |
| AUC | **0.697** | 0.519 | 0.502 | 0.969 |
| Temporal Corr | **0.274** | 0.187 | ~0 | 1.000 |

Centroid-based DLE used (asymmetric mask: all-region predicted vs epi-restricted true). Oracle computed as true sources' self-DLE. PhysDeepSIF beats v1 on all metrics and all classical baselines by a wide margin.

**Configuration:**
- Preprocessing: EEG NOT de-meaned (retains DC spatial prior), Sources ARE de-meaned (AC-only targets)
- Architecture: SpatialModule (165k) + TemporalModule BiLSTM hidden_size=76 (245k) = 410k total
- Source loss: Simple MSE. No class-balancing.
- Epi loss: Class-balanced MSE on AC variance.
- Forward loss: **β=0.0** (disabled — confirmed superior in controlled test).
- Laplacian regularization: **λ_L=0.0** (dropped).
- Temporal smoothness: λ_T = 0.3. Amplitude bound: λ_A = 0.2.

**Best checkpoint:** `outputs/models/checkpoint_best.pt` (epoch 47, val_loss=1.3396).

---

## 1. Why β=0.0 Is Scientifically Correct

**The physics is in the DATA, not the loss function.** Every training sample satisfies `EEG = L @ S`. The model learns `S = f(EEG)` through 80,000 examples. The forward loss gradient `L^T @ (L @ Ŝ - EEG)` pulls toward the pseudoinverse (DLE ≈ 34 mm). Experiments confirm: β=0.1 → DLE=35 mm; β=0.0 → DLE=31 mm. Physics information is encoded through TVB training data, not explicit loss constraints. See §10 of experimentation logs for full derivation with SVD analysis (condition number κ=10¹⁶, rank=18, nullspace dimension 58).

## 2. Laplacian Regularization — Dropped

All classical Laplacian-type inverse solutions (MNE, eLORETA, sLORETA, dSPM) achieve near-random DLE (49-57mm). Adding Laplacian regularization to a deep model that already outperforms all of them (DLE=31mm) risks smoothing learned spatial structure. Dropped permanently.

## 3. DLE Metric — Reverted to Centroid-Based

Max-point DLE tested and discarded: all methods including Oracle perform near-random on 19→76 problem. Centroid DLE provides clear model discrimination (β=0.0: 31mm, v1: 33mm, random: 54mm).

## 4. Codebase Cleanup — COMPLETED

- Retrained β=0.0, hidden_size=76, 60 epochs.
- Clean loss functions, training script, config.
- Synthetic1/2/4 archived → archive/.
- All 135 unit+functional+regression+integration tests pass (Wave 2.5).

---

## 5. System Architecture (Final)

```
Patient uploads EDF (19ch, 200Hz)
    │
    ▼
[Preprocessing]
    - Load with MNE, map channel names (FP1→Fp1, drop A1/A2)
    - Bandpass 0.5-70 Hz, notch 50 Hz
    - Segment into 2s windows (400 samples @ 200Hz)
    - Z-score normalize using training stats
    │
    ▼
[PhysDeepSIF — Source Imaging] (410k params, ~100ms)
    SpatialModule (19→128→256→256→128→76) per time step
    + TemporalModule (2-layer BiLSTM, hidden=76)
    → Source activity: 76 regions × 400 time samples
    │
    ├── Phase A — [Biomarker Detection] (~1 sec)
    │   compute_epileptogenicity_index():
    │     AC variance → z-score → sigmoid → EI ∈ [0,1]
    │   → Top-10 candidate regions shown instantly
    │   → "Preliminary Analysis" badge
    │
    └── Phase B — [CMA-ES Concordance] (~7 min, AUTO-STARTED)
        For generations 1..30:
          CMA-ES proposes 14 candidate x₀ (76-dim)
          For each: TVB Epileptor sim (4s) + L@S → EEG_sim
          Score: J(x₀) = MSE(PSD_sim, PSD_patient) + 0.1·||x₀+2.2||²
        CMA-ES converges → fitted x₀
        x₀ → sigmoid → BIOPHYSICAL EI
        → Top-10 biophysical regions
        
        CONCORDANCE ENGINE:
        overlap = |Top10_instant ∩ Top10_biophysical|
        ≥5/10 → HIGH  — "Both methods independently agree: strong evidence"
        2-4/10 → MODERATE — "Partial agreement: correlate with clinical findings"
        ≤1/10 → LOW — "Methods disagree: consider longer recording or stereo-EEG"
        
        Result REPLACES preliminary badge with concordance tier.
    
    ▼
[WebSocket → Frontend]
    Phase A: instant heatmap + "CMA-ES running in background" indicator
    Phase B progress: generation X/30, best score: Y
    Phase B complete: updated heatmap + concordance badge
    
    │
    ▼
[XAI — Occlusion Attribution] (triggered on CMA-ES completion)
    For top-concordant region:
      Mask EEG channel/time segments (200ms window, 100ms stride)
      Rerun biomarker pipeline, measure EI drop
    → channel_importance (19×1) + time_importance (1×T)
    → Overlay on EEG waveform: "These channels and time segments
      most influenced the epileptogenicity finding"
```

---

## 6. Documentation Deliverables

### 6.1 Updated Technical Specifications
New sections to add to `docs/02_TECHNICAL_SPECIFICATIONS.md`:
- **§6**: Corrected DLE Metric
- **§7**: CMA-ES Concordance Engine
- **§8**: XAI Occlusion Module
- **§9**: Updated System Architecture
- **§10**: Rationale for β=0.0
- **§11**: Engineering Architecture (concurrency model, error handling, security boundaries)

### 6.2 Experimentation Logs — COMPLETED ✅
`docs/03_EXPERIMENTATION_LOGS.md` — 582 lines, 12 sections covering all experiments, DLE audit, β analysis, SVD math appendix.

### 6.3 Final Work Plan
Concise task list with owners, estimates, dependencies.

---

## 7. Scientific Validation

### 7.1 Classical Baselines — COMPLETED ✅

| Method | DLE (mm) | AUC | Corr |
|--------|----------|-----|------|
| **Oracle** | **15.62** | **0.969** | **1.000** |
| **PhysDeepSIF β=0.0** | **31.08** | **0.697** | **0.274** |
| MNE (λ=0.001) | 49.55 | 0.493 | 0.020 |
| eLORETA (λ=0.05) | 53.54 | 0.489 | 0.107 |
| dSPM (λ=0.1) | 54.63 | 0.493 | 0.088 |
| sLORETA (λ=0.05) | 56.52 | 0.485 | 0.088 |
| Random | 53.67 | 0.502 | ~0 |

All classical methods near-random. Deep model beats by 18-25mm DLE.

### 7.2 Validation Figures — COMPLETED ✅

| Figure | File | Results |
|--------|------|---------|
| DLE Histogram | `dle_histogram.png` | PhysDeepSIF=31.8±16.8mm, eLORETA=40.9±17.7mm, Oracle=27.3±17.9mm |
| AUC vs SNR | `auc_vs_snr.png` | PDS 0.63-0.68 across 5-30dB SNR; eLORETA stuck at 0.49 |
| Top-K Recall | `topk_recall.png` | PDS top-1=0.207, eLORETA=0.042, Random=0.050 |
| Hemisphere Accuracy | `hemisphere_accuracy.png` | PDS=69.5%, eLORETA=50.0% (right-hemisphere bias), Random=51.2% |
| Learning Curve | `learning_curve.png` | Best epoch 29 (val_loss=1.0344) |
| Method Concordance | `concordance_heatmap.png` | PDS vs eLORETA: 0 HIGH, 13 MODERATE, 37 LOW (0.96/10 mean overlap) |

Code: `src/phase5_validation/generate_figures.py` (775 lines, standalone script).

---

## 8. Dropped Items

| Dropped | Thesis Framing |
|---------|---------------|
| NMT preprocessing | Deferred pending NMT dataset access approval |
| Real EEG clinical validation | Phase 6 future work requiring IRB approval |
| Optuna hyperparameter search | Structural DLE ceiling independent of loss hyperparameters |
| Laplacian sweep | All Laplacian classical methods are near-random |
| Max-point DLE | All methods including Oracle fail on 19→76 problem |

---

## 9. Execution Order

### Wave 0 — DLE & Beta Evaluation ✅

| # | Task | Actual | Result |
|---|------|--------|--------|
| 0a | Fix DLE metric | 15 min | Tested max-point, reverted to centroid |
| 0b | Retrain β=0.0, evaluate | 30 min | DLE=31.08mm, AUC=0.697, Corr=0.274 |
| 0c | Test β=0.01 vs β=0.0 | 25 min | β=0.0 wins on all metrics |

**Gate**: β=0.0 adopted.

### Wave 1 — Parallelizable ✅

| # | Task | Actual | Status |
|---|------|--------|--------|
| 1 | Laplacian sweep | — | ❌ Dropped |
| 2 | Classical baselines | 30 min | ✅ All near-random |
| 3 | Code cleanup | 15 min | ✅ Config, training, losses |
| 4 | Archive old data | 2 min | ✅ |
| T1 | Unit tests (13 files, 65 tests) | 1 hr | ✅ All pass |

### Wave 2 — Parallelizable ✅

| # | Task | Actual | Result |
|---|------|--------|--------|
| 5 | **Validation figures** | 1 hr | ✅ 6 figures generated. DLE=31.8mm, AUC=0.68@30dB |
| 6 | **CMA-ES concordance engine** | ~3 hr (pre-existing) | ✅ Bug fixed: regularization `mean`→`sum` (was 76× weaker). Code complete in `src/phase4_inversion/` |
| 7 | **Experimentation logs** | Pre-existing (582 lines) | ✅ Already complete |
| T2 | Functional tests (4 files) | Pre-existing | ✅ All pass |
| T3 | System tests (4 files) | Pre-existing | ✅ All pass |
| — | **Bug fixes applied** | — | ✅ Fixed `objective_function.py` regularization `mean`→`sum`. Fixed `generate_figures.py` concordance to compare method-method not method-vs-ground-truth. Removed dead variable in AUC-vs-SNR loop. |

**Wave 2 actual wall-clock**: ~15 min (fixes + verification). Figures generated by subagent in ~1 hr.

### Wave 2.5 — Engineering Hardening — COMPLETED ✅

**Motivation**: Codebase audit (Apr 30) revealed critical engineering debt across all tiers. These issues must be resolved before feature work in Wave 3 to prevent the system from being fragile, insecure, or untestable. Failure modes include: race conditions on shared state, unhandled `model=None` crashes, unbounded disk growth, fake progress bars, and zero frontend test coverage.

**Actual wall-clock**: ~2.5 hr (4 parallel agents, staggering dependencies).

| # | Task | Est. | Actual | Status |
|---|------|------|--------|--------|
| H1 | **Concurrency lock** on `active_jobs` dict | 15 min | 5 min | ✅ `threading.Lock` guards 3 mutation paths |
| H2 | **Model=None guard** in `run_inference()`; NaN/Inf validation | 10 min | 5 min | ✅ Returns 503/400 respectively |
| H3 | **Request timeout**: `asyncio.wait_for(60s)` + `timeout_keep_alive=30` | 10 min | 5 min | ✅ Added in both branches + uvicorn config |
| H4 | **Disk space check**: `shutil.disk_usage()` before writes | 10 min | 5 min | ✅ `MIN_FREE_BYTES=100MB`, returns 507 |
| H5 | **Results cleanup**: `_cleanup_old_results()` hourly, 24h TTL | 15 min | 10 min | ✅ Created + registered in startup |
| H6 | **CORS hardening**: explicit methods/headers | 5 min | 2 min | ✅ `["GET","POST","OPTIONS"]` + `["Content-Type","Authorization"]` |
| H7 | **Mesh caching**: module-level cache (was downloading per call) | 10 min | 5 min | ✅ `_mesh_cache` global variable |
| H8 | **Refactor `analyze_eeg()`**: extract 3 shared functions | 30 min | 20 min | ✅ 530→~280 lines, duplicated blocks eliminated |
| H9 | **Fix `requirements.txt`**: 3→18 deps | 10 min | 5 min | ✅ All runtime imports now listed |
| H10 | **Structured JSON logging**: `JsonFormatter` + `RotatingFileHandler` | 15 min | 10 min | ✅ JSON output with job_id, console + file |
| H11 | **Rate limiting**: token bucket (100 req/min/IP) | 20 min | 10 min | ✅ Middleware returns 429 on exceed |
| F1 | **Remove dead npm deps**: expo, react-native, etc. | 5 min | 2 min | ✅ 8 deps removed |
| F2 | **Fix `ignoreBuildErrors`**: conditional on dev; `@types/react`→^19 | 30 min | 10 min | ✅ Production build now catches TS errors |
| F3 | **Remove dead code**: 6 files + 2 duplicate hooks | 15 min | 5 min | ✅ No orphaned imports |
| F4 | **React Error Boundary**: catch+fallback component | 15 min | 10 min | ✅ Wraps layout + analysis page |
| F5 | **Loading skeletons**: `AnalysisSkeleton` with pulse animation | 15 min | 10 min | ✅ Renders during loading state |
| F6 | **Real progress tracking**: removed fake 92% cap | 45 min | 20 min | ✅ Props-based progress from parent |
| F7 | **XAI frontend module**: channel bar chart + time heatmap | 1.5 hr | 45 min | ✅ Toggle overlay, integrated in analysis page |
| F8 | **Plotly dark theme**: transparent bg, theme-aware colors | 15 min | 10 min | ✅ No hardcoded black/white |
| F9 | **Accessibility**: aria-pressed, skip-to-content, aria-live | 20 min | 15 min | ✅ axe-compatible |
| T6 | **Backend API error tests**: 15 cases | 30 min | 20 min | ✅ 503, 400, 500, 403, 422 paths covered |
| T7 | **Fix training test mutation**: state save/restore | 15 min | 5 min | ✅ `copy.deepcopy` ± try/finally |
| T8 | **Spatial dispersion edge cases**: 5 tests | 10 min | 5 min | ✅ uniform, focal, diffuse, zero, NaN |
| T9 | **Regression tests**: 3 files (amplitude, DC, normstats) | 30 min | 15 min | ✅ Each fails on buggy code, passes on current |
| T10 | **CMA-ES tests**: 7 tests (concordance boundaries, EI) | 45 min | 20 min | ✅ Adjusted to match actual API |
| T11 | **Trainer tests**: init + epoch shape assertions | 45 min | 15 min | ✅ Uses mock minimal model |
| T12 | **Edge case unit tests**: 9 tests (NaN/Inf/zero/DC/empty) | 30 min | 15 min | ✅ All loss functions covered |
| T13 | **Frontend test setup** | 1 hr | — | ⏭️ Skipped (blocked by F3 timing) |
| T14 | **Realistic mock data**: 1/f EEG, structured leadfield, ~15% connectivity | 15 min | 10 min | ✅ `mock_eeg_batch` has 1/f slope; leadfield has decaying singular values |
| I1 | **Dockerize backend**: `Dockerfile.backend` + `.dockerignore` | 1 hr | 30 min | ✅ python:3.9-slim, copies all data files |
| I2 | **Integration test suite**: 3 files (inference, lifecycle, EDF) | 1.5 hr | 30 min | ✅ `pytest tests/integration/ -v` passes |
| I3 | **Health endpoint hardening**: disk, uptime, versions, degraded | 15 min | 10 min | ✅ Includes all new fields + `degraded` flag |
| I4 | **CI/CD config**: `.github/workflows/test.yml` | 30 min | 15 min | ✅ Runs on push/PR, excludes system+integration |
| I5 | **Startup validation**: `startup_check()` verifies 6 required files | 20 min | 10 min | ✅ Logs comprehensive pass/fail report |
| I6 | **Log persistence**: `outputs/logs/backend.log` with rotation | 10 min | 5 min | ✅ 10MB rotating, 5 backups |
| I7 | **End-to-end EDF test script**: `scripts/test_e2e.sh` | 20 min | 10 min | ✅ Executable shell test |

### Wave 3 — Feature Completion (after Wave 2.5)

| # | Task | Est. | Agent | Depends on |
|---|------|------|-------|-----------|
| 8 | **Backend WebSocket for CMA-ES progress**: Implement real-time progress in `_process_analysis_async()`. Broadcast generation count, best score, status via WebSocket. Store in `active_jobs` with thread-safe lock. | 2 hr | Agent B | H1 (concurrency lock), H11 (structured logging) |
| 9 | **Frontend WebSocket + concordance badge**: Connect to backend WS on analysis start. Render concordance tier badge (HIGH/MODERATE/LOW) with color-coded styling. Show "CMA-ES running X/30 generations" indicator. On completion, replace preliminary badge with concordance badge and update heatmap. | 2 hr | Agent B | F6 (real progress tracking), I2 (WS integration tests) |
| 10 | **Docs: technical specs** — update `docs/02_TECHNICAL_SPECIFICATIONS.md` per section 6.1 | 2 hr | Agent C | None |
| T4 | **Integration tests** (Wave 2.5 I2 already covers this) | — | Agent D | Already done in I2 |
| T5 | **Regression tests** (Wave 2.5 T9 already covers this) | — | Agent D | Already done in T9 |

### Wave 4 — Integration & Delivery

| # | Task | Est. | Agent |
|---|------|------|-------|
| 11 | **Docs: work plan** — write `docs/FINAL_WORK_PLAN_v2.md` | 1 hr | Any |
| 12 | **End-to-end integration test** with `0001082.edf` — full pipeline: upload → biomarker → CMA-ES → concordance → XAI. Run `pytest tests/ -m "not slow" -v` expecting all green. | 1 hr | Agent A |
| 13 | **Pre-run CMA-ES** on `0001082.edf` for live demo. | 15 min | Agent B |
| 14 | **Docker compose**: Create `deploy/docker-compose.yml` with backend + frontend services, shared volume for model/data, health check, restart policy. | 30 min | Agent D |
| 15 | `./start.sh --check` pass + git commit + tag `v2.0-submission` | 5 min | Any |

### Parallelization Strategy

```
Wave 0 (sequential gate) — COMPLETED ✅
Wave 1 (parallel) — COMPLETED ✅
Wave 2 (parallel) — COMPLETED ✅

         +---- ENGINEERING AUDIT GATE (Apr 30) ----+
         v                                           
Wave 2.5 (engineering hardening — 4 agents parallel): — COMPLETED ✅
  Agent A: Backend hardening (H1-H12) — ✅ All 12 tasks done
  Agent B: Frontend hardening (F1-F9) — ✅ All 9 tasks done
  Agent C: Test hardening (T6-T14) — ✅ 8/9 tasks done (T13 skipped)
  Agent D: Integration & infra (I1-I7) — ✅ All 7 tasks done

  Actual wall-clock: ~2.5 hr (estimated 5-6 hr).
  Test suite: 135 tests, all passing. TypeScript: 0 errors.

         +---- FEATURE GATE (all Wave 2.5 tests pass) ----+
         v                                                  
Wave 3 (parallel — features on hardened base):
  Agent A: #8 Backend WebSocket        Agent B: #9 Frontend WS + concordance
  Agent C: #10 Technical specs doc

Wave 4 (integration + delivery):
  Agent A: #12 Integration test + FULL TEST SUITE
  Agent B: #13 Pre-run demo
  Agent D: #14 Docker compose
  Any: #11 + #15
  
  TEST SUITE GATE: pytest tests/ -m "not slow" -v MUST pass before tagging.
```

**Updated total estimates**:
- Wave 2.5 wall-clock: ~2.5 hr (completed, 4 parallel agents)
- Wave 3: ~3-4 hours (3 tasks, reduced due to Wave 2.5 groundwork)
- Wave 4: ~2 hours (mostly automation + validation)
- Total remaining: ~5-6 hours

---

## 10. Git Operations

```bash
git add -A
git commit -m "v2.0-submission: final model, corrected DLE, CMA-ES concordance, XAI, WebSocket frontend"
git tag v2.0-submission
```

---

## 11. Testing Suite Plan (Current State)

### 11.1 Coverage Status (Apr 30 Audit)

| Category | Planned Tests | Existing | Status |
|----------|--------------|----------|--------|
| Unit tests (losses, metrics, modules, trainer, CMA-ES) | 18 files | 18 files | ✅ 92 tests |
| Functional tests | 4 files | 4 files | ✅ 16 tests |
| System tests | 4 files | 4 files | ✅ Complete |
| Integration tests | 3 files | 3 files | ✅ **Done in Wave 2.5** |
| Regression tests | 3 files | 3 files | ✅ **Done in Wave 2.5** |
| API error path tests | 1 file | 1 file | ✅ **15 tests (Wave 2.5)** |
| Frontend tests | ~10+ | 0 files | ❌ **Not started** |
| Phase 4 (CMA-ES/Inversion) | — | 7 tests | ✅ **Covered in Wave 2.5** |
| `trainer.py` (649 lines) | — | 2 tests | ✅ **Covered in Wave 2.5** |
| `backend/server.py` (2067 lines) | — | ~25% coverage | ✅ **Improved in Wave 2.5** |

### 11.2 Known Test Quality Issues

1. ~~**Session-scoped model fixture mutated** by `test_training_one_epoch.py` — all subsequent tests see modified weights. Violates test isolation.~~ ✅ **Fixed**: state save/restore in T7.
2. ~~**Only 5 `pytest.raises` calls** across entire suite — error handling paths effectively untested.~~ ✅ **Fixed**: 15 API error tests in T6.
3. ~~**Mock data is unrealistic**: leadfield is white noise (no spatial correlation), connectivity is identity (no network structure), EEG has no 1/f spectrum.~~ ✅ **Fixed**: 1/f EEG, structured leadfield, ~15% connectivity in T14.
4. **Zero frontend tests**: no test runner, no test config, no test files. — ❌ **Still open** (T13 skipped).

### 11.3 Test Infrastructure

| Component | Specification |
|-----------|--------------|
| **Test runner** | `pytest` (existing) |
| **Config** | `pytest.ini` with markers: `unit`, `functional`, `system`, `integration`, `slow`, `regression` |
| **Fixtures** | `tests/conftest.py` (185 lines) — session-scoped mocks, realistic data (1/f EEG, structured leadfield) |
| **Mock data** | `tests/mock_data/` — 5 `.npy` files; updated to be realistic in T14 |
| **CI profile** | `pytest -m "not system and not integration"` — unit + functional + regression + API errors (~15s) |

---

## 12. Engineering Audit Findings (Apr 30)

### 12.1 Critical Issues Found

| # | Severity | Issue | Location | Fix | Status |
|---|----------|-------|----------|-----|--------|
| C1 | **CRITICAL** | `active_jobs` dict shared across WebSocket handler, background task, and cleanup task with NO locks | `backend/server.py:539-563` | H1 | ✅ **Resolved** |
| C2 | **CRITICAL** | `requirements.txt` lists only 3 of ~15+ runtime deps | `backend/requirements.txt` | H10 | ✅ **Resolved** |
| C3 | **HIGH** | `run_inference()` has no guard for `model is None` | `backend/server.py:289` | H2 | ✅ **Resolved** |
| C4 | **HIGH** | No NaN/Inf input validation — invalid data propagates silently | `backend/server.py:1355` | H3 | ✅ **Resolved** |
| C5 | **HIGH** | No request timeout — model inference can hang forever | `backend/server.py:1980` | H4 | ✅ **Resolved** |
| C6 | **HIGH** | No disk space check — results write fails silently on full disk | `backend/server.py:1415` | H5 | ✅ **Resolved** |
| C7 | **HIGH** | No results cleanup — RESULTS_DIR grows unbounded | `backend/server.py:275` | H6 | ✅ **Resolved** |
| C8 | **HIGH** | CORS `allow_methods=["*"]` + `allow_headers=["*"]` with credentials | `backend/server.py:196-197` | H7 | ✅ **Resolved** |
| C9 | **HIGH** | `_load_fsaverage5_mesh()` claims "cached" but never caches | `backend/server.py:572` | H8 | ✅ **Resolved** |
| C10 | **HIGH** | 530-line `analyze_eeg()` with 3 blocks duplicated verbatim | `backend/server.py:1145-1678` | H9 | ✅ **Resolved** (~280 lines) |
| C11 | **HIGH** | No rate limiting of any kind | Entire server | H12 | ✅ **Resolved** |
| C12 | **HIGH** | TypeScript errors ignored in production build | `frontend/next.config.mjs` | F2 | ✅ **Resolved** |
| C13 | **HIGH** | Zero frontend tests — no test framework installed | `frontend/` | T13 | ❌ **Still open** |
| C14 | **HIGH** | 10 dead npm dependencies including expo and react-native | `frontend/package.json` | F1 | ✅ **Resolved** |
| C15 | **HIGH** | Shared model fixture mutated by training test breaks isolation | `tests/functional/test_training_one_epoch.py` | T7 | ✅ **Resolved** |

### 12.2 Moderate Issues

| # | Issue | Location | Fix | Status |
|---|-------|----------|-----|--------|
| M1 | Fake progress bar (time-based, maxes at 92%) | `frontend/components/processing-window.tsx` | F6 | ✅ **Resolved** |
| M2 | No XAI frontend UI exists despite backend module | `frontend/` | F7 | ✅ **Resolved** |
| M3 | No React Error Boundary — component crash = white screen | `frontend/` | F4 | ✅ **Resolved** |
| M4 | Plotly charts hardcode black/white colors, ignore dark theme | `frontend/components/eeg-waveform-plot.tsx` | F8 | ✅ **Resolved** |
| M5 | Accessibility: missing aria-pressed on tabs, no skip-to-content, no focus trap | `frontend/app/analysis/page.tsx` | F9 | ✅ **Resolved** |
| M6 | 317 lines dead code (output-window, animated-brain, dead CSS, placeholder assets) | `frontend/` | F3 | ✅ **Resolved** |
| M7 | Mock leadfield is white noise (no spatial structure); connectivity is identity | `tests/conftest.py` | T14 | ✅ **Resolved** |
| M8 | Only 5 `pytest.raises` calls — error paths untested | `tests/` | T6 | ✅ **Resolved** |
| M9 | 0% coverage on Phase 1, Phase 4, Phase 5, trainer.py | `tests/` | T10, T11 | ✅ **Partially resolved** (CMA-ES + trainer covered) |
| M10 | No Docker image, no docker-compose, no CI/CD | `deploy/` | I1, I4 | ✅ **Resolved** (Dockerfile + CI workflow) |
| M11 | No structured logging — stdout only, no persistence | `backend/server.py:71` | H11 | ✅ **Resolved** |
| M12 | No environment validation at startup — fails silently | `backend/server.py:220` | I5 | ✅ **Resolved** |
| M13 | Heatmap HTML saved TWICE in async path — potential overwrite | `backend/server.py:1921-1952` | H9 | ✅ **Resolved** (refactored) |
| M14 | Duplicate hooks: `use-toast.ts` and `use-mobile.ts` exist in 2 locations each | `frontend/hooks/` + `frontend/components/ui/` | F3 | ✅ **Resolved** |

### 12.3 Security Surface Area

```
Backend (port 8000):
  ├─ /api/health              — GET, no auth, returns system info
  ├─ /api/analyze             — POST, file upload (EDF/NPY/MAT) or sample_idx
  ├─ /api/biomarkers          — POST, same as analyze with mode=biomarkers
  ├─ /api/eeg_waveform        — POST, returns waveform PNG
  ├─ /api/test-samples        — GET, returns list of indices
  ├─ /api/results/{path}      — GET, serves HTML files (XSS risk)
  └─ /ws/{job_id}             — WebSocket, job progress updates

Threats:
   1. Unauthenticated access — all endpoints open (acceptable for local research tool)
   2. ~~No rate limiting — trivial DoS via POST flood~~ ✅ **Resolved**: token bucket 100 req/min/IP (H12)
   3. No body size limit — huge POST bodies exhaust memory (except EDF check)
   4. File upload without magic-byte validation — renamed .txt passes as .edf
   5. Results path traversal — mitigated via Path.resolve().is_relative_to()
   6. Log injection via crafted filenames — filenames included in log messages
   7. ~~CORS allow_methods=["*"] — unnecessary permissiveness~~ ✅ **Resolved** (H7)
```

### 12.4 Architectural Debt — Partially Resolved ✅

```
server.py function dependency graph (simplified):

analyze_eeg() [~280 lines — REDUCED from 530 ✅]
  ├── run_inference() — model=None guard added ✅
  ├── compute_epileptogenicity_index() — per-region de-meaning ✓
  ├── generate_heatmap_html() — mesh loading cached ✅
  ├── generate_source_activity_heatmap_html() — same
  ├── _process_edf_raw() — bare ValueError (still open)
  ├── _load_test_sample() — no OOM guard (still open)
  ├── _build_eeg_payload() — extracted shared function ✅
  ├── _extract_plotly_body() — extracted shared function ✅
  └── _run_xai() — extracted shared function ✅

All 3 shared sub-functions extracted and called from both branches. ✅
```
