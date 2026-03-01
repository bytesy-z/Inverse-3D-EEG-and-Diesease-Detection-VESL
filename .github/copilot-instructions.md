# GitHub Copilot Instructions for PhysDeepSIF Project

## Core Principles

### 1. Documentation and Explanations
- **DO NOT** create documentation, explanations, or summaries unless explicitly requested
- **DO NOT** generate README files, docstrings for modules, or architectural overviews automatically
- Focus on code implementation only
- When asked, provide clear explanations suitable for developers unfamiliar with the codebase

### 2. Code Commenting Standards
- Write **extensive inline comments** for all code
- Every function, class, and non-trivial code block must have detailed comments explaining:
  - **What** it does in the context of the overall pipeline
  - **Why** it exists (its role in the grand scheme of PhysDeepSIF)
  - **How** it works (algorithm/approach in plain language)
  - **Input/Output** specifications with shapes and dtypes
  - **Dependencies** on other modules or data files
- Use plain language in comments—avoid unnecessary jargon
- Assume the reader has never seen this codebase before
- Comment parameter meanings, not just types
- Explain non-obvious array indexing and mathematical operations

**Example of good commenting:**
```python
def compute_dipole_localization_error(predicted_sources, true_sources, region_centers):
    """
    Calculate the Dipole Localization Error (DLE) between predicted and true source activity.
    
    DLE measures the Euclidean distance between the power-weighted centroid of the
    predicted source activity and the true epileptogenic center. This is a standard
    metric in EEG source imaging (Molins et al., 2008).
    
    Args:
        predicted_sources: ndarray (n_regions, n_timepoints)
            Estimated source activity from PhysDeepSIF network output
        true_sources: ndarray (n_regions, n_timepoints)
            Ground truth source activity from TVB simulation
        region_centers: ndarray (n_regions, 3)
            3D coordinates (x,y,z in mm, MNI space) of each region centroid
    
    Returns:
        float: DLE in millimeters. Lower is better (<20mm is target threshold)
    """
    # Step 1: Compute time-averaged power for each region
    # Power = mean of squared activity across time dimension
    # Shape: (n_regions,)
    pred_power = np.mean(predicted_sources ** 2, axis=1)  
    true_power = np.mean(true_sources ** 2, axis=1)
    
    # Step 2: Compute power-weighted centroid of predicted activity
    # This represents the "center of mass" of the predicted epileptogenic zone
    # Formula: sum(power_i * position_i) / sum(power_i)
    total_pred_power = np.sum(pred_power)  # Normalization factor
    if total_pred_power < 1e-10:  # Avoid division by zero if no activity
        pred_centroid = np.zeros(3)
    else:
        # Weighted average of region positions by their power
        # Broadcasting: (n_regions, 1) * (n_regions, 3) -> sum over regions
        pred_centroid = np.sum(pred_power[:, np.newaxis] * region_centers, axis=0) / total_pred_power
    
    # Step 3: Compute true epileptogenic centroid (same method)
    total_true_power = np.sum(true_power)
    if total_true_power < 1e-10:
        true_centroid = np.zeros(3)
    else:
        true_centroid = np.sum(true_power[:, np.newaxis] * region_centers, axis=0) / total_true_power
    
    # Step 4: Euclidean distance between the two centroids
    # This is our DLE metric - how far off was our spatial localization?
    dle = np.linalg.norm(pred_centroid - true_centroid)
    
    return dle  # Returns distance in mm
```

### 3. Coding Standards

#### PEP 8 Compliance
- Follow PEP 8 style guide strictly
- Line length: max 100 characters (not 79, to accommodate scientific code)
- Use 4 spaces for indentation (never tabs)
- Two blank lines between top-level functions/classes
- One blank line between methods in a class
- Imports: standard library → third-party → local, each group alphabetically sorted
- Naming conventions:
  - `snake_case` for functions, variables, module names
  - `PascalCase` for class names
  - `UPPER_SNAKE_CASE` for constants
  - Leading underscore `_internal_function` for private/internal use

#### Professional Practices
- **Type hints**: Use comprehensive type hints for all function signatures
  ```python
  from typing import Tuple, Optional
  import numpy as np
  from numpy.typing import NDArray
  
  def process_eeg(
      raw_eeg: NDArray[np.float32],
      sampling_rate: float = 200.0,
      filter_params: Optional[dict] = None
  ) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
  ```
- **Error handling**: Always include appropriate try-except blocks with specific exceptions
- **Input validation**: Check array shapes, dtypes, value ranges at function entry points
- **Logging**: Use Python `logging` module (not print statements) for runtime information
- **Constants**: Define magic numbers as named constants at module top
- **Docstrings**: NumPy-style docstrings for all public functions/classes (when documentation is requested)

### 4. Technical Specifications Reference

#### Primary Reference Document
- **Always** refer to `/home/zik/UniStuff/FYP/homogenized_pipeline/fyp-2.0/docs/02_TECHNICAL_SPECIFICATIONS.md` for:
  - Exact dimensional constants (76 regions, 19 channels, 200 Hz, etc.)
  - Mathematical formulas (loss functions, metrics, equations)
  - Data formats and interfaces between modules
  - Library versions and dependencies
  - Parameter values and ranges
  - File paths and naming conventions
  - Academic references and justifications

#### Important Rules
- **NEVER** edit or modify `02_TECHNICAL_SPECIFICATIONS.md` unless explicitly requested
- **NEVER** edit or modify `01_PLAIN_LANGUAGE_DESCRIPTION.md` unless explicitly requested
- If specifications are ambiguous, ask for clarification rather than assuming
- When implementing from specs, copy exact values (don't round or "simplify")
- Maintain consistency with all interface specifications in Section 8 of the technical doc

### 5. Code Quality Assurance

#### Before Submitting Code
- **Syntax**: Ensure code is syntactically correct Python
- **Logic**: Verify the implementation achieves the stated objective
- **Shapes**: Double-check all array operations for dimensional compatibility
- **Dtypes**: Ensure dtype consistency (float32 for tensors, float64 for numpy where specified)
- **Dependencies**: Confirm all imports are available in the specified environment
- **Edge cases**: Handle boundary conditions (empty arrays, zero division, etc.)
- **Memory**: Consider memory efficiency for large datasets (100k samples, 76×400 arrays)

#### Testing Mindset
- Write code that is **testable**
- Separate I/O from computation where possible
- Make functions pure (deterministic output for given input) when feasible
- Include assertions for critical invariants
- Provide clear error messages that aid debugging

### 6. Current Phase Context

**CURRENT MODE: WEB APPLICATION DEMO v3 (UNIFIED DASHBOARD) + PENDING DC OFFSET FIX**

**SITUATION:**
- ✅ **MODEL TRAINING v1 COMPLETE** (Feb 26, 34 epochs, early stopping at epoch 19)
- ✅ **MVP Demo delivered** (Feb 27) — cherry-picked samples with recall=1.0
- ✅ **DC offset root cause identified and validated via literature review**
- ✅ **De-meaning approach validated** — see Technical Specs §4.4.7
- ✅ **Web Application v3 COMPLETE** (Mar) — unified dashboard, single /analysis page, dark theme throughout, Plotly fixes
- Using `data/synthetic3/` dataset (80k training samples, generated with older code)
- Current code has significant improvements not reflected in synthetic3 data

**WEB APPLICATION (Section 9 of Technical Specs):**

The system has a clinical-grade web demo built on a **unified analysis dashboard** architecture:

**Shared UI Components:**
- `AppHeader`: Sticky top nav with VESL brand + single "Analyze EEG" tab → `/analysis` (uses `usePathname()`)
- `AppContainer`: Centered `max-w-6xl` container with consistent padding
- `StepIndicator`: Three-step workflow (Upload → Analyze → Results) with visual state tracking
- `FileUploadSection`: Drag-drop with validation, file display, clear button, keyboard accessible
- `BrainVisualization`: Plotly HTML container with skeleton loader, fullscreen toggle, resize handler, `playbackSpeed` prop
- `ResultsMeta`: Compact key-value stats row (filename, time, windows)
- `DetectedRegions`: Badge list with full anatomical names (e.g., "rAMYG (Right Amygdala)")
- `ProcessingWindow`: Timer + progress bar + pipeline step checklist
- `AppFooter`: Footer with VESL copyright + team credits (Hira Sardar, Muhammad Zikrullah Rehman, Shahliza Ahmad)

**Design System (`lib/theme.ts`):**
- Sage green accent (oklch hue ≈ 160), neutral grey base
- **Dark theme throughout** — all pages use `<div className="dark">`
- CSS tokens in `globals.css`: `:root` (light) + `.dark` (dark)
- `.plotly-container` CSS ensures proper Plotly sizing

**Landing Page (`/`):**
- Dark theme with hero section, problem & solution cards, capabilities cards, CTA → `/analysis`
- Professional clinical tone (no emoji, no "AI-y" language)
- Sections: "The Problem & Our Solution" (4 cards), "What VESL Does" (4 cards), CTA banner

**Unified Analysis Dashboard (`/analysis`):**
- Single page for both Source Localization and Biomarker Detection
- Dark theme with shared upload → parallel processing → tabbed results
- Both API calls (`/api/analyze-eeg` + `/api/physdeepsif`) run in parallel via `Promise.all`
- View mode toggle tabs: "Source Localization" (Brain icon) / "Biomarker Detection" (Activity icon)
- **Playback speed control** (0.5x, 1x, 2x, 4x) — shown only for animated ESI results
  - Dynamically adjusts Plotly frame duration via `Plotly.relayout()` on `BrainVisualization`
- ESI view: inferno colorscale, dark canvas, animation hint for multi-window
- Biomarkers view: warm gradient, dark canvas, DetectedRegions badges (clinical red, top-K=5)
- Legacy routes `/eeg-source-localization` and `/biomarkers` → 307 redirect to `/analysis`

**Architecture:**
- FastAPI backend (port 8000) — loads PhysDeepSIF model on CUDA, generates Plotly HTML
- Next.js 15 frontend (port 3000) — proxies `/api/*` to backend, renders Plotly HTML in container div
- `src/region_names.py` — maps 76 DK region codes to full anatomical names
- `start.sh` manages both servers (start/stop/kill)

**Key Implementation Details — Plotly Figures:**
- `auto_play=False` — prevents animation autoplay on load
- `autosize=True` + `config=dict(responsive=True)` — figures resize to fit container
- **No camera view buttons** — Left/Right/Front/Back/Top buttons removed from all figures
- **Styled play/pause** — `bgcolor='rgba(40,40,60,0.85)'`, border, consistent font
- **m:ss timestamp format** — slider labels use `M:SS.s` format (e.g., "0:01.0", "1:30.5")
- **Both modes dark canvas** — `#1a1a2e` background with `#e0e0e0` text
- Bottom margin `b=60` on animated figures ensures slider visibility
- No custom React play/pause — Plotly native controls + React speed control via `Plotly.relayout()`
- Region name mapping returns both short codes and full names in API responses (`*_full` fields)

**CRITICAL FINDING: Epileptor LFP Proxy DC Offset Problem**

The Epileptor x2-x1 LFP proxy has a large DC offset that varies with x0 (excitability):
- Healthy regions (x0 ∈ [-2.2, -2.05]): mean(x2-x1) ≈ 1.80
- Epileptogenic regions (x0 ∈ [-1.8, -1.2]): mean(x2-x1) ≈ 1.60
- **DC offset dominates**: power = mean² + var, and mean² accounts for 98.1% of total power

This causes two cascading problems:
1. **Source normalization masks the useful signal**: Global normalization (src_mean=1.792, src_std=0.258)
   shifts everything toward zero, but the *dynamics* (variance ~0.05) are only 1.9% of total power.
   The model learns to output a near-constant DC value (spatial CV = 0.002) with effectively zero
   temporal dynamics (pred temporal var ≈ 0.00002 vs true ≈ 0.05).

2. **Power-based epileptogenicity scoring inverts**: Because epileptogenic regions have LOWER mean
   signal (lower DC offset due to x0), time-averaged power incorrectly ranks them as *less* active.
   However, variance and ptp are 3.9x and 1.86x HIGHER in epileptogenic regions respectively.
   The useful discriminatory signal is in the AC component, not the DC component.

**Evidence (from 50 epileptogenic test samples):**
| Feature | Epileptogenic | Healthy | Ratio | Oracle top-10 recall |
|---------|-------------|---------|-------|---------------------|
| Power (mean s²) | 2.74 | 3.29 | 0.84 | 0.023 (inverted!) |
| Variance | 0.178 | 0.045 | 3.92 | 0.886 ✓ |
| Range (ptp) | 1.94 | 1.04 | 1.86 | 0.771 ✓ |
| Kurtosis_inv | — | — | — | 0.913 ✓ |

**This is NOT a bug** — it is physically correct Epileptor behavior (Jirsa et al., 2014):
- Near-critical x0 values → intermittent bursts with quiescent baseline → high variance
- The DC offset of x2-x1 shifts with x0 as the system's equilibrium point changes
- Clinical parallel: focal epilepsy shows background suppression between interictal discharges

**VALIDATED SOLUTION: Per-Region Temporal De-Meaning**

Literature review confirms this is the correct approach:
- **Yu et al. 2024 (MS-ESI, NeuroImage)**: Uses Wendling model generating interictal spikes — inherently AC, no DC issue
- **DeepSIF (Sun et al., 2022, PNAS)**: Also spike-based training data, no DC issue
- **Virtual Brain Twins (Hashemi et al., 2025)**: Uses same x0 range, confirms DC shifts are standard Epileptor behavior
- **Real EEG is AC-coupled**: All clinical EEG amplifiers use highpass ≥ 0.1 Hz; model cannot learn DC from input
- **Physics losses unaffected**: Laplacian/temporal smoothness operate on gradients, invariant to mean removal
- **CMA-ES compatible**: Welch PSD is inherently DC-free

Implementation: Subtract per-region temporal mean from source_activity AND per-channel temporal
mean from EEG in Dataset.__iter__() before global z-score normalization.

**CURRENT PHASE: Phase 2.5 — DC Offset Fix and Model Retraining**

**COMPLETION STATUS:**
- ✅ **PHASE 1 COMPLETE** — Forward Modeling and Synthetic Data Generation
  - ✅ 1.1: Source Space Definition (76 Desikan-Killiany regions)
  - ✅ 1.2: Neural Mass Model Configuration (Epileptor, 6 parameters per region)
  - ✅ 1.3: Leadfield Matrix Construction (BEM, 19×76, linked-ear reference)
  - ✅ 1.4: Synthetic Dataset Generation (100k samples, 3 splits)
    - Using `data/synthetic3/`: Train 79,995 | Val 10,010 | Test 9,980
    - **NOTE**: Generated with older code; current code includes spectral shaping improvements
  - ✅ 1.5: Biophysical Validation (13/13 metrics PASS)
  - ✅ 1.6: Integrated Spectral Shaping (STFT-based alpha/beta processing)

- ✅ **PHASE 2 COMPLETE** — Network Architecture and Training (v1)
  - ✅ 2.1: PhysDeepSIF Network Architecture (410,244 params)
  - ✅ 2.2: Physics-Informed Loss Functions (source MSE + forward + physics)
  - ✅ 2.3: Training Loop with early stopping
  - ✅ 2.4: Model Training v1 on synthetic3
    - **Best checkpoint:** Epoch 19, val_loss=1.0141
    - **Metrics:** DLE=10.41mm, SD=53.75mm, AUC=0.486, Corr=0.072
    - **Known issue:** Model outputs ~100% DC offset, near-zero temporal dynamics
    - Model learns average DC level well but fails to reconstruct per-region variance

- ✅ **MVP DEMO MODULE** — Biomarker Detection Prototype
  - ✅ End-to-end inference pipeline working
  - ✅ Inverted-range scoring (range_inv) — best post-processing feature found
    - 0.258 top-10 recall (2x chance) for model predictions
    - Perfect recall (1.0) on ~15% of epileptogenic patterns (left cingulate-insular network)
  - ✅ 3D brain visualization with dark=epileptogenic colorscale
  - ✅ Cherry-picked demo samples: indices 10, 25, 51 (recall=1.0)
  - ✅ Support for specifying multiple --sample-idx values
  - ✅ Threshold: 87.5th percentile (top ~10 regions)
  - ✅ Top-K recall metrics in output JSON

- ✅ **WEB APPLICATION v3** — Unified Dashboard + Plotly Fixes (Mar)
  - ✅ FastAPI backend (serves model inference + Plotly HTML with `auto_play=False`, `autosize=True`, `responsive=True`)
  - ✅ `src/region_names.py` — 76-entry DK region code → full anatomical name mapping
  - ✅ Next.js 15 frontend with clinical-grade design system
  - ✅ **Unified analysis dashboard** (`/analysis`) — single upload, parallel processing, tabbed results
  - ✅ Landing page with problem statement, benefits, CTA (professional clinical tone)
  - ✅ Single nav tab "Analyze EEG" → `/analysis` (replaced two separate tabs)
  - ✅ Legacy routes `/eeg-source-localization` and `/biomarkers` → 307 redirect to `/analysis`
  - ✅ **Dark theme throughout** — both ESI and biomarkers on dark canvas (`#1a1a2e`)
  - ✅ **Camera view buttons removed** — no Left/Right/Front/Back/Top in any Plotly figure
  - ✅ **Styled play/pause buttons** — consistent dark background, border, font
  - ✅ **m:ss timestamp format** — animation slider uses M:SS.s (e.g., "0:01.0", "1:30.5")
  - ✅ **Playback speed control** — 0.5x, 1x, 2x, 4x buttons, adjusts Plotly frame duration via `Plotly.relayout()`
  - ✅ **Responsive Plotly figures** — `autosize=True`, no fixed width/height, `config=dict(responsive=True)`
  - ✅ **Improved fullscreen** — dynamic height, Plotly resize after toggle
  - ✅ Parallel API calls via `Promise.all` for both ESI and biomarkers
  - ✅ AppFooter with team credits (Hira Sardar, Muhammad Zikrullah Rehman, Shahliza Ahmad)
  - ✅ TypeScript 0 errors, all routes responding correctly

**IMMEDIATE PRIORITIES (ordered):**
1. ✅ **Literature review**: De-meaning validated — real EEG is AC-coupled, MS-ESI/DeepSIF use
   spike-based (AC) training data, physics losses are gradient-based (mean-invariant), CMA-ES
   uses PSD (DC-free). Full analysis in Technical Specs §4.4.7.

2. **Implement per-region de-meaning in training pipeline**:
   - Modify `HDF5Dataset.__iter__()` in `scripts/03_train_network.py`
   - Subtract per-region temporal mean: `sources -= sources.mean(dim=-1, keepdim=True)`
   - Also de-mean EEG per channel for AC-coupling consistency
   - Apply **before** global z-score normalization
   - Update in-memory path (`normalize_data()`) similarly

3. **Quick validation training** (~5-10 epochs on synthetic3):
   - Verify AUC improves from 0.486 baseline
   - Verify temporal correlation improves from 0.072
   - Verify DLE stays stable (<20 mm)
   - If confirmed, proceed with full retraining

4. **Full retraining with de-meaned data** (if quick test passes)

**POST-DEMO TODOs (for after March 3, 2026):**

These improvements have been identified but deferred due to demo deadline constraints.
Bring these up when work resumes on March 3, 2026.

1. **Regenerate synthetic data with improved code**:
   - Current `data/synthetic3/` was generated with older code lacking spectral shaping
   - Regenerate using current `src/phase1_forward/synthetic_dataset.py` with integrated
     spectral shaping (STFT-based alpha/beta) and skull attenuation filter
   - Apply per-region de-meaning as a permanent preprocessing step in the Dataset class
   - Expected benefit: better spectral realism → better generalization to NMT real EEG

2. **Explore variance-based loss terms**:
   - Add explicit variance-matching loss: $\mathcal{L}_{var} = \sum_i (\text{Var}(\hat{S}_i) - \text{Var}(S_i))^2$
   - This directly optimizes the discriminative signal (epileptogenic regions have 3.9× higher variance)
   - Could complement MSE loss on de-meaned data for stronger dynamics learning

3. **Per-region normalization** (alternative to global z-score):
   - Instead of global stats, normalize each region by its own temporal statistics
   - Would equalize the learning signal across regions with different activity levels
   - Risk: may destroy spatial amplitude relationships needed for forward consistency
   - Test against global z-score to compare

4. **x0 gap sampling investigation**:
   - Currently x0 has a gap at [-2.05, -1.8] — never sampled
   - This creates a bimodal distribution. Unclear if this helps or hurts the model
   - Experiment with continuous x0 sampling to fill the gap
   - May improve model's ability to detect borderline epileptogenic regions

5. **Forward loss normalization**:
   - The forward loss $\|\mathbf{L}\hat{S} - \text{EEG}\|^2$ is ~41,456× larger than source loss
     at correct scale (see §4.4.5)
   - Even with reduced β, this scale mismatch may cause optimization issues
   - Options: normalize forward loss by EEG variance, use relative error,
     or scale leadfield to unit norm columns before computing forward loss

6. **Hyperparameter optimization** (Bayesian/TPE via Optuna):
   - Search space defined in Tech Specs §4.5.2
   - Should be run AFTER de-meaning fix is confirmed working
   - Optimize for temporal correlation as primary objective
   - 50-100 trials, 30-45 min each

7. **Classical baseline comparisons**:
   - Implement eLORETA, MNE, dSPM, LCMV for the same test set
   - Use same leadfield matrix for fair comparison
   - Required for thesis/paper to contextualize PhysDeepSIF performance

8. **Phase 3: NMT preprocessing pipeline**:
   - Load and preprocess real EEG from NMT dataset
   - Linked-ear reference (keep as-is), 0.5-70 Hz bandpass, 50 Hz notch, ICA

9. **Phase 4: CMA-ES parameter inversion**:
   - Fit x0 per region by matching simulated and real EEG PSDs
   - Requires Phase 3 output (preprocessed NMT segments)

10. **Comprehensive validation suite** (Tech Specs §7):
    - Full DLE/SD/AUC/correlation across noise levels
    - Patient consistency analysis
    - Normal vs abnormal NMT discrimination

---

## Module-Specific Guidelines

### Phase 1: Forward Modeling (`src/phase1_forward/`)

#### `source_space.py`
- Purpose: Load and validate TVB connectivity and parcellation data
- Key functions must handle TVB's data structures and convert to standard numpy
- Must validate region count (exactly 76), symmetry of connectivity, coordinate ranges

#### `epileptor_simulator.py`
- Purpose: Wrapper around TVB's Epileptor model for batch simulation
- Must handle parameter randomization as per Section 3.4.1
- Must extract correct output variable (x2 - x1 LFP proxy)
- Simulation length: 12 seconds, discard first 2 seconds transient
- Integration: dt=0.1 ms, HeunStochastic, Raw monitor (NOT TemporalAverage)
- Anti-aliased decimation: scipy.signal.decimate(ftype='fir'), 2-stage (10×10)
- TVB 2× rate factor: dt=0.1ms → 20,000 Hz actual output → decimate to 200 Hz
- Noise structure: [D,D,0,D,D,0] — no noise on slow variables z and g
- Parameter mapping: tau0 → model.r = 1/tau0, tau2 → model.tau, model.tt stays at 1.0

#### `leadfield_builder.py`
- Purpose: Construct 19×76 leadfield using MNE-Python BEM
- Must implement Approach B (vertex-level forward, then parcel averaging)
- Must apply linked-ear re-referencing transformation
- Must validate rank = 18 (Section 3.3.1, Step 4)

#### `synthetic_dataset.py`
- Purpose: Orchestrate TVB simulation + leadfield projection + noise addition + spectral shaping
- Must generate HDF5 files with exact schema from Section 3.4.4
- Must implement parallel simulation using joblib
- Must apply normalization only in PyTorch Dataset, not in HDF5 storage
- NaN/Inf detection: check source_activity after run_simulation(); discard diverged sims (~5-10%)
- Forward projection: EEG = leadfield (19×76) @ source_activity (76×T)
- Noise model: white noise (target SNR in dB) + colored noise (fraction of signal RMS)
- Colored noise: lowpass-filtered Gaussian, separate spatial correlation via leadfield
- Healthy-only samples (k=0): ~11% of dataset, all x0 values set to -2.2 (normal range)
- Simulation yields 10 seconds at 200 Hz = 2000 time points, segmented into 5 windows of 400
- Skull attenuation: 4th-order Butterworth LP @ 40 Hz applied after noise addition
- Spectral shaping: STFT-based delta/theta suppression + adaptive alpha redistribution + beta gradient gains
- Group-level validation: alpha/beta gradients checked on 5 anteroposterior groups, PDR [1.3, 5.0]

### Phase 1 Post-Processing Scripts (`scripts/`)

#### `scripts/07_biophysical_validation.py`
- Purpose: Validate biophysical properties of synthetic EEG (13 metrics)
- Checks: amplitude range, spectral content, temporal dynamics, noise levels
- Must PASS 13/13 metrics for data to be considered valid

#### `scripts/08_apply_spatial_gradients.py` — **DEPRECATED**
- Previously: Post-processing script for applying spatial-spectral gains to existing HDF5 data
- Now: Spectral shaping is integrated directly into `synthetic_dataset.py` during generation
- Script kept for reference only; no longer needed in the data generation pipeline

#### `scripts/09_validate_enhanced_data.py`
- Purpose: Comprehensive validation of generated data against legacy + spatial-spectral metrics
- Key thresholds: PDR [1.3, 5.0], group-level monotonic gradient, RMS [5-150 µV], gamma < 15%
- Uses Welch PSD: 200-sample Hann, 50% overlap (matching generation STFT)
- Channel groups: frontal_fp (Fp1,Fp2), frontal_f (F3,F4,F7,F8,Fz), central_c (C3,C4,T3,T4,Cz), parietal_p (P3,P4,T5,T6,Pz), occipital_o (O1,O2)
- Validates 100 random samples per split, outputs report to `outputs/validation_report.json`

### Phase 2: Network (`src/phase2_network/`)

#### `physdeepsif.py`
- Purpose: Main network architecture combining spatial and temporal modules
- Must match architecture diagram in Section 4.1.1
- Must register leadfield and connectivity Laplacian as non-trainable buffers
- Total parameters should be ~355,000

#### `loss_functions.py`
- Purpose: Implement composite physics-informed loss
- **CRITICAL**: Use exact formulas from Section 4.2 (do not simplify or approximate)
- Must implement all three sub-losses: source, forward, physics
- Physics loss has three components: Laplacian, temporal, amplitude
- Must handle the graph Laplacian construction from connectivity matrix

#### `trainer.py`
- Purpose: Training loop with monitoring and early stopping
- Must implement all data augmentation from Section 4.3.2
- Must compute all validation metrics from Section 4.3.3 (DLE, SD, AUC, correlation)
- Must save checkpoints and training logs to `outputs/models/`

### Phase 3: Inference (`src/phase3_inference/`)

#### `nmt_preprocessor.py`
- Purpose: Load and preprocess NMT EDF files
- Must keep linked-ear reference (do NOT re-reference to average)
- Must apply exact filtering: 0.5-70 Hz bandpass, 50 Hz notch
- Must implement ICA artifact removal with fastica method
- Output: (n_epochs, 19, 400) float32 z-scored arrays

### Phase 4: Inversion (`src/phase4_inversion/`)

#### `objective_function.py`
- Purpose: Compute J(x0) for CMA-ES optimization
- Must implement exact formulas from Section 6.1
- Must use Welch's method for PSD computation (1-second Hanning windows, 50% overlap)
- Must run TVB simulation inside objective (this makes it non-differentiable)

#### `cmaes_optimizer.py`
- Purpose: Wrapper for CMA-ES with TVB simulation in the loop
- Must use configuration from Section 6.2 (population=50, sigma=0.3, max_gen=200)
- Must parallelize objective evaluations across population members
- Must save optimization trajectory for analysis

### Phase 5: Validation (`src/phase5_validation/`)

#### `synthetic_metrics.py`
- Purpose: Compute all metrics from Section 7.1.1
- Must implement DLE and SD exactly as defined (Molins et al., 2008 formulas)
- Must compute metrics across noise levels (5, 10, 15, 20, 30 dB)

#### `classical_baselines.py`
- Purpose: Run eLORETA, MNE, dSPM, LCMV for comparison
- Must adapt vertex-level solutions to 76-region parcellation consistently
- Must use same leadfield matrix as PhysDeepSIF for fair comparison

---

## Code Organization Principles

### Imports Organization
```python
# Standard library imports (alphabetical)
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Third-party imports (alphabetical by package)
import h5py
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import roc_auc_score

# TVB imports (grouped)
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.simulator import Simulator

# MNE imports (grouped)
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator

# Local imports (relative, by proximity)
from ..phase1_forward.source_space import load_connectivity
from .spatial_module import SpatialModule
```

### File Header Template
```python
"""
Module: [module_name].py
Phase: [1-5] - [Phase Name]
Purpose: [One-line description of module's role in the pipeline]

This module is part of the PhysDeepSIF pipeline for epileptogenicity mapping.
See /docs/02_TECHNICAL_SPECIFICATIONS.md Section [X.Y] for full specifications.

Key dependencies:
- [Dependency 1]: [Why it's needed]
- [Dependency 2]: [Why it's needed]

Input data format:
- [Data object 1]: [shape, dtype, source]

Output data format:
- [Data object 1]: [shape, dtype, destination]

Author: [Generated by GitHub Copilot]
Date: [Auto-generated]
"""

# Imports here
# ...

# Module-level constants (from technical specs)
N_REGIONS = 76  # Desikan-Killiany parcellation
N_CHANNELS = 19  # Standard 10-20 montage
SAMPLING_RATE = 200.0  # Hz, matched to NMT dataset

# Configure module-level logger
logger = logging.getLogger(__name__)
```

### Function Template
```python
def function_name(
    param1: NDArray[np.float64],
    param2: int,
    optional_param: Optional[str] = None
) -> Tuple[NDArray[np.float32], bool]:
    """
    [One-line summary of what this function does]
    
    [Detailed explanation of the function's role in the pipeline,
    including which phase/subtask it belongs to and how it fits
    into the larger system.]
    
    Algorithm:
        1. [Step 1 description]
        2. [Step 2 description]
        ...
    
    Args:
        param1: [Description including what it represents in the pipeline,
                expected shape, and any constraints]
        param2: [Description]
        optional_param: [Description]. Defaults to [value].
    
    Returns:
        Tuple containing:
        - [First element]: [Description with shape and dtype]
        - [Second element]: [Description]
    
    Raises:
        ValueError: If [condition]
        RuntimeError: If [condition]
    
    References:
        Technical Specs Section [X.Y.Z]
        [Academic reference if applicable]
    
    Example:
        >>> [Usage example if helpful]
    """
    # Step 1: Validate inputs
    # [Comment explaining why this validation is necessary]
    if param1.shape[0] != N_REGIONS:
        raise ValueError(
            f"Expected param1 to have {N_REGIONS} regions, got {param1.shape[0]}. "
            f"This function operates on Desikan-Killiany 76-region parcellation."
        )
    
    # Step 2: [Main computation]
    # [Detailed comment explaining the operation in plain language]
    # [Reference to mathematical formula if applicable]
    result = ...  # [Inline comment about this specific line]
    
    # Step 3: [Validation of output]
    # [Comment explaining why this check is important]
    success = ...
    
    logger.debug(f"Function {function_name.__name__} completed: success={success}")
    
    return result, success
```

---

## Configuration Management

### YAML Configuration Loading
- Always load configuration from `config.yaml` in the project root
- Use the exact structure defined in Section 8.2 of technical specs
- Validate all required keys are present before proceeding
- Use default values from technical specs when optional keys are missing
- Never hardcode values that are in the config file

### Example Configuration Loader
```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """
    Load and validate the project configuration file.
    
    The configuration file controls all hyperparameters, file paths, and
    settings across the entire PhysDeepSIF pipeline. This centralizes
    configuration management and ensures consistency.
    
    Args:
        config_path: Path to config.yaml. Defaults to project root.
    
    Returns:
        Dict containing all configuration settings, organized by phase
    
    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        ValueError: If required keys are missing
    
    References:
        Technical Specs Section 8.2 (Configuration File Format)
    """
    # Load the YAML file
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Expected config.yaml in project root."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required top-level keys
    required_keys = [
        'source_space', 'neural_mass_model', 'forward_model',
        'synthetic_data', 'network', 'training', 'preprocessing',
        'parameter_inversion', 'heatmap'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Configuration file missing required keys: {missing_keys}"
        )
    
    logger.info(f"Loaded configuration from {config_path}")
    return config
```

---

## Data Interface Compliance

### Array Shape Validation
Every function that receives or produces numpy arrays or torch tensors must validate shapes:

```python
def validate_source_activity(source_activity: NDArray[np.float32]) -> None:
    """
    Validate that source activity array has correct shape and dtype.
    
    Source activity is the fundamental data type in this pipeline, representing
    time-series neural activity across the 76 brain regions.
    
    Args:
        source_activity: Array to validate
    
    Raises:
        ValueError: If shape or dtype is incorrect
    
    References:
        Technical Specs Section 8.1 (Interface Specification Table)
    """
    # Expected shape: (76 regions, variable time points)
    if source_activity.ndim != 2:
        raise ValueError(
            f"Source activity must be 2D (regions × time), got {source_activity.ndim}D"
        )
    
    if source_activity.shape[0] != N_REGIONS:
        raise ValueError(
            f"First dimension must be {N_REGIONS} regions, got {source_activity.shape[0]}"
        )
    
    # Check dtype matches specification
    if source_activity.dtype != np.float32 and source_activity.dtype != np.float64:
        raise ValueError(
            f"Source activity must be float32 or float64, got {source_activity.dtype}"
        )
    
    logger.debug(
        f"Validated source activity: shape {source_activity.shape}, "
        f"dtype {source_activity.dtype}"
    )
```

### HDF5 I/O Compliance
When reading/writing HDF5 files, strictly follow the schema in Section 3.4.4:

```python
def save_synthetic_dataset(
    output_path: Path,
    eeg_data: NDArray[np.float32],
    source_data: NDArray[np.float32],
    epileptogenic_mask: NDArray[np.bool_],
    x0_vector: NDArray[np.float32],
    snr_db: NDArray[np.float32],
    global_coupling: NDArray[np.float32],
    channel_names: List[str],
    region_names: List[str]
) -> None:
    """
    Save synthetic dataset to HDF5 file with standardized schema.
    
    This function creates the exact HDF5 structure specified in the
    technical documentation (Section 3.4.4). All training, validation,
    and test datasets must use this identical format.
    
    [Rest of docstring...]
    """
    # Validate all input shapes match expected format
    n_samples = eeg_data.shape[0]
    
    assert eeg_data.shape == (n_samples, N_CHANNELS, WINDOW_LENGTH)
    assert source_data.shape == (n_samples, N_REGIONS, WINDOW_LENGTH)
    assert epileptogenic_mask.shape == (n_samples, N_REGIONS)
    # ... validate all inputs
    
    # Create HDF5 file with exact schema from specs
    with h5py.File(output_path, 'w') as f:
        # Store main datasets (exact names from spec)
        f.create_dataset('eeg', data=eeg_data, dtype=np.float32, compression='gzip')
        f.create_dataset('source_activity', data=source_data, dtype=np.float32, compression='gzip')
        f.create_dataset('epileptogenic_mask', data=epileptogenic_mask, dtype=bool, compression='gzip')
        # ... create all datasets
        
        # Create metadata group
        metadata = f.create_group('metadata')
        metadata.create_dataset('channel_names', data=np.array(channel_names, dtype='S'))
        # ... store all metadata
        
    logger.info(f"Saved {n_samples} samples to {output_path}")
```

---

## Error Handling Standards

### Informative Error Messages
```python
# BAD: Vague error
if x.shape[0] != 76:
    raise ValueError("Wrong shape")

# GOOD: Informative error
if x.shape[0] != 76:
    raise ValueError(
        f"Source array must have {N_REGIONS} regions (Desikan-Killiany parcellation), "
        f"but got {x.shape[0]}. Check that you're using the correct parcellation. "
        f"See Technical Specs Section 3.1.1 for parcellation details."
    )
```

### Try-Except Best Practices
```python
try:
    # Attempt operation
    result = risky_operation(data)
except SpecificException as e:
    # Catch specific exceptions, add context
    logger.error(f"Failed during risky_operation: {e}")
    raise RuntimeError(
        f"Could not complete [operation] due to: {e}. "
        f"This likely means [specific cause]. "
        f"Check [what to check]."
    ) from e
```

---

## Logging Standards

### Logger Setup (per module)
```python
import logging

# Module-level logger
logger = logging.getLogger(__name__)

# In main script, configure root logger
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging for the entire application."""
    
    # Create formatter with timestamps and module names
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler (optional)
    handlers = [console_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers)
```

### Logging Usage
```python
# Log at appropriate levels
logger.debug("Detailed info for debugging: variable_value={value}")
logger.info("Key milestone: Completed [step], processed {n} samples")
logger.warning("Non-critical issue: [issue description], continuing...")
logger.error("Recoverable error: [error], attempting fallback")
logger.critical("Fatal error: [error], cannot continue")
```

---

## Performance Considerations

### Memory Management
- Use generators for large datasets when possible
- Load HDF5 datasets with chunking for memory efficiency
- Clear GPU memory explicitly after large operations: `torch.cuda.empty_cache()`
- Use `float32` for neural networks (GPU efficiency), `float64` only when numerical precision is critical

### Parallelization
- Use `joblib.Parallel` for CPU-bound tasks (TVB simulations)
- Use `torch.DataLoader` with `num_workers > 0` for data loading
- Parallelize CMA-ES objective evaluations across population members
- Log parallelization settings for reproducibility

---

## Summary Checklist for Every Code Contribution

Before submitting code, verify:

- [ ] Extensive inline comments explaining the "why" and "how" in plain language
- [ ] PEP 8 compliant (run `flake8` or `black` formatter)
- [ ] Type hints on all function signatures
- [ ] Input validation with informative error messages
- [ ] Shape and dtype assertions for all array operations
- [ ] Exact values from technical specifications (no rounding/simplification)
- [ ] Logging statements at appropriate levels
- [ ] No print() statements (use logger instead)
- [ ] Follows interface specifications from Section 8.1
- [ ] Constants defined at module level (no magic numbers)
- [ ] Code is syntactically correct and runs without errors
- [ ] Achieves the stated objective from current subtask
- [ ] No modifications to technical specifications documents
- [ ] No unnecessary documentation files generated
- [ ] Always activate the deepsif conda environment when running code on the tukl pc

---

**END OF COPILOT INSTRUCTIONS**

*This file should be referenced by GitHub Copilot for all code generation in the PhysDeepSIF project. Update the "Current Phase Context" section as the project progresses through phases and subtasks.*
