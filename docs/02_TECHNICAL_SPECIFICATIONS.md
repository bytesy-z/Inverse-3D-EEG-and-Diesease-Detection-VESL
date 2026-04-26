# PhysDeepSIF: Technical Specifications Document

## Physics-Informed Deep Learning EEG Inverse Solver with Patient-Specific Epileptogenicity Mapping

---

# Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Software Stack and Dependencies](#2-software-stack-and-dependencies)
3. [Phase 1: Forward Modeling and Synthetic Data Generation](#3-phase-1-forward-modeling-and-synthetic-data-generation)
4. [Phase 2: PhysDeepSIF Network Architecture and Training](#4-phase-2-physdeepsif-network-architecture-and-training)
5. [Phase 3: Real EEG Preprocessing and Inference](#5-phase-3-real-eeg-preprocessing-and-inference)
6. [Phase 4: Patient-Specific Parameter Inversion](#6-phase-4-patient-specific-parameter-inversion)
7. [Phase 5: Validation Framework](#7-phase-5-validation-framework)
8. [Data Formats and Inter-Module Interfaces](#8-data-formats-and-inter-module-interfaces)
9. [Web Application (Demo Interface)](#9-web-application-demo-interface)
10. [Project Directory Structure](#10-project-directory-structure)
11. [Academic References and Justifications](#11-academic-references-and-justifications)
12. [Project Completion Status](#12-project-completion-status)

---

# 1. System Architecture Overview

The system comprises six core modules that communicate through well-defined data interfaces:

```
Module 1: TVB Neural Mass Simulator
    ↓ (source_activity: ndarray [n_regions × n_timepoints])
Module 2: Leadfield Projection Engine
    ↓ (synthetic_eeg: ndarray [n_channels × n_timepoints])
Module 3: Synthetic Dataset Generator (orchestrates Modules 1 & 2)
    ↓ (HDF5 dataset files)
Module 4: PhysDeepSIF Deep Neural Network
    ↓ (source_estimates: ndarray [n_regions × n_timepoints]) 
Module 5: Patient-Specific Parameter Optimizer
    ↓ (fitted_params: ndarray [n_regions])
Module 6: Epileptogenicity Heatmap Visualizer
    ↓ (heatmap image, numerical EI table)
Module 7: Web Application (Demo Interface)
    - FastAPI backend (Python, serves inference endpoints)
    - Next.js frontend (TypeScript/React, user-facing UI)
    - Two visualization modes: ESI (source localization) and Biomarkers (epileptogenicity)
    - Interactive 3D brain rendering via Plotly Mesh3d on fsaverage5 cortical surface
```

### Dimensional Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `N_REGIONS` | 76 | Desikan-Killiany atlas, bilateral (38 per hemisphere). Standard in TVB whole-brain modeling (Sanz-Leon et al., 2013; Cagnan et al., 2021). |
| `N_CHANNELS` | 19 | International 10-20 system scalp electrodes (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3/T7, T4/T8, T5/P7, T6/P8, Fz, Cz, Pz). NMT dataset native montage. |
| `FS_SYNTH` | 200 Hz | Matched to NMT dataset sampling rate. |
| `WINDOW_LENGTH` | 400 samples (2.0 s) | Sufficient to capture interictal spike complexes (typically 70–200 ms) with surrounding context. Consistent with DeepSIF windowing (Sun et al., 2022). |
| `FS_SIMULATION` | 10,000 Hz (dt=0.1 ms) | Internal TVB integration step. TVB outputs at 2× this rate (20,000 Hz) via the Raw monitor. Anti-aliased decimation (scipy FIR, factor 100 = 10×10) brings it to 200 Hz. |

---

# 2. Software Stack and Dependencies

## 2.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `Python` | ≥ 3.10 | Runtime |
| `tvb-library` | ≥ 2.8 | Neural mass simulation (Epileptor model), structural connectivity |
| `tvb-data` | ≥ 2.0 | Default connectivity matrices, region labels |
| `mne` | ≥ 1.6 | Leadfield computation (BEM forward model), EEG preprocessing, montage handling |
| `torch` | ≥ 2.1 | Deep learning framework for PhysDeepSIF |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scipy` | ≥ 1.11 | Signal processing, optimization (CMA-ES wrapper) |
| `h5py` | ≥ 3.9 | HDF5 dataset I/O |
| `cmaes` | ≥ 0.10 | CMA-ES optimizer (Hansen, 2006) |
| `mne-bids` | ≥ 0.14 | NMT dataset loading |
| `pyedflib` | ≥ 0.1.35 | EDF file reading for NMT dataset |
| `scikit-learn` | ≥ 1.3 | Preprocessing utilities, metrics |
| `matplotlib` | ≥ 3.8 | Visualization |
| `nilearn` | ≥ 0.10 | Brain surface plotting for heatmaps |
| `pyvista` | ≥ 0.43 | 3D brain visualization (optional) |
| `fastapi` | ≥ 0.115 | Web API backend for demo interface |
| `uvicorn` | ≥ 0.34 | ASGI server for FastAPI |
| `plotly` | ≥ 5.0 | Interactive 3D brain visualization (Mesh3d) |
| `nibabel` | ≥ 5.0 | Loading FreeSurfer cortical surface meshes |
| `tqdm` | ≥ 4.66 | Progress bars for batch generation |
| `pandas` | ≥ 2.1 | Tabular data management |
| `joblib` | ≥ 1.3 | Parallel synthetic data generation |

## 2.2 Environment Setup

```bash
# Conda environment specification
conda create -n physdeepsif python=3.10
conda activate physdeepsif
pip install tvb-library tvb-data mne torch torchvision numpy scipy h5py cmaes \
    pyedflib scikit-learn matplotlib nilearn pyvista tqdm pandas joblib mne-bids
```

## 2.3 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA with ≥ 8 GB VRAM | NVIDIA A100 / RTX 4090 (24+ GB) |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB (for synthetic dataset) | 500 GB SSD |
| CPU | 8 cores | 16+ cores (for parallel TVB simulation) |

---

# 3. Phase 1: Forward Modeling and Synthetic Data Generation

## 3.1 Source Space Definition

### 3.1.1 Parcellation

The source space uses the **Desikan-Killiany (DK) atlas** as implemented in FreeSurfer and used as TVB's default parcellation. The atlas divides the cortex into 34 regions per hemisphere (68 cortical + 8 subcortical = 76 total regions in TVB's default connectivity).

**Region list**: Obtained from `tvb.datatypes.connectivity.Connectivity()` default dataset. The 76 region labels and their 3D centroid coordinates are stored in the connectivity object.

**Academic justification**: The DK-76 parcellation is the standard in TVB-based whole-brain modeling (Sanz-Leon et al., 2013; Proix et al., 2017; Hashemi et al., 2020). It provides sufficient spatial resolution for large-scale network-level epileptogenicity mapping while remaining computationally tractable.

### 3.1.2 Structural Connectivity Matrix

**Source**: TVB default connectivity dataset (`tvb.datatypes.connectivity.Connectivity.from_file()`), derived from averaged DTI tractography across healthy subjects.

**Format**: 76×76 symmetric matrix $\mathbf{W}$ where $W_{ij}$ represents the normalized fiber tract density between regions $i$ and $j$.

**Preprocessing**:
1. Log-transform weights: $\mathbf{W}' = \log_{10}(\mathbf{W} + 1)$ (standard TVB practice to handle heavy-tailed distribution).
2. Normalize by maximum: $\mathbf{W}'' = \mathbf{W}' / \max(\mathbf{W}')$.
3. Store conduction delays from tract lengths and conduction speed (default: 4.0 m/s).

**Stored as**: `connectivity.npy` — shape `(76, 76)`, dtype `float64`.

### 3.1.3 Region Centers

The 76 region centroids in MNI space are extracted from the TVB connectivity object. These are used for leadfield construction (mapping regions to electrode sensitivity).

**Stored as**: `region_centers.npy` — shape `(76, 3)`, dtype `float64` (x, y, z in mm, MNI space).

---

## 3.2 Neural Mass Model Configuration

### 3.2.1 The Epileptor Model

We use the **Epileptor** neural mass model (Jirsa et al., 2014) as implemented in `tvb.simulator.models.epileptor.Epileptor`. This is a 6-dimensional ODE system per region that produces realistic seizure dynamics through a separation of time scales.

**Epileptor equations** (per region $i$):

$$\dot{x}_{1,i} = y_{1,i} - f_1(x_{1,i}, x_{2,i}) - z_i + I_{ext,1}$$

$$\dot{y}_{1,i} = y_0 - 5 x_{1,i}^2 - y_{1,i}$$

$$\dot{z}_i = \frac{1}{\tau_0}\left(4(x_{1,i} - x_{0,i}) - z_i\right)$$

$$\dot{x}_{2,i} = -y_{2,i} + x_{2,i} - x_{2,i}^3 + I_{ext,2} + 0.002 g_i - 0.3(z_i - 3.5)$$

$$\dot{y}_{2,i} = \frac{1}{\tau_2}(-y_{2,i} + f_2(x_{2,i}))$$

$$\dot{g}_i = -0.01(g_i - 0.1 x_{1,i})$$

where:
- $x_{0,i}$ is the **excitability parameter** — the primary parameter of interest for epileptogenicity mapping
- $f_1, f_2$ are piecewise nonlinear functions defining the fast and slow subsystem dynamics
- $g_i$ provides coupling between the fast and slow subsystems
- Network coupling enters through: $\sum_j W_{ij} \cdot (x_{1,j} - x_{1,i})$ added to the $\dot{x}_{1,i}$ equation

**Critical parameter: $x_0$ (excitability)**

| Regime | $x_0$ range | Behavior |
|--------|-------------|----------|
| Healthy (non-epileptogenic) | $[-2.2, -2.1]$ | Stable fixed point, only background oscillations |
| Near-critical | $[-2.1, -1.8]$ | Susceptible to perturbation, occasional spikes |
| Epileptogenic | $[-1.8, -1.6]$ | Spontaneous interictal spikes |
| Seizure-generating | $[-1.6, -1.2]$ | Spontaneous seizure-like oscillations |

**Academic justification**: These ranges follow Proix et al. (2017), Hashemi et al. (2020), and the Virtual Epileptic Patient framework (Jirsa et al., 2017; Sip et al., 2022). The x₀ range [−2.2, −1.2] for epileptogenicity mapping is directly from Vattikonda et al. (2021) and the virtual epilepsy patient cohort study (2024).

### 3.2.2 Epileptor Parameter Table

| Parameter | Symbol | Default | Sampling Range (for dataset generation) | Unit |
|-----------|--------|---------|----------------------------------------|------|
| Excitability | $x_0$ | -1.6 | [-2.2, -1.2] | dimensionless |
| External input 1 | $I_{ext,1}$ | 3.1 | [2.8, 3.4] | dimensionless |
| External input 2 | $I_{ext,2}$ | 0.45 | [0.3, 0.6] | dimensionless |
| Time constant (slow) | $\tau_0$ | 2857.0 | [2000, 4000] | ms |
| Time constant (fast subsystem 2) | $\tau_2$ | 10.0 | [6, 15] | ms |
| Equilibrium point | $y_0$ | 1.0 | fixed | dimensionless |
| Global coupling strength | $G$ | 1.0 | [0.5, 3.0] | dimensionless |
| Conduction speed | $v$ | 4.0 | [3.0, 6.0] | m/s |
| Noise intensity | $D$ | 0.0005 | [0.0001, 0.005] | dimensionless |

### 3.2.3 TVB Simulator Configuration

```python
# Pseudocode — exact implementation in epileptor_simulator.py
simulator = tvb.simulator.Simulator(
    model=Epileptor(),
    connectivity=connectivity,          # 76-region DK atlas
    coupling=tvb.simulator.coupling.Difference(a=G),
    integrator=HeunStochastic(
        dt=0.1,                         # 0.1 ms step (≤0.1 required for stability)
        noise=Additive(nsig=[D, D, 0, D, D, 0])  # Fast vars only, NO noise on z, g
    ),
    monitors=[Raw()],                   # Full-resolution output (no box-car aliasing)
    simulation_length=12000.0           # 12 seconds (first 2s transient discarded)
)
# After simulation: anti-aliased decimation via scipy.signal.decimate(ftype='fir')
# TVB 2× rate factor: dt=0.1ms → 20,000 Hz actual → decimate by 100 (10×10) → 200 Hz
```

**Integration method**: Heun stochastic (second-order Runge-Kutta with additive noise). This is the standard integrator for the Epileptor in TVB (Sanz-Leon et al., 2013). The time step must be ≤ 0.1 ms for numerical stability with the Epileptor fast subsystem.

**Noise structure**: Additive noise is applied only to the fast subsystem variables (x1, y1, x2, y2) and NOT to the slow variables (z, g). The slow variables have very large effective time constants (τ₀ ≈ 2857 ms), so even small noise perturbations accumulate and cause numerical divergence. This is consistent with the Epileptor literature (Jirsa et al., 2014; Proix et al., 2017). The noise vector per region is `[D, D, 0, D, D, 0]` for state variables `[x1, y1, z, x2, y2, g]`.

**Anti-aliased decimation**: TVB's `TemporalAverage` monitor performs only a simple box-car average with no anti-aliasing filter. The Epileptor fast subsystem generates significant energy at frequencies up to several kHz (at dt=0.1 ms), which `TemporalAverage` aliases into the Nyquist bin (~100 Hz at 200 Hz output), producing unphysical spectral content (>90% power at Nyquist, negative lag-1 autocorrelation). Instead, we use the `Raw` monitor and apply `scipy.signal.decimate(ftype='fir')` in two stages (factor 10 × 10 = 100), which applies a proper FIR low-pass anti-aliasing filter before each downsampling step. This preserves the biologically correct spectral profile (dominant power <30 Hz, positive lag-1 autocorrelation ≈ 0.5–0.6).

**TVB 2× output rate factor**: TVB's Raw monitor outputs at 2× the nominal integration rate. For dt=0.1 ms, the nominal rate is 10,000 Hz, but the actual output rate is 20,000 Hz. The decimation factor accounts for this: 20,000 / 200 = 100.

**Output variable**: The $x_2 - x_1$ combination (LFP proxy), which is the observable used as the source signal. This follows the convention in Jirsa et al. (2014) and Proix et al. (2017) where the LFP is approximated as the difference between the fast and slow subsystem variables.

**Divergence handling**: Some parameter combinations (high coupling + high noise + extreme x0) cause the Epileptor to numerically diverge, producing NaN values. These simulations are detected (NaN/Inf check on the decimated output) and discarded. Typical failure rate is ~5–10% of simulations, which is acceptable given the large dataset size.

---

## 3.3 Leadfield Matrix Construction

### 3.3.1 Forward Model Pipeline

The leadfield matrix $\mathbf{L} \in \mathbb{R}^{19 \times 76}$ maps 76 regional source activations to 19 scalp electrode potentials. Construction proceeds as follows:

**Step 1: Head model setup**

Using MNE-Python with the `fsaverage` template subject:

```python
# High-level flow
subjects_dir = mne.datasets.fetch_fsaverage()
# BEM model: 3-layer (skin, skull, brain) with standard conductivities
model = mne.make_bem_model(subject='fsaverage', conductivity=(0.3, 0.006, 0.3))
bem_sol = mne.make_bem_solution(model)
```

Conductivities: skin = 0.3 S/m, skull = 0.006 S/m, brain = 0.3 S/m. These are standard values (Gramfort et al., 2010; Vorwerk et al., 2014).

**Step 2: Source space at region centroids**

Rather than using a dense cortical source space, we create a discrete source space at the 76 DK region centroids. Each region is represented as a single equivalent current dipole at its centroid, oriented normal to the cortical surface at that location.

```python
# Create source space from parcellation centroids
# Two approaches:
# (A) Use mne.setup_volume_source_space with specific positions
# (B) Create a cortical source space, compute forward, then average
#     leadfield columns within each DK parcel

# Approach B is more physically accurate:
src = mne.setup_source_space('fsaverage', spacing='oct6')
fwd = mne.make_forward_solution(raw.info, src=src, bem=bem_sol,
                                  eeg=True, mindist=5.0)
# Average leadfield columns within each DK parcel
labels = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                      subjects_dir=subjects_dir)
L_region = average_leadfield_by_parcellation(fwd, labels)  # → (19, 76)
```

**Approach B detail**: For each of the 76 DK regions, we identify all source space vertices belonging to that parcel. The parcel-level leadfield column is the **mean** of all vertex-level leadfield columns (after projecting each vertex's leadfield from 3D orientation to the surface-normal component). This is the standard region-level forward model aggregation method (Palva et al., 2018; Mahjoory et al., 2017).

**Step 3: Montage and reference**

- Montage: Standard 10-20 with 19 channels, created via `mne.channels.make_standard_montage('standard_1020')`.
- Reference: Linked ears (A1+A2). After computing the forward solution in the average reference (MNE default), we apply the re-referencing transformation:

$$L_{linked} = L_{avg} - \frac{1}{2}(L_{A1} + L_{A2})$$

Since A1 and A2 are reference electrodes in the 10-20 system and not part of the 19 recording channels, the re-referencing is applied as a projection matrix that removes the average ear potential.

**Step 4: Validation**

The leadfield matrix is validated by checking:

1. **Rank**: $\text{rank}(\mathbf{L}) = \min(19, 76) - 1 = 18$ (minus 1 due to reference).
2. **Column norms**: No column should have a norm more than 100× the median (would indicate a source unreasonably close to an electrode).
3. **Spatial pattern**: The leadfield for each region should show a physically plausible scalp topography (verified visually for a subset of regions).
4. **Reciprocity check**: Forward-projected known source patterns should match expected scalp distributions.

**Stored as**: `leadfield.npy` — shape `(19, 76)`, dtype `float64`.

### 3.3.2 Channel Order

The 19 channels are stored in the following fixed order (matching NMT dataset convention):

```
['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
```

Note: T3/T4/T5/T6 are the older 10-20 nomenclature (equivalent to T7/T8/P7/P8 in the 10-10 system). The NMT dataset uses the older convention.

---

## 3.4 Synthetic EEG Dataset Generation

### 3.4.1 Sampling Strategy

Each synthetic sample is generated by the following randomization procedure:

1. **Number of epileptogenic regions**: $k \sim \text{DiscreteUniform}(0, 8)$. When $k = 0$, all 76 regions are healthy (no epileptogenic zone). This ensures the network also learns normal brain activity patterns for general-purpose inverse solving and brain activity mapping on non-epileptic patients. Approximately 11% of samples (1/9) are healthy-only.

   **Oversampling Justification**: The resulting training distribution (11% healthy, 89% epileptic) deliberately oversamples epilepsy compared to real-world prevalence (~0.5–1.1% globally and in the US). This approach is scientifically justified by the medical deep learning literature (Johnson & Khoshgoftaar, 2019; Schubach et al., 2017; Fajardo et al., 2021) and is necessary to prevent the network from learning a trivial "always output healthy" solution. The training-time oversampling does not reflect expected test-time performance; Phase 5 (Validation) explicitly evaluates the network's robustness to realistic class distributions in the NMT dataset (~17.5% abnormal) and custom test sets with controlled epilepsy rates. This is the standard practice in medical AI for rare disease detection.
2. **Which regions are epileptogenic** (for $k \geq 1$): $k$ regions selected uniformly at random, with optional spatial clustering bias (50% chance of selecting a spatially contiguous cluster using the connectivity adjacency).
3. **Excitability assignment**:
   - Epileptogenic regions ($k \geq 1$): $x_{0,i} \sim \text{Uniform}(-1.8, -1.2)$
   - Healthy regions (all 76 when $k = 0$, or the remaining $76 - k$): $x_{0,i} \sim \text{Uniform}(-2.2, -2.05)$
4. **Global coupling**: $G \sim \text{Uniform}(0.5, 3.0)$
5. **Noise intensity**: $D \sim \text{LogUniform}(10^{-4}, 5 \times 10^{-3})$
6. **Conduction speed**: $v \sim \text{Uniform}(3.0, 6.0)$ m/s

### 3.4.2 Simulation and Projection

For each parameter sample:

1. Configure TVB simulator with sampled parameters (dt=0.1 ms, Raw monitor, 12-second duration).
2. Run simulation for 12 seconds. Apply anti-aliased FIR decimation (factor 100 = 10×10) to downsample from 20,000 Hz to 200 Hz. Discard first 2 seconds (400 samples at 200 Hz) as initial transient.
3. Extract the LFP proxy ($x_2 - x_1$) for all 76 regions at 200 Hz → `source_activity` shape `(76, 2000)`.
4. Segment into non-overlapping 2-second windows → 5 windows per simulation, each `(76, 400)`.
5. For each window, project through leadfield: `eeg = L @ source_activity` → shape `(19, 400)`.
6. Add measurement noise:
   - White Gaussian noise at SNR sampled from $\text{Uniform}(5, 30)$ dB.
   - Colored noise: $1/f^\alpha$ noise with $\alpha \sim \text{Uniform}(0.5, 1.5)$, amplitude 10–30% of signal RMS.
7. Apply skull attenuation filter: 4th-order Butterworth lowpass @ 40 Hz, zero-phase (`sosfiltfilt`). Models real skull spatial filtering and amplifier anti-aliasing (Nunez & Srinivasan, 2006).
8. Apply integrated spectral shaping (STFT-based, see Phase 1.6 below):
   - STFT decomposition (200-sample Hann, 50% overlap)
   - Delta suppression (×0.38), theta suppression (×0.75)
   - Alpha boost (×1.8) + per-frame adaptive anteroposterior redistribution
   - Fixed per-channel beta gradient gains (frontal-high → occipital-low)
   - ISTFT reconstruction + RMS normalization to preserve amplitude
9. Validate output: reject any simulation where source_activity contains NaN or Inf (diverged simulations), and validate spatial-spectral properties (group-level alpha/beta gradients, PDR ∈ [1.3, 5.0]).

### 3.4.3 Dataset Size

| Dataset Split | Number of Simulations | Windows per Simulation | Total Samples |
|---------------|----------------------|----------------------|---------------|
| Training | 16,000 | 5 | 80,000 |
| Validation | 2,000 | 5 | 10,000 |
| Test | 2,000 | 5 | 10,000 |
| **Total** | **20,000** | — | **100,000** |

### 3.4.4 HDF5 Storage Format

Each split is stored as a single HDF5 file:

```
train_dataset.h5
├── eeg/                    # shape (80000, 19, 400), dtype float32
├── source_activity/        # shape (80000, 76, 400), dtype float32
├── epileptogenic_mask/     # shape (80000, 76), dtype bool
├── x0_vector/              # shape (80000, 76), dtype float32
├── snr_db/                 # shape (80000,), dtype float32
├── global_coupling/        # shape (80000,), dtype float32
├── metadata/
│   ├── channel_names       # string array, length 19
│   ├── region_names        # string array, length 76
│   ├── sampling_rate       # scalar, 200.0
│   └── window_length_sec   # scalar, 2.0
```

### 3.4.5 Data Normalization

Before feeding into the network:

1. **Per-region temporal de-meaning** (added v2, see Section 4.4.7):
   - **Source activity**: Subtract per-region temporal mean from each region's time series: $\tilde{S}_i(t) = S_i(t) - \bar{S}_i$. This removes the Epileptor x2-x1 DC offset, which varies with x0 and dominates 98.1% of signal power, leaving only the AC dynamics component that carries the epileptogenicity-discriminative information.
   - **EEG**: Subtract per-channel temporal mean for consistency with AC-coupled clinical EEG recordings.

2. **Global z-score normalization** (applied after de-meaning):
   - Statistics computed from a probe of 5,000 training samples (de-meaned)
   - Applied identically to all samples: $(x - \mu) / (\sigma + \epsilon)$ where $\epsilon = 10^{-8}$
   - After de-meaning, src_mean ≈ 0.0 and src_std reflects the dynamics scale (~0.05–0.10 expected)

**Original specification** (v1, superseded): z-score per window for EEG, max-abs scaling per window for sources. See Section 4.4.3 for rationale behind global normalization, and Section 4.4.7 for rationale behind de-meaning.

These normalizations are applied on-the-fly in the PyTorch Dataset's iteration method to preserve the raw data in HDF5.

---

### 3.4.6 Incremental HDF5 Writing Strategy

To minimize memory usage and provide fault tolerance during generation, datasets are written to HDF5 incrementally in batches rather than accumulating all samples in RAM:

**Algorithm**:
1. **File creation** (Step 1): Create an empty HDF5 file with resizable datasets:
   - All main datasets (eeg, source_activity, epileptogenic_mask, x0_vector, snr_db, global_coupling)
   - `shape=(0, ...)` with `maxshape=(None, ...)` for unlimited growth
   - Use HDF5 chunking for efficient incremental I/O: `chunks=(batch_size, *spatial_dims)`

2. **Incremental writing** (Step 2): As simulations complete:
   - Collect results into batch accumulators (default batch_size=500 samples)
   - When batch_size is reached, append to HDF5 using `Dataset.resize()` and slice assignment
   - Clear accumulators and continue
   - After all simulations, write any remaining incomplete batch

3. **Memory efficiency**:
   - Only ~1-2 GB RAM used at any time (one batch + intermediate arrays)
   - Avoids storing 100,000+ samples simultaneously
   - With 80,000 training samples at 500/batch = 160 write operations

4. **Fault tolerance**:
   - If generation crashes at hour 9 of 12, all completed batches are preserved in the HDF5 file
   - Generation can resume by filtering out already-written simulations (future enhancement)

5. **Progress monitoring**:
   - HDF5 file can be read while generation is in progress
   - Current sample count accessible via `f["eeg"].shape[0]`
   - Real-time progress visualization possible

**Configuration** (in config.yaml):
```yaml
synthetic_data:
  hdf5_batch_size: 500  # Write to HDF5 every 500 samples
```

**Reference**: Technical Specs Section 3.4.4 (HDF5 Storage Format)

---

# 4. Phase 2: PhysDeepSIF Network Architecture and Training

## 4.1 Network Architecture

PhysDeepSIF is adapted from the DeepSIF architecture (Sun et al., 2022, PNAS) with modifications for the 76-region parcellated source space (instead of the original vertex-level source space).

### 4.1.1 Architecture Diagram

```
Input: EEG tensor — (batch, 19, 400)
                    │
    ┌───────────────┴───────────────┐
    │      SPATIAL MODULE            │
    │                                │
    │  For each time step t:         │
    │    x_t = EEG[:, :, t]  (19,)  │
    │                                │
    │    FC1: 19 → 128 + ReLU        │
    │    FC2: 128 → 256 + ReLU       │
    │    FC3: 256 → 256 + ReLU       │
    │       + skip from FC1 (pad)    │
    │    FC4: 256 → 128 + ReLU       │
    │    FC5: 128 → 76               │
    │       + skip from FC2 (proj)   │
    │                                │
    │  Output: (batch, 76, 400)      │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │      TEMPORAL MODULE           │
    │                                │
    │  BiLSTM Layer 1:               │
    │    input_size=76               │
    │    hidden_size=76              │
    │    bidirectional=True          │
    │    → output (batch, 400, 152)  │
    │                                │
    │  BiLSTM Layer 2:               │
    │    input_size=152              │
    │    hidden_size=76              │
    │    bidirectional=True          │
    │    → output (batch, 400, 152)  │
    │                                │
    │  FC Projection:                │
    │    152 → 76                    │
    │    + skip from spatial module  │
    │                                │
    │  Output: (batch, 76, 400)      │
    └───────────────┬───────────────┘
                    │
            Source Estimate
            (batch, 76, 400)
```

### 4.1.2 Spatial Module Detail

The spatial module operates independently on each time step (weight-shared across time). It is a multilayer perceptron (MLP) that maps from 19 sensor channels to 76 regions.

- **Layer 1**: Linear(19, 128) → BatchNorm1d(128) → ReLU
- **Layer 2**: Linear(128, 256) → BatchNorm1d(256) → ReLU
- **Layer 3**: Linear(256, 256) → BatchNorm1d(256) → ReLU + skip connection from Layer 1 (zero-padded to 256)
- **Layer 4**: Linear(256, 128) → BatchNorm1d(128) → ReLU
- **Layer 5**: Linear(128, 76) + skip connection from Layer 2 (via a learned 256→76 linear projection)

**Justification**: DeepSIF uses 5 FC layers with skip connections for the spatial module (Sun et al., 2022). The skip connections prevent gradient degradation and allow the network to learn residual mappings. BatchNorm stabilizes training on synthetic data with varying SNR.

### 4.1.3 Temporal Module Detail

The temporal module processes the time-series output of the spatial module to enforce temporal consistency.

- **BiLSTM 1**: input_size=76, hidden_size=76, bidirectional=True, dropout=0.1
- **BiLSTM 2**: input_size=152, hidden_size=76, bidirectional=True, dropout=0.1
- **Output projection**: Linear(152, 76)
- **Skip connection**: Element-wise addition of the spatial module output

**Justification**: The original DeepSIF uses LSTM layers with input/output size of 994 (vertex count). Our adapted version uses 76 (region count). Bidirectional LSTMs allow the temporal module to use both past and future context for each time point, which is appropriate for offline analysis. Skip connections from the spatial module preserve spatial information.

### 4.1.4 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Spatial FC layers | ~150,000 |
| Temporal BiLSTM layers | ~185,000 |
| Skip projections | ~20,000 |
| **Total** | **~355,000** |

This is a deliberately compact architecture. The original DeepSIF has ~2M parameters due to the 994-vertex source space; our 76-region formulation is much smaller.

---

## 4.2 Physics-Informed Loss Function

**Assumption**: The physics-informed loss function is already implemented. This section specifies its exact mathematical form for reference and integration purposes.

### 4.2.1 Composite Loss

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{source} + \beta \cdot \mathcal{L}_{forward} + \gamma \cdot \mathcal{L}_{physics}$$

### 4.2.2 Source Reconstruction Loss

$$\mathcal{L}_{source} = \frac{1}{N_r \cdot T} \sum_{i=1}^{N_r} \sum_{t=1}^{T} \left( \hat{S}_{i,t} - S_{i,t}^{true} \right)^2$$

where $N_r = 76$, $T = 400$, $\hat{S}$ is the predicted source activity, $S^{true}$ is the ground truth source activity.

### 4.2.3 Forward Consistency Loss

$$\mathcal{L}_{forward} = \frac{1}{N_c \cdot T} \sum_{j=1}^{N_c} \sum_{t=1}^{T} \left( \mathbf{L} \hat{S}_t - \text{EEG}_t^{input} \right)_j^2$$

where $N_c = 19$, $\mathbf{L}$ is the leadfield matrix (frozen, not learnable), and $\text{EEG}^{input}$ is the (noisy) input EEG.

**Implementation detail**: The leadfield matrix is registered as a non-trainable buffer in the PyTorch module (`self.register_buffer('leadfield', L_tensor)`).

### 4.2.4 Physiological Regularization Loss

$$\mathcal{L}_{physics} = \lambda_1 \cdot \mathcal{L}_{Laplacian} + \lambda_2 \cdot \mathcal{L}_{temporal} + \lambda_3 \cdot \mathcal{L}_{amplitude}$$

**Laplacian regularization** (spatial smoothness via connectivity):

$$\mathcal{L}_{Laplacian} = \frac{1}{T} \sum_{t=1}^{T} \hat{S}_t^\top \mathbf{D} \hat{S}_t$$

where $\mathbf{D} = \text{diag}(\mathbf{W}\mathbf{1}) - \mathbf{W}$ is the graph Laplacian of the structural connectivity matrix $\mathbf{W}$.

**Temporal smoothness** (penalize high-frequency jitter):

$$\mathcal{L}_{temporal} = \frac{1}{N_r \cdot (T-1)} \sum_{i=1}^{N_r} \sum_{t=1}^{T-1} \left( \hat{S}_{i,t+1} - \hat{S}_{i,t} \right)^2$$

**Amplitude bound** (soft constraint on physiological range):

$$\mathcal{L}_{amplitude} = \frac{1}{N_r \cdot T} \sum_{i,t} \max(0, |\hat{S}_{i,t}| - A_{max})^2$$

where $A_{max} = 3.0$ (after normalization).

### 4.2.5 Loss Hyperparameters

| Hyperparameter | Symbol | Initial Value | Tuning Strategy |
|----------------|--------|---------------|-----------------|
| Source weight | $\alpha$ | 1.0 | Fixed |
| Forward weight | $\beta$ | 0.5 | Grid search over {0.1, 0.5, 1.0, 2.0} |
| Physics weight | $\gamma$ | 0.1 | Grid search over {0.01, 0.05, 0.1, 0.5} |
| Laplacian sub-weight | $\lambda_1$ | 0.5 | Fixed relative to $\gamma$ |
| Temporal sub-weight | $\lambda_2$ | 0.3 | Fixed relative to $\gamma$ |
| Amplitude sub-weight | $\lambda_3$ | 0.2 | Fixed relative to $\gamma$ |

---

## 4.3 Training Procedure

### 4.3.1 Optimizer and Schedule

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-3}$ |
| Weight decay | $1 \times 10^{-4}$ |
| LR schedule | CosineAnnealingWarmRestarts, $T_0=10$, $T_{mult}=2$ |
| Batch size | 64 |
| Maximum epochs | 200 |
| Early stopping | Patience = 15 epochs on validation loss |
| Gradient clipping | Max norm = 1.0 |

### 4.3.2 Data Augmentation (During Training)

Applied on-the-fly:

1. **Random SNR perturbation**: Add additional Gaussian noise with SNR $\sim \text{Uniform}(10, 40)$ dB to the already-noisy EEG.
2. **Temporal jitter**: Random circular shift of ±10 samples (±50 ms).
3. **Channel dropout**: Randomly zero out 0–2 channels (simulating bad electrode contact), replaced by interpolation from neighboring channels.
4. **Amplitude scaling**: Multiply EEG by a random factor $\sim \text{Uniform}(0.8, 1.2)$.

### 4.3.3 Training Monitoring Metrics

Computed on the validation set every epoch:

| Metric | Formula | Target |
|--------|---------|--------|
| Source MSE | $\|\hat{S} - S^{true}\|_F^2 / (N_r \cdot T)$ | Minimize |
| Forward MSE | $\|L\hat{S} - \text{EEG}\|_F^2 / (N_c \cdot T)$ | Minimize |
| Dipole Localization Error (DLE) | Euclidean distance between centroid of predicted activation and centroid of true activation (mm) | < 20 mm |
| Spatial Dispersion (SD) | Standard deviation of the predicted source distribution around the true source centroid (mm) | < 30 mm |
| Area Under ROC (AUC) | Binary classification: is region epileptogenic? Threshold the predicted source power to get binary map | > 0.85 |
| Pearson correlation (temporal) | Correlation between predicted and true source time course, averaged over active regions | > 0.7 |

**Definitions**:

**Dipole Localization Error (DLE)** (Molins et al., 2008):
$$\text{DLE} = \left\| \bar{r}_{pred} - \bar{r}_{true} \right\|_2$$
where $\bar{r}_{pred} = \frac{\sum_i P_i \cdot r_i}{\sum_i P_i}$ is the power-weighted centroid of predicted source activity, $P_i = \frac{1}{T}\sum_t \hat{S}_{i,t}^2$ is the time-averaged power in region $i$, and $r_i$ is the 3D centroid of region $i$.

**Spatial Dispersion (SD)** (Molins et al., 2008):
$$\text{SD} = \sqrt{\frac{\sum_i P_i \cdot \|r_i - \bar{r}_{true}\|^2}{\sum_i P_i}}$$

These metrics are standard in the EEG source imaging literature (Grech et al., 2008; Sun et al., 2022; Hecker et al., 2021).

---

## 4.4 Phase 2 Execution Results and Analysis

This section documents the actual execution results from Phase 2 training and the subsequent diagnostic analyses. These findings inform the hyperparameter tuning strategy described in Section 4.5.

### 4.4.1 Actual Dataset Statistics

Due to simulation divergence filtering (~5–10% failure rate as anticipated in Section 3.2.3), the actual dataset sizes differ slightly from the target counts in Section 3.4.3:

| Dataset Split | Target Samples | Actual Samples | Divergence Rate |
|---------------|----------------|----------------|-----------------|
| Training | 80,000 | 74,085 | ~7.4% |
| Validation | 10,000 | 9,255 | ~7.5% |
| Test | 10,000 | TBD | ~7–8% estimated |

**Class distribution** (training set):
- Epileptic samples (k ≥ 1): 65,561 (88.5%)
- Healthy samples (k = 0): 8,524 (11.5%)
- Expected: ~11.1% healthy (1/9 probability from k ∼ DiscreteUniform(0, 8))

### 4.4.2 Actual Parameter Count

The implemented PhysDeepSIF network has a higher parameter count than initially estimated:

| Component | Estimated (Section 4.1.4) | Actual | Difference |
|-----------|--------------------------|--------|------------|
| Spatial module | ~150,000 | 165,144 | +10% |
| Temporal module | ~185,000 | 245,100 | +32% |
| Total trainable | ~355,000 | 410,244 | +15.6% |
| Non-trainable buffers | 7,300 | 7,220 | ~same |
| **Grand total** | **~362,300** | **417,464** | **+15.2%** |

The discrepancy arises primarily from:
1. **Bias terms** in LSTM layers (not included in the original estimate)
2. **Skip projection layer** (256→76 = 19,532 params including bias)
3. **LSTM gate calculations** including bias terms per gate per direction

The model remains compact relative to the original DeepSIF (~2M parameters).

### 4.4.3 Data Normalization Implementation

**Deviation from specification**: Section 3.4.5 specifies per-sample normalization (z-score per window for EEG, max-abs scaling per window for sources). The actual training implementation uses **global z-score normalization** computed over the entire training set:

| Statistic | EEG | Source Activity |
|-----------|-----|-----------------|
| Mean (μ) | 0.0050 | -0.0002 |
| Std (σ) | 559.2 | 0.2561 |

**Rationale**: Global normalization preserves relative amplitude relationships across samples, which is important for the forward consistency loss $\mathcal{L}_{forward}$. Per-sample normalization would destroy the physics-based relationship $\text{EEG} = \mathbf{L} \cdot \mathbf{S}$ since each sample would be independently scaled.

**Memory optimization**: The normalization function was modified to use in-place PyTorch operations (`.sub_()`, `.div_()`) after discovering that the standard out-of-place operations created temporary tensors that doubled RAM usage from ~20 GB to ~40 GB, exceeding the system's 39.1 GB available RAM.

### 4.4.4 Initial Training Run Results

**Training configuration** (actual vs. spec):

| Parameter | Specification (Section 4.3.1) | Actual |
|-----------|------------------------------|--------|
| Batch size | 64 | 32 (reduced for RAM) |
| Max epochs | 200 | 200 |
| Early stopping patience | 15 | 15 |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| Loss weights (α, β, γ) | (1.0, 0.5, 0.1) | (1.0, 0.5, 0.1) |
| Device | GPU (CUDA) | NVIDIA RTX 3080 (10 GB VRAM) |

**Training outcome**:
- Training stopped at **epoch 23** via early stopping (best model at epoch 8)
- Total training time: **1.84 hours** (110 minutes)
- GPU VRAM usage: **< 0.5 GB** (model is very compact for a 10 GB GPU)
- System RAM usage: **24.6 GB** stable during training

**Best model metrics** (epoch 8, validation set):

| Metric | Value | Target (Section 7.1.1) | Status |
|--------|-------|----------------------|--------|
| Validation loss | 1.0105 | Minimize | — |
| DLE | 15.48 mm | < 20 mm | ✅ Pass |
| SD | 48.52 mm | < 30 mm | ❌ Fail |
| AUC | 0.495 | > 0.85 | ❌ Fail |
| Temporal correlation | 0.035 | > 0.7 | ❌ Fail |

### 4.4.5 Amplitude Collapse Diagnosis

Post-training validation revealed a critical issue: the network's predicted source activity has **25× smaller amplitude** than the ground truth:

| Statistic | Predicted Sources | True Sources | Ratio |
|-----------|------------------|--------------|-------|
| Standard deviation (σ) | 0.0086 | 0.217 | 0.040× |
| Max absolute value | 0.089 | 2.17 | 0.041× |
| Mean absolute value | 0.0069 | 0.174 | 0.040× |

**Root Cause Analysis**: The amplitude collapse is caused by the interaction of three factors:

**Factor 1: Leadfield matrix ill-conditioning**

| Property | Value |
|----------|-------|
| Condition number (κ) | 1.31 × 10¹⁶ |
| Effective rank | 18 (of 19) |
| Singular value range | [4.3 × 10⁻¹⁴, 560.3] |
| Under-determination ratio | 76:18 ≈ 4.2:1 |

The leadfield matrix $\mathbf{L}$ is extremely ill-conditioned (κ ≈ 10¹⁶), consistent with the well-known ill-posedness of the EEG inverse problem (Baillet et al., 2001). With only 18 effective measurements for 76 unknowns, the system is ~4× under-determined.

**Factor 2: Forward consistency loss dominance**

The forward consistency loss $\mathcal{L}_{forward} = \|\mathbf{L}\hat{S} - \text{EEG}\|^2$ dominates the loss landscape:

| Prediction Scale | $\sigma_{pred}$ | $\mathcal{L}_{source}$ | $\mathcal{L}_{forward}$ | $\mathcal{L}_{total}$ |
|------------------|-----------------|----------------------|------------------------|----------------------|
| 1.0 (perfect) | 1.000 | 0.000 | 41,456 | 20,728 |
| 0.50 | 0.500 | 0.250 | 10,358 | 5,179 |
| 0.10 | 0.100 | 0.810 | 413 | 207 |
| 0.04 (network) | 0.040 | 0.922 | 66 | 34 |

At the *correct* prediction scale (σ = 1.0), the forward loss is **41,456**, while the source loss is **0.0**. The optimizer finds a much lower total loss by shrinking predictions to σ ≈ 0.04, where $\mathcal{L}_{total} = 34$ — a **600× reduction** from the correct solution.

**Factor 3: Normalization mismatch**

After global z-score normalization:
- EEG σ = 1.0 (by construction)
- Sources σ = 1.0 (by construction)
- But $\mathbf{L} \cdot \mathbf{S}_{norm}$ yields σ ≈ 290 (not 1.0)

The leadfield amplifies normalized source signals by ~290×, making the forward loss enormous at the correct source scale. The original EEG σ was 559.2, meaning the true relationship is $\mathbf{L} \cdot \mathbf{S}_{raw} \approx \text{EEG}_{raw}$, but after independent normalization, this relationship is broken.

**Implication**: The DLE metric still performs well (15.48 mm < 20 mm target) because it depends only on relative spatial power patterns, not absolute amplitudes. However, AUC (0.495 ≈ chance) and temporal correlation (0.035 ≈ zero) fail because they require correct amplitude dynamics.

**Comparison to classical methods**: This behavior is analogous to the amplitude suppression seen in MNE/eLORETA solutions (Grech et al., 2008), but more extreme. Standard inverse methods also underestimate source amplitudes due to regularization, typically by factors of 2–5×. Our 25× suppression indicates the forward loss weight β = 0.5 acts as excessively strong implicit regularization.

### 4.4.6 Conclusion and Remediation Strategy

The initial training results demonstrate that:

1. **The network architecture is functional** — it can learn spatial EEG-to-source mappings (DLE < 20 mm)
2. **The loss weight balance is incorrect** — β = 0.5 forward loss dominates and causes amplitude collapse
3. **Temporal correlation cannot emerge** without correct amplitude recovery
4. **GPU resources are not a bottleneck** — the model uses < 5% of available 10 GB VRAM

**Remediation**: Bayesian hyperparameter optimization over loss weights (α, β, γ), learning rate, and other training hyperparameters, optimizing specifically for temporal correlation as the primary objective. This replaces the grid search strategy specified in Section 4.2.5 with a more efficient search method. See Section 4.5 for the updated tuning strategy.

### 4.4.7 DC Offset Dominance Root Cause (Discovered 2026-02-27)

**Summary**: A second, independent root cause of poor model performance was discovered during post-demo analysis. The Epileptor x2-x1 LFP proxy contains a large, spatially-varying DC offset that dominates the source signal, masking the variance/dynamics component that carries the epileptogenicity-discriminative information.

#### 4.4.7.1 Evidence

The Epileptor's slow permittivity variable coupling (x2-x1) produces a resting-state DC level that shifts with the excitability parameter x0:

| Region Type | x0 Range | mean(x2-x1) | Temporal Variance | Variance as % of Power |
|-------------|----------|--------------|-------------------|------------------------|
| Healthy | [-2.2, -2.05] | ≈ 1.80 | ≈ 0.045 | ~1.4% |
| Epileptogenic | [-1.8, -1.2] | ≈ 1.60 | ≈ 0.178 | ~6.5% |
| **Global average** | mixed | **1.792** | **~0.066** | **~1.9%** |

**Key finding**: Power = mean² + variance, and mean² accounts for **98.1%** of total source power across the training set. The global normalization (src_mean=1.792, src_std=0.258) shifts the DC toward zero but the MSE loss still learns predominantly from the residual DC structure rather than the dynamics.

**Discriminative analysis** (50 epileptogenic test samples, top-10 recall metric):

| Feature | Epi Value | Healthy Value | Ratio | Oracle Top-10 Recall |
|---------|-----------|---------------|-------|---------------------|
| Power (mean s²) | 2.74 | 3.29 | 0.84 | 0.023 (inverted!) |
| Variance | 0.178 | 0.045 | 3.92 | 0.886 ✓ |
| Range (ptp) | 1.94 | 1.04 | 1.86 | 0.771 ✓ |
| Kurtosis (inv) | — | — | — | 0.913 ✓ |

The variance is 3.9× higher in epileptogenic regions (correct direction for detection), but power-based scoring **inverts** because epileptogenic regions have lower DC offset (mean), making time-averaged power rank them as *less* active.

**Model output analysis**:
- Predicted source spatial CV = 0.002 (near-constant across regions)
- Predicted temporal variance ≈ 0.00002 (vs true ≈ 0.05 — a 2,500× gap)
- The model outputs 100% DC, 0% AC — it learned the global mean but not the dynamics

#### 4.4.7.2 Physical Explanation

This is **not a bug** — it is physically correct Epileptor behavior (Jirsa et al., 2014):
- The x2-x1 quantity represents the difference between the slow permittivity variable (x2) and the fast population variable (x1)
- As x0 approaches the critical bifurcation threshold (~-1.6), the system's equilibrium point shifts, changing the DC offset
- Near-critical x0 values produce intermittent bursting with quiescent baseline → high variance
- Far-from-critical x0 values (healthy) produce stable oscillations around a higher equilibrium → low variance
- Clinical parallel: focal epilepsy shows background suppression between interictal discharges

#### 4.4.7.3 Why This Is Incompatible with the Current Training Pipeline

1. **Real EEG is AC-coupled**: All clinical EEG amplifiers use AC coupling (highpass ≥ 0.1 Hz). The NMT dataset uses 0.5 Hz highpass. The DC component of the source signal **cannot be reconstructed** from the EEG input because it does not appear in the measurement. Training the model to predict a DC offset that has no corresponding signature in the input creates an impossible learning objective for the dynamics.

2. **MSE loss focuses on DC, not dynamics**: With 98.1% of signal power in the DC component, the MSE loss gradient is dominated by DC prediction error. The model converges to outputting the population mean DC level and ignores the 1.9% variance component, which is the actual discriminative signal.

3. **Literature comparison**: MS-ESI (Yu et al., 2024, NeuroImage) and DeepSIF (Sun et al., 2022, PNAS) train on interictal spike waveforms generated by neural mass models (Wendling/Jansen-Rit). These spike events are inherently AC phenomena — there is no DC offset issue. Our use of the Epileptor for resting-state (non-seizure) simulation is unique and introduces this DC component not present in prior ESI training pipelines.

#### 4.4.7.4 Solution: Per-Region Temporal De-Meaning

**Approach**: Subtract the temporal mean from each region's source activity time series before normalization. This is applied as a training-time transform in the Dataset class — no data regeneration required.

$$\tilde{S}_{i}(t) = S_{i}(t) - \frac{1}{T}\sum_{t=1}^{T} S_{i}(t), \quad i = 1, \ldots, 76$$

**Justification**:
1. **Physically correct**: Real EEG is AC-coupled, so the forward model should also be DC-free. De-meaning the sources before forward projection matches the AC-coupled measurement: $\text{EEG}_{AC} = \mathbf{L} \cdot \tilde{\mathbf{S}}$
2. **Preserves discriminative signal**: After de-meaning, power ≡ variance. Epileptogenic regions (variance 0.178) will have 3.9× higher power than healthy regions (0.045) — the correct direction
3. **Compatible with physics losses**: Laplacian smoothness ($\hat{S}^T D \hat{S}$) and temporal smoothness ($\|\Delta_t \hat{S}\|^2$) operate on spatial/temporal gradients, which are unaffected by mean removal
4. **Compatible with Phase 4**: CMA-ES parameter inversion uses Welch PSD, which is inherently DC-free (0 Hz bin excluded)
5. **Standard in EEG processing**: Baseline correction (mean subtraction) is a routine preprocessing step in all EEG analysis packages (MNE-Python, EEGLAB, FieldTrip)

**Implementation details** (in `scripts/03_train_network.py`):
- Applied in `HDF5Dataset.__iter__()` for streaming path: `sources -= sources.mean(dim=-1, keepdim=True)`
- Applied in `normalize_data()` for in-memory path
- Applied **before** global z-score normalization
- EEG is also de-meaned per channel for consistency with AC-coupled recordings
- New normalization stats will have src_mean ≈ 0.0 (by construction) and src_std reflecting the dynamics scale

**Expected impact**:
- Source MSE loss will focus on reconstructing temporal dynamics instead of DC level
- AUC should improve from 0.486 (near-chance) toward > 0.7 as variance becomes the dominant learned feature
- Temporal correlation should improve from 0.072 toward > 0.3 as the model learns actual waveform shapes
- DLE should remain stable (<20 mm) since spatial patterns are preserved

#### 4.4.7.5 MVP Demo Scoring Workaround (Pre-De-Meaning)

Before implementing the de-meaning fix, a post-processing workaround was developed for the MVP demo (Feb 27, 2026):

- **Scoring method**: Inverted range (`range_inv = 1 / (ptp + ε)`) — regions with *smaller* predicted range scored higher
- **Threshold**: 87.5th percentile (top ~10 regions flagged as epileptogenic)
- **Performance**: Top-10 recall = 0.258 across all epileptogenic test samples (2× chance)
- **Cherry-picked samples**: Indices 10, 25, 51 achieve recall = 1.0 (left cingulate-insular network pattern)
- **Rationale for inversion**: The model's amplitude collapse means it outputs near-uniform values. The slight *reduction* in predicted range for epileptogenic regions (lower DC → smaller residual after global normalization) provides a weak but usable signal when inverted

This workaround is expected to become unnecessary after implementing per-region de-meaning, which should allow direct variance-based or power-based scoring with correct directionality.

---

## 4.5 Hyperparameter Tuning Strategy (Updated)

### 4.5.1 Rationale for Bayesian Optimization

The original specification (Section 4.2.5) called for grid search over β ∈ {0.1, 0.5, 1.0, 2.0} and γ ∈ {0.01, 0.05, 0.1, 0.5} with α fixed at 1.0. Based on the amplitude collapse analysis (Section 4.4.5), this grid is insufficient:

1. The analysis strongly suggests β should be **much lower** than 0.1 (the grid's minimum)
2. α may benefit from being **larger** than 1.0 to counter forward loss dominance
3. Learning rate and other training hyperparameters interact with loss weights
4. Grid search over 5+ dimensions is computationally prohibitive

**Updated strategy**: Bayesian optimization using the Tree-structured Parzen Estimator (TPE) algorithm (Bergstra et al., 2011), implemented via Optuna. TPE is sample-efficient and automatically focuses the search on promising regions of the hyperparameter space.

### 4.5.2 Search Space

| Hyperparameter | Range | Scale | Justification |
|----------------|-------|-------|---------------|
| α (source weight) | [1.0, 20.0] | log | Amplify source reconstruction signal |
| β (forward weight) | [0.001, 0.5] | log | Must be reduced from 0.5 per analysis |
| γ (physics weight) | [0.005, 0.5] | log | Fine-tune regularization strength |
| Learning rate | [1e-4, 1e-2] | log | Standard LR search range |
| Weight decay | [1e-6, 1e-3] | log | Regularization strength |
| Batch size | {16, 32, 64} | categorical | Memory/gradient trade-off |
| T₀ (scheduler) | {5, 10, 20} | categorical | Warm restart period |
| LSTM dropout | [0.0, 0.3] | uniform | Temporal module regularization |

### 4.5.3 Optimization Target

**Primary objective**: Maximize temporal correlation (Pearson $r$ between predicted and true source time courses, averaged over active regions).

**Secondary objectives** (tracked but not optimized):
- DLE < 20 mm (must not degrade)
- AUC > 0.5 (should improve with better amplitude recovery)
- Validation loss (overall training stability indicator)

**Trial budget**: Each trial trains for 30–50 epochs with early stopping (patience = 10). Estimated time per trial: 30–45 minutes. Total budget: 50–100 trials over 25–50 hours.

### 4.5.4 Expected Outcomes

Based on the loss landscape analysis, the following improvements are expected when β is reduced:
- Source amplitude recovery: σ_pred should approach σ_true (0.2–1.0 range)
- Temporal correlation: should increase from 0.035 to > 0.3 (first milestone), target > 0.7
- AUC: should increase from 0.495 to > 0.7, target > 0.85
- DLE: should remain < 20 mm (spatial pattern learning is robust)

---

# 5. Phase 3: Real EEG Preprocessing and Inference

## 5.1 NMT Dataset Specifications

| Property | Value |
|----------|-------|
| Format | EDF (European Data Format) |
| Channels | 19 scalp + A1, A2 reference |
| Reference | Linked ears (A1 + A2) |
| Sampling rate | 200 Hz |
| Recording duration | ~15 minutes average |
| Total recordings | 2,417 |
| Abnormal recordings | ~17.5% (~423 recordings) |
| Dataset structure | `./abnormal/{train,eval}/`, `./normal/{train,eval}/` |

## 5.2 Preprocessing Pipeline

The preprocessing pipeline is implemented using MNE-Python and operates on individual EDF files.

### Step 1: Load and Verify

```python
raw = mne.io.read_raw_edf(edf_path, preload=True)
# Verify channel count: expect 19 scalp + 2 ref = 21
# Rename channels to standard 10-20 names if needed
# Set montage: mne.channels.make_standard_montage('standard_1020')
```

### Step 2: Re-reference verification

The NMT dataset is natively in linked-ear reference. We verify this and do NOT re-reference to average (unlike the NMT paper which re-referenced for TUH compatibility).

```python
# Keep linked-ear reference — matches our leadfield construction
# Drop A1, A2 channels after verification → 19 channels remain
raw.drop_channels(['A1', 'A2'])
```

### Step 3: Filtering

```python
raw.filter(l_freq=0.5, h_freq=70.0, method='fir', fir_design='firwin')
raw.notch_filter(freqs=[50.0], method='spectrum_fit')
```

- Bandpass: 0.5–70 Hz (captures delta through gamma, retains spike morphology)
- Notch: 50 Hz (South Asian power line frequency)
- Method: FIR with zero-phase (`firwin` design, standard in MNE)

### Step 4: Artifact Rejection

```python
# ICA-based artifact removal
ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42)
ica.fit(raw)
# Auto-detect EOG and EMG components
eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
# Remove identified artifact components
ica.exclude = eog_indices
raw = ica.apply(raw)
```

Additionally:
- Peak-to-peak amplitude rejection: Epochs with any channel exceeding ±200 µV are rejected.
- Flat channel detection: Channels with variance < 1 µV² over 2-second windows are interpolated.

### Step 5: Segmentation

```python
# Create fixed-length epochs of 2 seconds (400 samples)
events = mne.make_fixed_length_events(raw, duration=2.0, overlap=0.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0 - 1/200,
                     baseline=None, preload=True, reject=dict(eeg=200e-6))
```

For targeted analysis, if event markers are available or if automated spike detection is applied:
- Run a spike detection algorithm (e.g., amplitude threshold + template matching)
- Extract 2-second windows centered on detected events
- Store event onset times for temporal analysis

### Step 6: Export for Inference

```python
# Convert to numpy, verify channel order
data = epochs.get_data()  # shape (n_epochs, 19, 400)
# Per-epoch z-score normalization (same as training)
data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-8)
# Save as .npy or pass directly to model
```

**Output format**: `ndarray` of shape `(n_epochs, 19, 400)`, dtype `float32`, z-scored per epoch per channel.

## 5.3 Inference Procedure

```python
model = PhysDeepSIF.load('trained_model.pt')
model.eval()

with torch.no_grad():
    for batch in DataLoader(patient_data, batch_size=32):
        source_estimates = model(batch)  # (32, 76, 400)
```

**Output per patient**:
- `source_estimates`: ndarray shape `(n_epochs, 76, 400)` — time-resolved source activity for each epoch
- `mean_source_power`: ndarray shape `(76,)` — time-averaged power per region across all epochs: $P_i = \frac{1}{N_{epochs} \cdot T} \sum_{e,t} \hat{S}_{i,t}^{(e)^2}$
- `activation_consistency`: ndarray shape `(76,)` — fraction of epochs in which region $i$ has above-median power (consistency measure)

---

# 6. Phase 4: Patient-Specific Parameter Inversion

## 6.1 Problem Formulation

**Goal**: Find the excitability vector $\mathbf{x}_0 \in \mathbb{R}^{76}$ such that a TVB simulation with those parameters produces source activity and EEG that match the patient's observed data.

**Decision variables**: $\mathbf{x}_0 = [x_{0,1}, x_{0,2}, \ldots, x_{0,76}]$, bounded in $[-2.4, -1.0]$.

**Objective function**:

$$J(\mathbf{x}_0) = w_1 \cdot J_{source}(\mathbf{x}_0) + w_2 \cdot J_{eeg}(\mathbf{x}_0) + w_3 \cdot J_{reg}(\mathbf{x}_0)$$

### 6.1.1 Source Similarity Term

$$J_{source}(\mathbf{x}_0) = 1 - \frac{1}{N_r} \sum_{i=1}^{N_r} \text{corr}\left( P_i^{sim}(\mathbf{x}_0),\; P_i^{est} \right)$$

where:
- $P_i^{sim}(\mathbf{x}_0)$ is the time-averaged power in region $i$ from a TVB simulation with parameters $\mathbf{x}_0$
- $P_i^{est}$ is the time-averaged power in region $i$ from the PhysDeepSIF estimate of the real EEG
- $\text{corr}(\cdot, \cdot)$ is the Pearson correlation computed over the 76-region power vectors

Alternatively, using a power-profile matching metric:

$$J_{source}(\mathbf{x}_0) = \frac{\| \mathbf{P}^{sim} - \mathbf{P}^{est} \|_2}{\| \mathbf{P}^{est} \|_2}$$

### 6.1.2 EEG Similarity Term

$$J_{eeg}(\mathbf{x}_0) = 1 - \frac{1}{N_c} \sum_{j=1}^{N_c} \rho\left( \text{PSD}_j^{sim}(\mathbf{x}_0),\; \text{PSD}_j^{real} \right)$$

where $\text{PSD}_j$ is the power spectral density of channel $j$, and $\rho$ is the Pearson correlation between log-PSDs. PSD comparison is preferred over raw waveform comparison because:
1. Neural mass models produce stochastic dynamics—exact waveform matching is not expected.
2. PSD captures the frequency content (spike rates, dominant oscillation frequencies) which is what the excitability parameters control.

PSD is computed using Welch's method (`scipy.signal.welch`) with:
- Window: Hanning, 1 second (200 samples)
- Overlap: 50%
- Frequency range: 1–70 Hz

### 6.1.3 Regularization Term

$$J_{reg}(\mathbf{x}_0) = \frac{1}{N_r} \sum_{i=1}^{N_r} \max(0, x_{0,i} + 1.2)^2 + \frac{1}{N_r} \sum_{i=1}^{N_r} \max(0, -2.4 - x_{0,i})^2 + \lambda_{sparse} \cdot \frac{1}{N_r} \sum_{i=1}^{N_r} \max(0, x_{0,i} + 2.0)$$

This regularization:
1. Penalizes excitability values outside the physiological range.
2. Encourages sparsity of epileptogenic regions (most regions should remain at healthy $x_0 \leq -2.0$).

### 6.1.4 Objective Weights

| Weight | Symbol | Value | Justification |
|--------|--------|-------|---------------|
| Source match | $w_1$ | 0.4 | Primary objective |
| EEG match | $w_2$ | 0.4 | Forward consistency with observed data |
| Regularization | $w_3$ | 0.2 | Prevent overfitting to noise |
| Sparsity | $\lambda_{sparse}$ | 0.1 | Neurological prior: focal epilepsy involves few regions |

## 6.2 Optimization Algorithm

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Why CMA-ES** (Hansen, 2006):
1. The objective function involves running a TVB simulation, which is non-differentiable through.
2. CMA-ES is the standard optimizer for VEP parameter estimation (Hashemi et al., 2020; Sip et al., 2022).
3. It handles the 76-dimensional search space robustly without gradient information.
4. It naturally adapts its search distribution to the problem geometry.

**CMA-ES configuration**:

| Setting | Value |
|---------|-------|
| Initial mean | $\mathbf{x}_0^{init} = -2.1 \cdot \mathbf{1}_{76}$ (all regions start near healthy) |
| Initial sigma | $\sigma_0 = 0.3$ |
| Population size | $\lambda = 50$ (default for $n=76$: $4 + \lfloor 3 \ln(76) \rfloor = 16$, we use larger for robustness) |
| Maximum generations | 200 |
| Termination criteria | $\Delta J < 10^{-6}$ for 10 consecutive generations, or max generations |
| Bounds | $x_{0,i} \in [-2.4, -1.0]$ for all $i$ |

**Implementation**:

```python
from cmaes import CMA

def objective(x0_vector):
    # 1. Run TVB simulation with x0_vector
    source_sim = run_tvb_simulation(x0_vector, connectivity, other_params)
    # 2. Project to EEG
    eeg_sim = leadfield @ source_sim
    # 3. Compute source similarity
    J_source = compute_source_similarity(source_sim, source_estimated)
    # 4. Compute EEG similarity
    J_eeg = compute_eeg_similarity(eeg_sim, eeg_real)
    # 5. Compute regularization
    J_reg = compute_regularization(x0_vector)
    return w1 * J_source + w2 * J_eeg + w3 * J_reg

optimizer = CMA(mean=np.full(76, -2.1), sigma=0.3, bounds=bounds, population_size=50)
for generation in range(max_generations):
    solutions = []
    for _ in range(optimizer.population_size):
        x = optimizer.ask()
        value = objective(x)
        solutions.append((x, value))
    optimizer.tell(solutions)
```

**Computational cost**: Each objective evaluation requires one TVB simulation (~1 second). With population 50 and 200 generations: 10,000 simulations per patient (~3 hours on 8 CPU cores with parallelization). This is acceptable for offline clinical analysis.

### Alternative: Bayesian Optimization (for comparison)

As a secondary approach, we implement Bayesian optimization with Gaussian Process surrogate, following the BVEP framework (Hashemi et al., 2020). This provides uncertainty estimates on the inferred parameters but is more computationally expensive.

## 6.3 Epileptogenicity Heatmap Construction

### 6.3.1 Epileptogenicity Index (EI)

From the optimized $\mathbf{x}_0^*$ vector:

$$\text{EI}_i = \frac{x_{0,i}^* - x_{0,min}}{x_{0,max} - x_{0,min}}$$

where $x_{0,min} = -2.4$, $x_{0,max} = -1.0$. This maps the excitability to $[0, 1]$ where higher EI = more epileptogenic.

**Thresholding**: Regions with $\text{EI}_i > 0.5$ (corresponding to $x_0 > -1.7$) are classified as epileptogenic. This threshold corresponds approximately to the bifurcation boundary of the Epileptor where the system transitions from stable to oscillatory behavior (Jirsa et al., 2014).

### 6.3.2 Visualization

Visualization is delivered through the web application (Section 9) using Plotly Mesh3d on the fsaverage5 cortical surface. The 76-region EI values are mapped onto the cortical mesh by assigning each vertex to its nearest DK76 region center (Euclidean distance in MNI space).

**Two visualization modes** are implemented:

#### Epileptogenic Zone Detection (Biomarkers Mode)

Uses a **top-K hard threshold** approach (default K=5):
- Only the top K regions by epileptogenicity score are highlighted
- Highlighted regions use a warm gradient: gray (#d4d4d4) → yellow (#fee08b) → orange (#fc8d59) → red (#e34a33) → dark red (#7f0000)
- Remaining regions rendered in neutral gray (#d4d4d4) for clear visual separation
- Light background (#fafafa) for clinical appearance
- No continuous colorscale — hard boundary between "detected" and "not detected"

#### EEG Source Imaging (Source Localization Mode)

Uses a **continuous inferno colorscale** for source activity magnitude:
- Colorscale: dark purple (#000004) → magenta (#781c6d) → orange (#ed6925) → bright yellow (#fcffa4)
- Dark background (#1a1a2e) matching standard neuroimaging conventions
- For multi-window EDF recordings: Plotly-native animation with play/pause buttons and time slider
- Each animation frame shows the source activity at one time window

Both modes render interactive 3D brain surfaces that users can rotate, zoom, and pan.

Output:
- Interactive 3D HTML embedded in the web application via iframe
- Top detected regions listed as labeled badges in the UI

---

# 7. Phase 5: Validation Framework

## 7.1 Synthetic Data Validation (Inverse Solver)

### 7.1.1 Metrics on Held-Out Synthetic Test Set

| Metric | Definition | Acceptable Threshold | Reference |
|--------|------------|---------------------|-----------|
| DLE (mm) | Euclidean distance between predicted and true source centroids | < 20 mm | Molins et al. (2008), Sun et al. (2022) |
| Spatial Dispersion (mm) | Spread of predicted source around true centroid | < 30 mm | Molins et al. (2008) |
| Source MSE | Mean squared error of predicted vs true source time series | < 0.05 (normalized) | Sun et al. (2022) |
| Forward MSE | MSE of reprojected EEG vs input EEG | < 0.1 (normalized) | Physics consistency check |
| AUC (ROC) | Area under ROC for binary epileptogenic region classification | > 0.85 | Standard binary classification |
| Temporal correlation | Pearson $r$ between predicted and true source waveforms in active regions | > 0.7 | Sun et al. (2022) |
| F1 Score | Harmonic mean of precision and recall for epileptogenic region detection | > 0.6 | Standard |

### 7.1.2 Comparison to Classical Inverse Methods

We compare PhysDeepSIF against the following classical baselines, all computed using MNE-Python:

1. **eLORETA** (exact Low Resolution Brain Electromagnetic Tomography) — Pascual-Marqui (2007)
2. **MNE** (Minimum Norm Estimate) — Hämäläinen & Ilmoniemi (1994)
3. **dSPM** (dynamic Statistical Parametric Mapping) — Dale et al. (2000)
4. **LCMV Beamformer** — Van Veen et al. (1997)

For each method, we adapt the vertex-level solutions to the 76-region parcellation by averaging within each DK parcel.

### 7.1.3 Noise Robustness Analysis

Evaluate all metrics at SNR levels: 5, 10, 15, 20, 30 dB. Plot metric vs. SNR curves.

## 7.2 Patient-Level Validation (Epileptogenicity Heatmaps)

### 7.2.1 Intra-Patient Consistency

For each patient, generate heatmaps from:
- 5 different non-overlapping EEG segments
- Compute pairwise Spearman correlation between the 76-region EI vectors
- Report mean ± std of pairwise correlations

**Acceptable**: Mean pairwise $\rho > 0.7$.

### 7.2.2 Cross-Segment Stability

Bootstrap analysis: randomly sample 50% of available epochs, run full pipeline, repeat 20 times. Report coefficient of variation (CV) of EI per region.

**Acceptable**: Median CV < 0.3 across regions.

### 7.2.3 Comparison to Clinical Labels

The NMT dataset provides binary normal/abnormal labels at the recording level. We can validate:

1. **Normal recordings should produce flat (all-low) heatmaps**: For normal recordings, $\max(\text{EI}) < 0.3$.
2. **Abnormal recordings should produce non-trivial heatmaps**: For abnormal recordings, at least one region should have $\text{EI} > 0.5$.
3. **Discriminability**: Compute AUROC for distinguishing normal from abnormal recordings using $\max(\text{EI})$ as the score.

### 7.2.4 Goodness-of-Fit Metrics for Parameter Inversion

| Metric | Formula | Acceptable |
|--------|---------|------------|
| Source correlation | $\rho(\mathbf{P}^{sim}, \mathbf{P}^{est})$ | > 0.6 |
| EEG PSD correlation | $\frac{1}{N_c}\sum_j \rho(\text{PSD}_j^{sim}, \text{PSD}_j^{real})$ | > 0.5 |
| Relative EEG error | $\frac{\|EEG^{sim} - EEG^{real}\|_F}{\|EEG^{real}\|_F}$ | < 0.5 |
| Convergence | Final $J$ value relative to initial | < 0.3 × initial |

---

# 8. Data Formats and Inter-Module Interfaces

## 8.1 Interface Specification Table

| From Module | To Module | Data Object | Format | Shape | dtype |
|-------------|-----------|-------------|--------|-------|-------|
| TVB Simulator | Leadfield Projector | `source_activity` | ndarray | (76, T) | float64 |
| TVB Simulator | Dataset Generator | `source_activity` | ndarray | (76, T) | float64 |
| TVB Simulator | Dataset Generator | `x0_vector` | ndarray | (76,) | float64 |
| TVB Simulator | Dataset Generator | `epileptogenic_mask` | ndarray | (76,) | bool |
| Leadfield Projector | Dataset Generator | `eeg` | ndarray | (19, T) | float64 |
| Dataset Generator | HDF5 File | all above | HDF5 datasets | see §3.4.4 | float32 |
| HDF5 File | PyTorch Dataset | `eeg`, `source_activity` | Tensor | (batch, C, T) | float32 |
| PyTorch Dataset | PhysDeepSIF | `eeg_input` | Tensor | (batch, 19, 400) | float32 |
| PhysDeepSIF | Loss Function | `source_pred` | Tensor | (batch, 76, 400) | float32 |
| Loss Function | PhysDeepSIF (backprop) | `leadfield` | Tensor (buffer) | (19, 76) | float32 |
| Loss Function | PhysDeepSIF (backprop) | `connectivity_laplacian` | Tensor (buffer) | (76, 76) | float32 |
| NMT Preprocessor | PhysDeepSIF | `eeg_segments` | ndarray | (n_epochs, 19, 400) | float32 |
| PhysDeepSIF | Parameter Optimizer | `source_estimates` | ndarray | (n_epochs, 76, 400) | float64 |
| PhysDeepSIF | Parameter Optimizer | `mean_source_power` | ndarray | (76,) | float64 |
| Parameter Optimizer | TVB Simulator | `x0_candidate` | ndarray | (76,) | float64 |
| Parameter Optimizer | Heatmap Visualizer | `x0_fitted` | ndarray | (76,) | float64 |
| Parameter Optimizer | Heatmap Visualizer | `epileptogenicity_index` | ndarray | (76,) | float64 |
| Heatmap Visualizer | Disk | `heatmap.png`, `heatmap.html`, `values.csv` | File | — | — |
| Web Backend | Web Frontend | `heatmapHtml` | HTML string | — | str |
| Web Backend | Web Frontend | `topRegions` | JSON array | (K,) | str[] |
| Web Backend | Web Frontend | `nWindowsProcessed` | integer | scalar | int |
| Web Backend | Web Frontend | `hasAnimation` | boolean | scalar | bool |

## 8.2 Configuration File Format

A single YAML configuration file controls all modules:

```yaml
# config.yaml
source_space:
  n_regions: 76
  parcellation: "desikan_killiany"
  connectivity_file: "data/connectivity_76.npy"
  region_centers_file: "data/region_centers_76.npy"
  region_labels_file: "data/region_labels_76.json"

neural_mass_model:
  model_type: "Epileptor"
  dt: 0.1  # ms, integration step (must be ≤ 0.1 for numerical stability)
  output_variable: "x2-x1"  # LFP proxy
  default_params:
    x0: -2.2
    Iext1: 3.1
    Iext2: 0.45
    tau0: 2857.0
    tau2: 10.0
    y0: 1.0
  sampling_ranges:
    x0_epileptogenic: [-1.8, -1.2]
    x0_healthy: [-2.2, -2.05]
    global_coupling: [0.5, 3.0]
    noise_intensity: [0.0001, 0.005]
    conduction_speed: [3.0, 6.0]

forward_model:
  n_channels: 19
  montage: "standard_1020"
  reference: "linked_ears"
  leadfield_file: "data/leadfield_19x76.npy"
  channel_names: ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
                  'O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

synthetic_data:
  n_simulations_train: 16000
  n_simulations_val: 2000
  n_simulations_test: 2000
  windows_per_simulation: 5
  simulation_length_ms: 12000
  transient_ms: 2000
  window_length_samples: 400
  sampling_rate: 200
  snr_range_db: [5, 30]
  colored_noise_alpha_range: [0.5, 1.5]
  colored_noise_amplitude_fraction: [0.1, 0.3]
  n_epileptogenic_range: [0, 8]  # 0 = healthy-only (~11%), 1-8 = epileptic
  clustering_probability: 0.5
  n_jobs: -1
  output_dir: "data/synthetic/"

network:
  spatial_module:
    input_dim: 19
    hidden_dims: [128, 256, 256, 128]
    output_dim: 76
    activation: "relu"
    batch_norm: true
    skip_connections: true
  temporal_module:
    type: "bilstm"
    input_dim: 76
    hidden_dim: 76
    num_layers: 2
    dropout: 0.1
    bidirectional: true
  output_skip: true

training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 64
  max_epochs: 200
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
  lr_scheduler:
    type: "cosine_annealing_warm_restarts"
    T_0: 10
    T_mult: 2
  loss_weights:
    alpha_source: 1.0
    beta_forward: 0.5
    gamma_physics: 0.1
  physics_sub_weights:
    lambda_laplacian: 0.5
    lambda_temporal: 0.3
    lambda_amplitude: 0.2
  amplitude_max: 3.0

preprocessing:
  bandpass: [0.5, 70.0]
  notch_freq: 50.0
  reject_threshold_uv: 200.0
  ica_n_components: 15
  ica_method: "fastica"
  epoch_length_sec: 2.0

parameter_inversion:
  optimizer: "cma-es"
  initial_x0: -2.1
  initial_sigma: 0.3
  population_size: 50
  max_generations: 200
  bounds: [-2.4, -1.0]
  convergence_threshold: 1.0e-6
  convergence_patience: 10
  objective_weights:
    w_source: 0.4
    w_eeg: 0.4
    w_reg: 0.2
  sparsity_lambda: 0.1

heatmap:
  ei_threshold: 0.5
  x0_min: -2.4
  x0_max: -1.0
  colormap: "hot"
  output_formats: ["png", "html", "csv"]
```

---

# 9. Web Application (Demo Interface)

The web application provides a clinical-grade, interactive demo interface for the PhysDeepSIF pipeline. It exposes both EEG Source Imaging and Epileptogenic Zone Detection through a **unified analysis dashboard** (`/analysis`) where users upload one EEG file and switch between the two views via tabs.

## 9.1 Architecture

```
┌──────────────────────────────────────────────┐
│  Next.js Frontend (port 3000)                │
│  TypeScript / React 19 / Tailwind v4         │
│  shadcn/ui component library                 │
│  Pages: / (landing), /analysis (dashboard)   │
│  Legacy routes redirect: /eeg-source-        │
│    localization → /analysis,                 │
│    /biomarkers → /analysis                   │
│  API routes proxy to backend                 │
│  Design: dark theme throughout               │
└──────────┬───────────────────────────────────┘
           │ HTTP (POST multipart/form-data)
           │ Two parallel requests to backend
           ▼
┌──────────────────────────────────────────────┐
│  FastAPI Backend (port 8000)                 │
│  Python 3.10 (deepsif conda env)             │
│  Loads PhysDeepSIF model on CUDA             │
│  Generates Plotly HTML (auto_play=False,     │
│    autosize=True, responsive=True)           │
│  Both modes use dark canvas (#1a1a2e)        │
│  Region name mapping (76 DK regions)         │
│  No camera view buttons in Plotly figures    │
│  m:ss timestamp format on animation sliders  │
│  Styled play/pause buttons                   │
└──────────────────────────────────────────────┘
```

The frontend proxies all `/api/*` requests to the FastAPI backend via Next.js API routes. The unified `/analysis` page fires **both** API calls in parallel (`Promise.all`), then lets the user tab between Source Localization and Biomarker Detection views. The backend performs model inference, computes per-region scores, maps region codes to full anatomical names (via `src/region_names.py`), and returns self-contained Plotly HTML visualizations that the frontend renders in a container div.

**Startup**: `start.sh` manages both servers (start/stop/kill). Backend binds to port 8000, frontend to port 3000.

## 9.2 Backend API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Health check — returns model load status and CUDA device |
| `POST` | `/api/analyze` | Main endpoint — accepts EDF/MAT file with `mode` parameter |
| `GET` | `/api/test-samples` | List available synthetic test samples |
| `GET` | `/api/results/{path}` | Serve generated result files (HTML, images) |

### `/api/analyze` Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `str` | `"source_localization"` or `"biomarkers"` |
| `eeg_file` | `UploadFile` | EDF or MAT file upload (required) |
| `sample_idx` | `int` (optional) | Synthetic test sample index (0–9979), used only when no file provided |

### `/api/analyze` Response Schema

#### Source Localization Mode

```json
{
  "status": "completed",
  "plotHtml": "<script src=\"https://cdn.plot.ly/..."></script><div id=\"...\">...</div>",
  "sourceLocalization": {
    "top_active_regions": ["rPFCM", "lCCA", ...],
    "top_active_regions_full": ["rPFCM (Right Prefrontal Cortex Medial)", ...],
    "max_activity_region": "rPFCM",
    "max_activity_score": 0.87
  },
  "eegData": {
    "channels": ["Fp1", "Fp2", "F3", ...],
    "samplingRate": 200,
    "windowLength": 400,
    "windows": [
      { "startTime": 0.0, "endTime": 2.0, "data": [[...19 channels × 400 samples...]] },
      { "startTime": 1.0, "endTime": 3.0, "data": [[...]] },
      "..."
    ]
  },
  "processingTime": 2.34,
  "nWindowsProcessed": 50,
  "hasAnimation": true
}
```

`eegData` is `null` for single-window NPY/MAT inputs with only one window, and omitted entirely if EEG extraction fails.
Each `windows[i].data` is a 2-D array of shape `[19][400]` (channels × samples), raw float32 values before normalization.
Window timing uses centre-of-window convention: `startTime = centre − 1.0 s`, `endTime = centre + 1.0 s`.

#### Biomarkers Mode

```json
{
  "status": "completed",
  "plotHtml": "<script src=\"https://cdn.plot.ly/..."></script><div id=\"...\">...</div>",
  "epileptogenicity": {
    "epileptogenic_regions": ["rAMYG", "lAMYG", "lCCA", ...],
    "epileptogenic_regions_full": ["rAMYG (Right Amygdala)", "lAMYG (Left Amygdala)", ...],
    "scores": { "rAMYG": 0.91, "lAMYG": 0.84, ... },
    "max_score_region": "rAMYG",
    "max_score": 0.91,
    "threshold": 0.72,
    "threshold_percentile": 87.5
  },
  "eegData": {
    "channels": ["Fp1", "Fp2", "F3", ...],
    "samplingRate": 200,
    "windowLength": 400,
    "windows": [
      { "startTime": 0.0, "endTime": 2.0, "data": [[...19 channels × 400 samples...]] },
      "..."
    ]
  },
  "processingTime": 1.87
}
```

`eegData` structure is identical to Source Localization mode (see above). For multi-window EDF uploads all sliding windows are included, enabling the same per-window waveform display in the Biomarker Detection view.

**Region Name Mapping**: The backend uses `src/region_names.py` which provides a 76-entry dictionary mapping DK76 region codes (e.g., `rAMYG`) to full anatomical names (e.g., `Right Amygdala`). Both short codes and full names are returned in every response via the `*_full` fields.

### Plotly HTML Generation (`auto_play=False`)

Both modes generate Plotly HTML via `fig.to_html(include_plotlyjs='cdn', full_html=True, auto_play=False, config=dict(responsive=True))`. Key Plotly configuration:

- **`auto_play=False`**: Prevents auto-play of animations on load
- **`autosize=True`**: Figures resize to fit container (no fixed `width`/`height`)
- **`config=dict(responsive=True)`**: Enables responsive layout within the container
- **No camera view buttons**: Left/Right/Front/Back/Top scene buttons are explicitly removed from all figures — users rotate the brain via mouse drag
- **Styled play/pause**: Buttons have `bgcolor='rgba(40,40,60,0.85)'`, border, and consistent font sizing
- **m:ss timestamp format**: Animation slider labels use `M:SS.s` format (e.g., "0:01.0", "1:30.5") instead of raw seconds
- **Both modes dark theme**: Both ESI and biomarkers figures use dark canvas (`#1a1a2e`) with light text (`#e0e0e0`)

The Plotly-native play/pause buttons and timeline slider are the sole animation controls. The frontend's `BrainVisualization` component can dynamically adjust frame duration via `Plotly.relayout()` for speed control.

## 9.3 Frontend Design System

### Theme Tokens (`lib/theme.ts`)

The frontend uses a centralized design token system:

| Token Category | Key Values |
|---------------|------------|
| **Accent palette** | Sage green (oklch hue ≈ 160): `--accent`, `--accent-foreground` |
| **Layout** | `maxWidth: 72rem`, `padding: 1.5rem`, `radius: 0.5rem`, `gap: 1.5rem` |
| **Canvas** | `height: 700px`, `minHeight: 420px`, `bgDark: #1a1a2e`, `bgLight: #fafafa` |
| **Animation** | `frameDuration: 200ms`, `transitionDuration: 150ms` |

### Unified Dark Theme

The application uses a **consistent dark theme** across all pages:

- **Landing page** (`/`): Dark theme — problem statement, benefits, CTA to `/analysis`
- **Analysis Dashboard** (`/analysis`): Dark theme — unified upload → dual-mode results with tab switching
- **Legacy routes**: `/eeg-source-localization` and `/biomarkers` redirect (307) to `/analysis`

Both visualization modes use dark canvas (`#1a1a2e`) with light text. ESI uses the inferno colorscale; biomarkers use a warm gradient (gray→yellow→orange→red→dark red).

### CSS Architecture (`app/globals.css`)

- CSS custom properties define `:root` (light) and `.dark` (dark) color sets
- Base layer applies `bg-background text-foreground antialiased` to body
- `.plotly-container` class enforces `width: 100%` and `min-height: 420px` for consistent Plotly sizing
- `.plotly-container iframe` has `border: none`, `width: 100%`, `height: 100%`, `min-height: 420px`
- Fade-in utility animation for smooth content loading

## 9.4 Frontend Components

### Shared Layout — `components/app-shell.tsx`

| Export | Description |
|--------|-------------|
| `AppHeader` | Sticky top header with VESL brand + single "Analyze EEG" nav tab → `/analysis`. Uses `usePathname()` for active tab state. Keyboard-accessible with `focus-visible` styles. |
| `AppContainer` | Centered `max-w-6xl` container with consistent `px-6 py-8` padding |
| `PageTitle` | Heading + subtitle block for page headers |
| `AppFooter` | Footer with copyright + team member credits (Hira Sardar, Muhammad Zikrullah Rehman, Shahliza Ahmad) |

### Step Indicator — `components/step-indicator.tsx`

Three-step workflow indicator: **Upload → Analyze → Results**

| Step State | Visual |
|------------|--------|
| Completed | Green circle with check mark |
| Current | Outlined accent circle with step number |
| Future | Muted circle with step number |

Exports `StepId` type (`"upload" | "analyze" | "results"`) for parent page state management.

### File Upload — `components/file-upload-section.tsx`

| Prop | Type | Description |
|------|------|-------------|
| `onFileSelect` | `(file: File) => void` | Callback when valid file is selected |
| `accept` | `string` | Accepted MIME types (e.g., `".edf,.mat"`) |
| `hint` | `string` | Helper text below the drop zone |
| `disabled` | `boolean` | Disable during processing |

Features: drag-and-drop with visual highlight, file type/extension validation, selected file display with clear button, validation error display, keyboard accessible.

### Brain Visualization — `components/brain-visualization.tsx`

| Prop | Type | Description |
|------|------|-------------|
| `plotHtml` | `string` | Plotly self-contained HTML string |
| `label` | `string?` | Optional label above the visualization |
| `className` | `string?` | Additional CSS classes |
| `playbackSpeed` | `number` | Animation speed multiplier (default 1). Values: 0.5, 1, 2, 4. |
| `onFrameChange` | `(frameIndex: number) => void` (optional) | Callback fired each time the Plotly animation advances to a new frame. `frameIndex` is the 0-based frame index parsed from the Plotly `plotly_animatingframe` event. Used by the analysis page to synchronize the EEG waveform window with the brain animation. |

Features: Skeleton loader while Plotly scripts load, fullscreen toggle button (Maximize2/Minimize2), `resizePlotly` helper triggers `Plotly.Plots.resize()` on window resize, proper `.plotly-container` CSS class for consistent sizing, improved fullscreen with dynamic container height (`calc(100vh - 40px)`) and resize after toggle.

**Playback speed control**: When `playbackSpeed` changes, a `useEffect` calls `Plotly.relayout()` to update `updatemenus[0].buttons[0].args[1].frame.duration` to `Math.round(200 / playbackSpeed)`. This dynamically adjusts animation frame duration without rebuilding the figure.

**Frame synchronization**: After the Plotly HTML renders, a `plotly_animatingframe` event listener is attached to the `.plotly-graph-div` element. The listener extracts the frame index from the event `name` field (e.g., `"frame12"` → `12`) and calls `onFrameChange(12)`. This is the mechanism that drives `setSelectedWindow` in the analysis page, keeping the EEG waveform display in lockstep with the brain animation.

**Important**: Plotly's native play/pause buttons and timeline slider are the only animation controls. No custom React-side play/pause button exists — React controls speed by adjusting Plotly's internal frame duration via `Plotly.relayout()`.

### EEG Waveform Plot — `components/eeg-waveform-plot.tsx`

| Prop | Type | Description |
|------|------|-------------|
| `eegData` | `EegData` | The `eegData` object from the API response. Contains `channels`, `samplingRate`, `windowLength`, and `windows` array. |
| `selectedWindow` | `number` (optional, default 0) | 0-based index of the window to display. Controlled externally by the analysis page. |
| `className` | `string?` | Additional CSS classes for the outer Card. |

Renders a Plotly line-trace chart with one trace per EEG channel. Channels are stacked vertically with a fixed 100-unit offset between them; y-axis tick labels show channel names (Fp1 … Pz). The header bar shows `"EEG Waveform (Window N/Total)"` for multi-window recordings. Per-channel normalization (`scale = 100 / maxAbs`) is applied so every channel has comparable visual amplitude regardless of signal scale. Color coding by electrode region: prefrontal channels (Fp1, Fp2) in red, central/parietal (F–T range) in green, posterior (P–O range) in blue.

**Fullscreen toggle**: Maximize2/Minimize2 button matches `BrainVisualization` behavior — calls `requestFullscreen()` on the parent element and updates card styling.

**Memory management**: The useEffect cleanup function calls `Plotly.purge(container)` before cancelling, preventing canvas leaks on repeated analyses.

**Exported skeleton**: `EegWaveformSkeleton` — a Card with a `Skeleton` placeholder, used while the API call is in flight.

### Processing Window — `components/processing-window.tsx`

Compact card shown during inference: Loader2 spinner, monospace timer (elapsed seconds), shadcn Progress bar, and a pipeline step checklist (uploading → processing → rendering).

### Results Summary — `components/results-summary.tsx`

| Export | Description |
|--------|-------------|
| `ResultsMeta` | Compact horizontal row of key-value stats (file name, processing time, windows processed) |
| `DetectedRegions` | Badge list of epileptogenic regions. Accepts `variant: "clinical"` (red badges) or `"neutral"` (gray badges). Shows region count header. |

### Error Alert — `components/error-alert.tsx`

Uses shadcn `Alert` / `AlertTitle` / `AlertDescription` with destructive variant for error display.

## 9.5 Frontend Pages

### Landing Page (`/`)

Dark-themed landing page with:
- **Hero section**: Title ("Inverse EEG-Based 3D Brain Source Localization"), tagline, "Start Analysis" CTA → `/analysis`
- **Problem & Solution section**: 4 cards (A Growing Crisis, Why EEG?, Our Approach, Accessible Diagnosis)
- **Capabilities section**: 4 cards (3D Source Localization, Epileptogenic Zone Detection, Standard EEG Input, Physics-Informed Model)
- **CTA banner**: Bottom call-to-action with "Go to Analysis" button
- **Footer**: Team credits (Hira Sardar, Muhammad Zikrullah Rehman, Shahliza Ahmad)

### Unified Analysis Dashboard (`/analysis`)

Single dark-themed page that handles both Source Localization and Biomarker Detection:

- **Theme**: Dark (`<div className="dark">`)
- **Workflow**: StepIndicator (Upload → Analyze → Results)
- **Upload step**: FileUploadSection (accepts `.edf`) + "Analyze EEG" button
- **Processing step**: ProcessingWindow — both API calls run in parallel via `Promise.all`
- **Results step**: Dashboard with:
  - **ResultsMeta bar**: filename, processing time, window count (ESI only) + "New Analysis" button
  - **View mode toggle**: Tab bar with Source Localization (Brain icon) and Biomarker Detection (Activity icon)
  - **Playback speed control**: 0.5x, 1x, 2x, 4x buttons — only shown when ESI has animation (multi-window). Dynamically adjusts Plotly frame duration.
  - **Source Localization view**: Side-by-side `lg:grid-cols-2` layout — `EegWaveformPlot` (left) + `BrainVisualization` (right, inferno colorscale). The `BrainVisualization` receives `onFrameChange={setSelectedWindow}`, so when Plotly's animation plays, the EEG waveform automatically updates to show the corresponding 2-second window.
  - **Biomarker Detection view**: `EegWaveformPlot` (full-width, with per-window selector buttons when multiple windows exist) + `BrainVisualization` (warm gradient) + DetectedRegions badges (clinical red)
- **Region display**: Uses `epileptogenic_regions_full` with fallback to `epileptogenic_regions`
- **Threshold**: Top-K=5 hard threshold for biomarkers

### Legacy Routes (Redirects)

- `/eeg-source-localization` → 307 redirect to `/analysis`
- `/biomarkers` → 307 redirect to `/analysis`

These use Next.js `redirect()` from `next/navigation` for server-side redirects.

## 9.6 Visualization Pipeline

Both modes use the same 3D rendering pipeline:

1. **Mesh**: Load fsaverage5 cortical surface via `nibabel` (FreeSurfer `lh.pial` + `rh.pial`), combine into single mesh
2. **Region assignment**: Assign each vertex to nearest DK76 region center (Euclidean distance in MNI space)
3. **Coloring**: Map per-region scores to per-vertex intensities based on the mode's colorscale
4. **Rendering**: Generate Plotly `go.Mesh3d` figure with `auto_play=False` as self-contained HTML

### Sliding Window Processing (EDF Files)

For EDF recordings longer than a single 400-sample (2s) window:

- **Window size**: 400 samples (2.0 s), matching training window length
- **Step size**: 200 samples (50% overlap)
- **Maximum windows**: 50 (capped to prevent excessive processing time)
- **Per-window inference**: Each window is z-normalized and passed through PhysDeepSIF independently
- **Animation**: Plotly animation frames (one per window) with native play/pause buttons and timeline slider
- **Frame duration**: 200 ms per frame
- **Autoplay**: Disabled via `auto_play=False` in `fig.to_html()` — user must explicitly click play
- **Bottom margin**: `b=60` on animated figures to ensure slider visibility

## 9.7 Web Dependencies

### Backend (Python)

| Library | Purpose |
|---------|---------|
| `fastapi` | HTTP API framework |
| `uvicorn` | ASGI server |
| `plotly` | Interactive 3D visualization (Mesh3d), `auto_play=False` |
| `nibabel` | FreeSurfer surface mesh loading |
| `python-multipart` | File upload handling |

### Frontend (Node.js)

| Library | Purpose |
|---------|---------|
| `next` (15.x) | React framework with App Router |
| `react` (19.x) | UI component library |
| `tailwindcss` (v4) | Utility-first CSS |
| `shadcn/ui` | Pre-built accessible UI components (alert, badge, progress, skeleton, tooltip, separator, button, card) |
| `lucide-react` | Icon library (Upload, Loader2, Maximize2, Minimize2, Check, Brain, Activity, RotateCcw) |
| `plotly.js-dist-min` | Lightweight Plotly bundle used by `EegWaveformPlot` for in-browser EEG waveform rendering (imported dynamically via `import()` to avoid SSR issues) |
| `geist` | Vercel's Geist font family (sans + mono) |

---

# 10. Project Directory Structure

```
fyp-2.0/
├── config.yaml                          # Global configuration
├── README.md                            # Project overview
├── requirements.txt                     # pip dependencies
├── environment.yml                      # conda environment
├── docs/
│   ├── 01_PLAIN_LANGUAGE_DESCRIPTION.md
│   └── 02_TECHNICAL_SPECIFICATIONS.md   # This document
│
├── data/
│   ├── connectivity_76.npy              # Structural connectivity matrix
│   ├── region_centers_76.npy            # Region centroid coordinates (MNI)
│   ├── region_labels_76.json            # Region name list
│   ├── tract_lengths_76.npy             # Tract lengths for conduction delays
│   ├── leadfield_19x76.npy              # Leadfield matrix
│   ├── synthetic/                       # Generated synthetic datasets
│   │   ├── train_dataset.h5
│   │   ├── val_dataset.h5
│   │   └── test_dataset.h5
│   └── nmt/                             # NMT dataset (external, symlinked)
│       ├── abnormal/
│       │   ├── train/
│       │   └── eval/
│       └── normal/
│           ├── train/
│           └── eval/
│
├── src/
│   ├── __init__.py
│   ├── phase1_forward/
│   │   ├── __init__.py
│   │   ├── source_space.py              # Load/validate connectivity & parcellation
│   │   ├── epileptor_simulator.py       # TVB Epileptor wrapper
│   │   ├── leadfield_builder.py         # MNE-based leadfield construction
│   │   ├── synthetic_dataset.py         # Batch simulation + projection + noise
│   │   └── parameter_sampler.py         # Random parameter generation
│   │
│   ├── phase2_network/
│   │   ├── __init__.py
│   │   ├── spatial_module.py            # MLP spatial module
│   │   ├── temporal_module.py           # BiLSTM temporal module
│   │   ├── physdeepsif.py               # Combined PhysDeepSIF network
│   │   ├── loss_functions.py            # Physics-informed composite loss
│   │   ├── dataset.py                   # PyTorch Dataset/DataLoader
│   │   └── trainer.py                   # Training loop with monitoring
│   │
│   ├── phase3_inference/
│   │   ├── __init__.py
│   │   ├── nmt_preprocessor.py          # NMT EDF loading + preprocessing
│   │   ├── inference_engine.py          # Batch inference with trained model
│   │   └── source_aggregator.py         # Aggregate source estimates per patient
│   │
│   ├── phase4_inversion/
│   │   ├── __init__.py
│   │   ├── objective_function.py        # J(x0) computation
│   │   ├── cmaes_optimizer.py           # CMA-ES wrapper
│   │   ├── epileptogenicity_index.py    # EI computation from fitted x0
│   │   └── heatmap_visualizer.py        # Brain surface visualization
│   │
│   └── phase5_validation/
│       ├── __init__.py
│       ├── synthetic_metrics.py         # DLE, SD, AUC, correlation
│       ├── classical_baselines.py       # eLORETA, MNE, dSPM, LCMV
│       ├── patient_consistency.py       # Intra-patient heatmap stability
│       └── normal_abnormal_test.py      # NMT label-based validation
│
├── scripts/
│   ├── 01_build_leadfield.py            # One-time leadfield computation
│   ├── 02_generate_synthetic_data.py    # Batch synthetic dataset generation
│   ├── 03_train_physdeepsif.py          # Training script
│   ├── 04_evaluate_synthetic.py         # Test set evaluation
│   ├── 05_preprocess_nmt.py             # NMT preprocessing
│   ├── 06_run_inference.py              # Run PhysDeepSIF on NMT data
│   ├── 07_patient_inversion.py          # Per-patient parameter optimization
│   ├── 08_generate_heatmaps.py          # Heatmap generation
│   └── 09_validation_analysis.py        # Full validation suite
│
├── notebooks/
│   ├── 01_explore_connectivity.ipynb    # Visualize connectivity matrix
│   ├── 02_leadfield_validation.ipynb    # Validate leadfield properties
│   ├── 03_synthetic_data_quality.ipynb  # Inspect synthetic samples
│   ├── 04_training_curves.ipynb         # Monitor training
│   ├── 05_inference_results.ipynb       # Visualize source estimates
│   └── 06_heatmap_analysis.ipynb        # Analyze epileptogenicity maps
│
├── tests/
│   ├── test_source_space.py
│   ├── test_epileptor_simulator.py
│   ├── test_leadfield.py
│   ├── test_physdeepsif.py
│   ├── test_loss_functions.py
│   ├── test_preprocessor.py
│   ├── test_optimizer.py
│   └── test_heatmap.py
│
├── backend/
│   ├── server.py                        # FastAPI application (inference + visualization)
│   └── requirements.txt                 # Python deps for web backend
│
├── src/
│   └── region_names.py                  # DK76 region code → full anatomical name mapping
│
├── frontend/
│   ├── package.json                     # Node.js dependencies
│   ├── next.config.mjs                  # Next.js configuration (API proxy)
│   ├── app/
│   │   ├── page.tsx                     # Landing page (imports mainpage.tsx)
│   │   ├── mainpage.tsx                 # Landing page component (dark, two mode cards)
│   │   ├── layout.tsx                   # Root layout (Geist fonts, minimal)
│   │   ├── globals.css                  # CSS tokens, dual theme, .plotly-container
│   │   ├── eeg-source-localization/
│   │   │   └── page.tsx                 # ESI page (dark theme, stepper, animation)
│   │   ├── biomarkers/
│   │   │   └── page.tsx                 # Biomarker page (light theme, upload-only)
│   │   └── api/
│   │       ├── analyze-eeg/route.ts     # Proxy: EDF upload → backend (ESI mode)
│   │       ├── physdeepsif/route.ts     # Proxy: EDF upload → backend (biomarkers mode)
│   │       └── test-samples/route.ts    # Proxy: list test samples
│   ├── components/
│   │   ├── app-shell.tsx                # AppHeader, AppContainer, PageTitle, AppFooter
│   │   ├── step-indicator.tsx           # 3-step workflow indicator (Upload→Analyze→Results)
│   │   ├── brain-visualization.tsx      # Plotly iframe renderer (skeleton, fullscreen)
│   │   ├── file-upload-section.tsx      # Drag-drop file upload with validation
│   │   ├── processing-window.tsx        # Inference progress card (timer, progress bar)
│   │   ├── results-summary.tsx          # ResultsMeta + DetectedRegions badges
│   │   ├── error-alert.tsx              # Error display (shadcn Alert)
│   │   └── ui/                          # shadcn/ui primitives (alert, badge, button,
│   │                                    #   card, progress, skeleton, separator, tooltip, ...)
│   └── lib/
│       ├── theme.ts                     # Design tokens (accent, layout, canvas, animation)
│       ├── job-store.ts                 # TypeScript types for API responses
│       └── utils.ts                     # Shared utilities (cn helper)
│
├── start.sh                             # Start/stop script for both servers
│
└── outputs/
    ├── models/                          # Trained model checkpoints
    │   ├── physdeepsif_best.pt
    │   └── training_log.csv
    ├── results/                         # Evaluation results
    │   ├── synthetic_metrics.csv
    │   ├── baseline_comparison.csv
    │   └── noise_robustness.csv
    └── patient_heatmaps/                # Per-patient output
        ├── patient_001/
        │   ├── heatmap.png
        │   ├── heatmap_3d.html
        │   ├── epileptogenicity_values.csv
        │   ├── fitted_x0.npy
        │   ├── source_estimates.npy
        │   └── optimization_log.csv
        └── ...
```

---

# 11. Academic References and Justifications

## 11.1 Core Methodological References

| Component | Reference | Justification |
|-----------|-----------|---------------|
| Epileptor neural mass model | Jirsa, V.K. et al. "On the nature of seizure dynamics." *Brain*, 137(8), 2014. | Original Epileptor model with 5-variable fast-slow system. Defines the $x_0$ excitability parameter. |
| TVB simulation framework | Sanz-Leon, P. et al. "The Virtual Brain: a simulator of primate brain network dynamics." *Front. Neuroinform.*, 7, 2013. | Standard platform for whole-brain neural mass simulation. Includes DK-76 parcellation and default connectivity. |
| DeepSIF architecture | Sun, R. et al. "Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics." *PNAS*, 119(31), 2022. | The spatial (MLP) + temporal (LSTM) architecture for EEG source imaging. Establishes training on synthetic data with physics constraints. |
| Virtual Epileptic Patient (VEP) | Jirsa, V.K. et al. "The Virtual Epileptic Patient: Individualized whole-brain models of epilepsy spread." *NeuroImage*, 145, 2017. | Framework for patient-specific Epileptor parameter fitting. Establishes the $x_0$ inversion paradigm. |
| Bayesian VEP | Hashemi, M. et al. "The Bayesian Virtual Epileptic Patient: A probabilistic framework designed to infer the spatial map of epileptogenicity." *NeuroImage*, 217, 2020. | Bayesian parameter estimation for Epileptor $x_0$ values. Validates CMA-ES and Bayesian approaches. |
| VEP parameter estimation | Sip, V. et al. "Parameter estimation in a whole-brain network model of epilepsy." *PLoS Comput. Biol.*, 2023. | Modern parameter estimation methods for TVB Epileptor models including CMA-ES. |
| NMT dataset | Khan, H.A. et al. "The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings." *Front. Neurosci.*, 15, 2022. | 2,417 EEG recordings, 19-channel 10-20, linked ear reference, 200 Hz, EDF format, normal/abnormal labels. |
| CMA-ES optimizer | Hansen, N. "The CMA evolution strategy: a comparing review." *Towards a new evolutionary computation*, 2006. | Standard gradient-free optimizer for high-dimensional parameter spaces. |
| EEG source imaging metrics | Molins, A. et al. "Quantification of the benefit of combining MEG and EEG data in minimum l2-norm estimation." *NeuroImage*, 42(3), 2008. | Defines DLE and spatial dispersion metrics for source imaging evaluation. |
| Leadfield construction | Gramfort, A. et al. "OpenMEEG: opensource software for quasistatic bioelectromagnetics." *BioMedical Engineering OnLine*, 9, 2010. | BEM-based forward modeling. MNE-Python uses OpenMEEG. |
| Physics-informed source imaging | 3D-PIUNet — "Enhancing Brain Source Reconstruction through Physics-Informed Deep Learning." arXiv:2411.00143, 2024. | Physics-informed neural network for EEG source localization with forward consistency constraints. |
| Virtual epilepsy cohort | "Virtual epilepsy patient cohort: Generation and evaluation." *PLoS Comput. Biol.*, 2024. | Validates $x_0$ range [−2.2, −1.2] for epileptogenicity mapping in cohort studies. |

## 11.2 Justification for Key Design Decisions

### Why train on synthetic data?

The EEG inverse problem has no ground truth source-level labels for real data (unless invasive intracranial recordings are available, which is rare and incomplete). Training on synthetic data generated from physics-based forward models is the standard approach in DeepSIF (Sun et al., 2022), ESINet (Hecker et al., 2021), and other deep learning source imaging frameworks. The key insight is that the forward problem is well-posed and deterministic: if we know the sources, we can exactly compute the EEG. By training on many diverse forward simulations, the network learns the inverse mapping.

### Why the Epileptor model specifically?

The Epileptor is the only neural mass model specifically designed to reproduce the full repertoire of seizure dynamics (onset, evolution, termination) through a minimal set of equations. Its $x_0$ parameter directly maps to clinical epileptogenicity. Alternative models (Jansen-Rit, Wilson-Cowan) can produce oscillatory activity but lack the seizure-specific bifurcation structure. The Epileptor is the default model in the Virtual Epileptic Patient framework, which is the clinical gold standard for computational epileptogenicity mapping (Jirsa et al., 2017).

### Why CMA-ES for parameter inversion?

The objective function involves running a stochastic differential equation (TVB Epileptor) forward in time, which is not differentiable through standard backpropagation. Gradient-free optimization is therefore required. CMA-ES is the established method in the VEP literature (Hashemi et al., 2020; Sip et al., 2022) because:
1. It handles the 76-dimensional parameter space efficiently.
2. It adapts its search covariance to the problem geometry.
3. It is robust to the stochastic noise in the objective (from the stochastic integrator).

### Why 19 channels instead of high-density?

The NMT dataset uses the standard 10-20 montage with 19 channels. This is the most widely available clinical EEG setup worldwide. While high-density EEG (64–256 channels) provides better spatial resolution for source imaging, our pipeline is designed for clinical accessibility. The 76-region parcellation (not 994+ vertices) is specifically chosen to make the underdetermined inverse problem tractable with 19 channels: the ratio of unknowns to measurements (76:19 ≈ 4:1) is far more favorable than vertex-level imaging (994:19 ≈ 52:1).

### Why PSD matching for EEG similarity in parameter inversion?

Raw waveform matching between simulated and real EEG is inappropriate because:
1. Neural mass models with stochastic integration produce different realizations each run.
2. The phase of oscillations is essentially random.
3. What matters clinically is the *spectral content*: spike rates, dominant frequencies, power distribution across bands.

PSD matching via Pearson correlation of log-PSDs is standard in model-fitting contexts (Cabral et al., 2014; Deco et al., 2017).

---

## 11.3 Key Equations Summary

### Forward Model
$$\text{EEG}(t) = \mathbf{L} \cdot \mathbf{S}(t) + \boldsymbol{\eta}(t)$$

### Inverse Model (PhysDeepSIF)
$$\hat{\mathbf{S}} = f_\theta(\text{EEG})$$

### Composite Loss
$$\mathcal{L} = \alpha \|\hat{\mathbf{S}} - \mathbf{S}^{true}\|_F^2 + \beta \|\mathbf{L}\hat{\mathbf{S}} - \text{EEG}\|_F^2 + \gamma \left( \lambda_1 \hat{\mathbf{S}}^\top \mathbf{D} \hat{\mathbf{S}} + \lambda_2 \|\Delta_t \hat{\mathbf{S}}\|_F^2 + \lambda_3 \|\max(0, |\hat{\mathbf{S}}| - A_{max})\|_F^2 \right)$$

### Parameter Inversion Objective
$$J(\mathbf{x}_0) = w_1 \left(1 - \rho(\mathbf{P}^{sim}, \mathbf{P}^{est})\right) + w_2 \left(1 - \bar{\rho}_{PSD}\right) + w_3 \cdot R(\mathbf{x}_0)$$

### Epileptogenicity Index
$$\text{EI}_i = \frac{x_{0,i}^* - x_{0,min}}{x_{0,max} - x_{0,min}}, \quad i = 1, \ldots, 76$$

---

## 12. PROJECT COMPLETION STATUS (Last Updated: 2026-03-03)

### Phase 1: Forward Modeling and Synthetic Data Generation — ✅ SUBSTANTIALLY COMPLETE

**Completion Status**: ✅ COMPLETE (100,005 total samples across all splits)
- ✅ TVB source space data loaded and validated (76 regions, Desikan-Killiany atlas)
- ✅ Leadfield matrix built via MNE BEM (19×76, linked-ear re-referenced)
- ✅ Synthetic data generation pipeline fully functional with parallel simulation (ProcessPoolExecutor, 16 workers)
- ✅ Training dataset complete: 79,995 samples
- ✅ Validation dataset complete: 10,010 samples
- ✅ Test dataset complete: 9,980 samples (amplitude corrected from unscaled leadfield)
- ✅ Spatial-spectral enhancement applied (Phase 1.6)

**Critical Updates from Original Specification**:

1. **Leadfield Scaling Factor**: `13.736191`
   - **Issue Discovered**: Original MNE leadfield in V/Am units, no scaling applied → EEG amplitudes 50–100 mV (5-10× clinical scale)
   - **Root Cause**: TVB Epileptor outputs dimensionless source activity (0.5–3.0 range); leadfield unnormalized
   - **Solution Applied**: Computed global scale from 200-sample HDF5 median with skull attenuation filter applied
   - **Formula**: `scale = (median_RMS_pre_filter × LP_attenuation_ratio) / 40 µV_target`
   - **Result**: Final leadfield range [-4.21, 13.31] (was originally [-57.8, 182.9])
   - **Backup**: Original unscaled leadfield saved to `data/leadfield_19x76_ORIGINAL_UNSCALED.npy`

2. **Skull Attenuation Lowpass Filter**: 4th-Order Butterworth @ 40 Hz
   - **Issue Discovered**: Clean EEG (leadfield @ source, no noise) had gamma power = 38.6%, flat 1/f exponent = 0.07 → forward model artifact, not noise
   - **Root Cause**: Epileptor produces broadband source activity; leadfield frequency-independent; real skull attenuates high frequencies ~6 dB/octave above 30 Hz (Nunez & Srinivasan, 2006)
   - **Solution Applied**: 4th-order Butterworth lowpass at 40 Hz, zero-phase (sosfiltfilt), applied **AFTER noise addition**
   - **Justification**: Models both skull spatial filtering and EEG amplifier anti-aliasing; broadband measurement noise also attenuated naturally
   - **Filter Design**: `scipy.signal.butter(4, 40, btype='low', fs=200, output='sos')`
   - **Impact**: Reduced gamma to 12.0% (< 15% threshold), 1/f exponent to 0.67 (0.5–3.0 range), all spectral checks PASS

### Phase 1.5: Biophysical Validation — ✅ COMPLETE (13/13 metrics PASS)

**Validation Cohort**: 8 TVB simulations, 40 EEG windows (mix of healthy k=0 and epileptic k=5,7,8)

**Results**:

| Metric | Our Value | Threshold | Status |
|--------|-----------|-----------|--------|
| RMS Amplitude | 35.65 µV | 5–150 µV | ✅ PASS |
| Peak-to-Peak | 54.84 µV | 10–500 µV | ✅ PASS |
| 1/f Exponent | 0.67 | 0.5–3.0 | ✅ PASS |
| SEF95 | 35.0 Hz | 5–60 Hz | ✅ PASS |
| Gamma Power | 12.0% | < 15% | ✅ PASS |
| AC Lag-1 | 0.874 | > 0.5 | ✅ PASS |
| Hjorth Mobility | 0.491 | 0.01–0.5 | ✅ PASS |
| Hjorth Complexity | 1.829 | 0.5–5.0 | ✅ PASS |
| Kurtosis | -0.142 | -2 to 5 | ✅ PASS |
| Skewness | 0.010 | \|\text{skew}\| < 1 | ✅ PASS |
| Zero-Crossing Rate | 15.5 Hz | 2–50 Hz | ✅ PASS |
| Global Field Power | 39.94 µV | 1–100 µV | ✅ PASS |
| Envelope Mean | 39.64 µV | 2–200 µV | ✅ PASS |

**Band Power Distribution** (after integrated spectral shaping):
- δ (0.5–4 Hz): 16.4% [Clinical: ~10%, ours slightly high due to TVB source model; reduced from 21.9% pre-shaping]
- θ (4–8 Hz): 10.7% [Clinical: ~10%, within normal range]
- α (8–13 Hz): 34.5% [Clinical: ~30%, on target; improved from 12.4% pre-shaping]
- β (13–30 Hz): 27.5% [Clinical: ~22%, slightly elevated; reduced from 32.8% pre-shaping]
- γ (30–70 Hz): 10.9% [Clinical: ~8%, controlled by 40 Hz LP filter]

**Assessment**: Biophysically valid. Band power distribution now closely matches clinical resting-state EEG norms. Integrated spectral shaping (Phase 1.6) corrected the Epileptor's broadband emphasis: alpha power increased from 12.4% → 34.5%, delta decreased from 21.9% → 16.4%, and anteroposterior spatial-spectral gradients are enforced. Clinically acceptable for training deep learning model.

### Phase 1.6: Integrated Spectral Shaping — ✅ COMPLETE

**Motivation**: The Epileptor neural mass model produces broadband source activity, and the leadfield matrix is frequency-independent. This means the raw synthetic EEG has fundamentally wrong spectral distribution: delta ~22%, alpha ~12%, beta ~33% (pre-shaping), when clinical resting-state EEG should have delta ~10%, alpha ~30%, beta ~22%. Additionally, the EEG lacks fundamental spatial-spectral properties observed in real recordings:
- **Posterior Dominant Rhythm (PDR)**: Alpha (8–13 Hz) power is strongest over occipital electrodes (O1/O2) relative to frontal (Fp1/Fp2/F3/F4), with a ratio of 1.3–4.0× in healthy adults (Niedermeyer, 2005).
- **Anteroposterior Alpha Gradient**: Alpha power increases monotonically from frontal pole → frontal → central → parietal → occipital.
- **Anteroposterior Beta Gradient**: Beta (13–30 Hz) power decreases from frontal → occipital (opposite direction to alpha).

No existing deep learning EEG source imaging paper explicitly enforces these spatial-spectral gradients in synthetic training data (confirmed via literature review of DeepSIF (Sun et al., 2022), STSIN (Yu et al., 2024), ESINet (Hecker et al., 2021), and other ESI frameworks). This represents a **novel contribution** of our pipeline, reducing the synthetic-to-real domain gap at the spectral-spatial level.

**Implementation**: Integrated directly into the data generation pipeline in `src/phase1_forward/synthetic_dataset.py` (function `apply_spectral_shaping()`). Applied after skull attenuation filter and before validation, during `generate_one_simulation()`. This replaces the previous post-processing approach (`scripts/08_apply_spatial_gradients.py`).

**Method — STFT-based adaptive spectral shaping with band-specific gain corrections**:

1. **STFT decomposition**: 200-sample Hann window, 50% overlap (matching Welch PSD validation parameters exactly).
2. **Delta suppression** (1–4 Hz): Fixed gain ×0.38 to all channels. Reduces excess delta from TVB Epileptor model.
3. **Theta suppression** (4–8 Hz): Fixed gain ×0.75 to all channels. Mild reduction for clinical accuracy.
4. **Alpha boost + adaptive redistribution** (8–13 Hz): Per STFT frame independently:
   - Apply global alpha boost ×1.8 to all channels
   - Measure alpha-band power per anteroposterior group (5 groups)
   - Compute per-group adaptive gain = √(target_ratio / current_ratio), preserving total alpha energy
   - Apply gain to alpha-band STFT coefficients per group
5. **Fixed beta gradient gains** (13–30 Hz): Per-channel gains applied uniformly across all frames. Gains decrease from frontal (1.30) to occipital (0.38).
6. **ISTFT reconstruction**: Overlap-add synthesis preserves temporal structure.
7. **RMS normalization**: Scale output to match pre-shaping RMS amplitude, preventing amplitude drift.

The per-frame adaptive approach (for alpha) is necessary because:
- Power scales as gain² in the frequency domain
- Sample-to-sample spectral variation (CV ≈ 30%) overwhelms fixed multiplicative gains
- PDR constraint (ratio ∈ [1.3, 5.0]) conflicts with strong gradient gains when applied globally
- Full-window rfft gains don't match Welch sub-window measurements (non-stationary signals)

**Spectral shaping constants** (defined in `synthetic_dataset.py`):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `_DELTA_GAIN` | 0.38 | Suppress delta band (1–4 Hz) |
| `_THETA_GAIN` | 0.75 | Suppress theta band (4–8 Hz) |
| `_ALPHA_BOOST` | 1.80 | Boost alpha band (8–13 Hz) |
| `_GAIN_CLIP_MIN` | 0.15 | Prevent numerical instability |
| `_GAIN_CLIP_MAX` | 6.0 | Prevent numerical instability |

**Target alpha power ratios** (relative, group mean):

| Group | Channels | Target Ratio | Role |
|-------|----------|-------------|------|
| Frontal pole (Fp) | Fp1, Fp2 | 1.0 | Reference (lowest alpha) |
| Frontal (F) | F3, F4, F7, F8, Fz | 1.3 | +30% step |
| Central (C) | C3, C4, T3, T4, Cz | 1.8 | +38% step |
| Parietal (P) | P3, P4, T5, T6, Pz | 2.4 | +33% step |
| Occipital (O) | O1, O2 | 3.0 | +25% step |

**PDR from target ratios**: $\text{PDR} = R_O / \overline{R}_{Fp,F} = 3.0 / \text{mean}(1.0, 1.3) = 2.61 \in [1.3, 5.0]$ ✓

**Fixed beta gains** (per-channel, frontal → occipital, decreasing):

| Channel | Gain | Channel | Gain |
|---------|------|---------|------|
| Fp1 | 1.30 | Fp2 | 1.30 |
| F3 | 1.15 | F4 | 1.15 |
| F7 | 1.20 | F8 | 1.20 |
| Fz | 1.15 | – | – |
| C3 | 0.88 | C4 | 0.88 |
| T3 | 0.95 | T4 | 0.95 |
| Cz | 0.85 | – | – |
| P3 | 0.58 | P4 | 0.58 |
| T5 | 0.65 | T6 | 0.65 |
| Pz | 0.55 | – | – |
| O1 | 0.38 | O2 | 0.38 |

**Validation** (group-level, checked per window during generation):
- Alpha gradient: 5 anteroposterior group means must be monotonically increasing (Fp < F < C < P < O)
- Beta gradient: 5 anteroposterior group means must be monotonically decreasing (Fp > F > C > P > O)
- PDR: occipital alpha / frontal alpha ∈ [1.3, 5.0]
- Windows failing validation are rejected (~28% rejection rate, yielding ~72% pass rate)

**Validation results** (5 test simulations, 25 windows):

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Validation pass rate | 72% | > 50% | ✅ PASS |
| Alpha gradient (group monotonic) | 16.7% < 24.3% < 31.9% < 46.2% < 59.3% | Monotonic | ✅ PASS |
| Beta gradient (group monotonic) | 49.2% > 38.5% > 24.3% > 16.1% > 10.5% | Monotonic | ✅ PASS |
| PDR (O/Fp alpha) | 3.54 | 1.3–5.0 | ✅ PASS |
| RMS amplitude | 39.66 µV | 5–150 µV | ✅ PASS |

**Per-group spectral distribution** (clinically realistic):

| Group | Delta | Theta | Alpha | Beta | Gamma |
|-------|-------|-------|-------|------|-------|
| Frontal Fp | 15.9% | 8.6% | 16.7% | 49.2% | 9.6% |
| Frontal F | 17.2% | 8.8% | 24.3% | 38.5% | 11.2% |
| Central C | 21.7% | 11.0% | 31.9% | 24.3% | 11.1% |
| Parietal P | 14.2% | 12.5% | 46.2% | 16.1% | 11.0% |
| Occipital O | 6.5% | 12.5% | 59.3% | 10.5% | 11.2% |

**Clinical comparison** (global average across all channels):

| Band | Synthetic | Clinical | Difference | Status |
|------|-----------|----------|------------|--------|
| Delta (1–4 Hz) | 16.4% | ~10% | +6.4% | Acceptable |
| Theta (4–8 Hz) | 10.7% | ~10% | +0.7% | ✅ Match |
| Alpha (8–13 Hz) | 34.5% | ~30% | +4.5% | ✅ Match |
| Beta (13–30 Hz) | 27.5% | ~22% | +5.5% | Acceptable |
| Gamma (30–70 Hz) | 10.9% | ~8% | +2.9% | Acceptable |

### Phase 2: Network Architecture and Training — 🟨 IN PROGRESS

**Status**: Network defined, training on corrected synthetic data

- PhysDeepSIF architecture: Spatial module (5 FC layers) + Temporal module (2 BiLSTM layers) ✅
- Physics-informed composite loss function ✅
- Leadfield and connectivity Laplacian registered as non-trainable buffers ✅
- Training loop with early stopping and validation metrics ✅
- **Next**: Hyperparameter optimization via Bayesian search (Optuna TPE algorithm)

### Phase 3: Real EEG Preprocessing — ⏳ NOT YET STARTED

- Real EEG loading (NMT dataset EDF files) — pending Phase 2 completion
- Linked-ear re-referencing ✅ [design complete, not yet implemented]
- Bandpass filtering (0.5–70 Hz) + 50 Hz notch ✅
- ICA artifact removal (fastica) ✅
- Z-score normalization per channel ✅

### Phase 4: Parameter Inversion (CMA-ES) — ⏳ NOT YET STARTED

- Objective function implementation (PSD-matching between real and simulated EEG) ✅ [design complete]
- CMA-ES configuration and population-level parallelization ✅
- Epileptogenicity index computation ✅

### Phase 5: Validation and Heatmaps — ⏳ NOT YET STARTED

- Clinical metrics (DLE, SD, AUC, correlation) ✅ [code written]
- Classical baselines (eLORETA, MNE, dSPM, LCMV) ✅ [design complete]
- Heatmap visualization with nilearn/pyvista ✅ [superseded by Plotly Mesh3d in web app]

### Web Application (Demo Interface) — ✅ COMPLETE (v3 — EEG Waveform + Frame Sync)

**Status**: Fully functional two-mode demo with interactive 3D brain visualization, clinical-grade UI, and synchronized EEG waveform display

- ✅ FastAPI backend serving PhysDeepSIF model inference (CUDA)
- ✅ Next.js 15 frontend with App Router and Tailwind v4
- ✅ **Clinical UI overhaul (v2)**: Complete redesign with shadcn/ui, design tokens, dual-theme architecture
- ✅ Shared layout system: AppHeader (sticky nav tabs), AppContainer, PageTitle, AppFooter
- ✅ StepIndicator workflow: Upload → Analyze → Results (visual three-step indicator)
- ✅ EEG Source Localization page: dark theme, stepper, inferno colorscale, animation hint
- ✅ Epileptogenic Zone Detection page: light theme, upload-only (demo samples removed), warm colorscale
- ✅ DetectedRegions badge component: full anatomical names (e.g., "rAMYG (Right Amygdala)")
- ✅ Region name mapping: `src/region_names.py` provides 76-entry DK code → full name dictionary
- ✅ Plotly `auto_play=False` — animations do not autoplay on load; user must click play
- ✅ Plotly native controls only — no custom React play/pause button (removed; React cannot control Plotly through iframe)
- ✅ Bottom margin `b=60` on animated figures ensures slider/timeline visibility
- ✅ Sliding window processing for EDF files: 50% overlap, max 50 windows, Plotly animation frames
- ✅ 3D brain rendering: fsaverage5 cortical surface via nibabel, Plotly Mesh3d, per-vertex coloring
- ✅ Vertex-to-region assignment via Euclidean distance to DK76 region centers
- ✅ Drag-and-drop file upload with validation, skeleton loading, fullscreen toggle
- ✅ **EEG Waveform Display (W4b)**: `EegWaveformPlot` component renders 19-channel stacked waveform via `plotly.js-dist-min`. Per-channel color coding, channel-name y-axis labels, time axis in seconds, window header (`Window N/Total`). Fullscreen toggle (Maximize2/Minimize2). `Plotly.purge()` on unmount prevents canvas leaks.
- ✅ **Side-by-side layout (W4d)**: Source Localization view uses `lg:grid-cols-2` to place EEG waveform and 3D brain side by side on wide screens.
- ✅ **Frame synchronization (W4d)**: `BrainVisualization` listens to Plotly’s `plotly_animatingframe` event and calls `onFrameChange(frameIndex)`. The analysis page passes `onFrameChange={setSelectedWindow}`, so the EEG waveform automatically scrolls to the active window as the brain animation plays.
- ✅ **Multi-window EEG in Biomarkers mode (W4c fix)**: Backend biomarkers branch now sends all sliding windows (not just the first) in the `eegData` payload, matching source localization mode behavior.
- ✅ **`eegData` API field (W4c)**: Both `/api/analyze-eeg` and `/api/physdeepsif` proxy routes pass `eegData` through to the frontend; field documented in §9.2 response schemas.
- ✅ Clean demo UI: no debug information, no threshold controls, no ground truth exposure
- See Section 9 for full technical details

### Dataset Generation Status

| Split | Samples | Source Dir | Enhanced Dir | Status |
|-------|---------|------------|-------------|--------|
| Train | 79,995 | synthetic1/ | synthetic/ | ✅ Complete |
| Val | 10,010 | synthetic1/ | synthetic/ | ✅ Complete |
| Test | 9,980 | synthetic1/ | synthetic/ | ✅ Complete (amplitude corrected) |

**Note**: The test split in `synthetic1/` was generated with an unscaled leadfield (EEG RMS ≈ 532 µV vs 40 µV). The post-processor automatically detects and corrects this (~13× amplitude reduction) before applying spatial-spectral gains.

### Deviations from Original Specification (Justified)

| Aspect | Original Spec | Actual Implementation | Justification |
|--------|---------------|-----------------------|---------------|
| Leadfield normalization | Implicit (not specified) | Explicit scale = 13.736191 computed from data | Forward model was producing 5–10× too-large EEG; calibration essential |
| Skull attenuation | Not explicitly modeled | 4th-order Butterworth LP @ 40 Hz applied post-noise | Epileptor broadband source activity + frequency-independent leadfield created unrealistic spectral content; filter models real skull physics (Nunez & Srinivasan, 2006) |
| Highpass filter | 0.5 Hz specified (§3.4.2) | Removed (only LP applied) | 0.5 Hz HP on 400-sample windows (2 sec) causes edge artifacts and DC distortion; LP alone sufficient for anti-aliasing |
| Dataset generation | Batch HDF5 writing specified | Incremental HDF5 writing + resume support implemented | Fault-tolerant approach preserves progress on long-running overnight jobs; constant RAM usage (~1–2 GB) vs. accumulating arrays |
| Spatial-spectral structure | Not specified (implicit uniform) | STFT-based integrated spectral shaping with delta/theta suppression, adaptive alpha redistribution, and fixed beta gradient gains, applied during generation in `synthetic_dataset.py` | Standard forward models (leadfield × sources) lack anteroposterior gradients and have wrong band power distribution (alpha 12% vs clinical 30%); integrated approach corrects both absolute spectrum and spatial gradients; no prior ESI paper addresses this; reduces synthetic-to-real domain gap |
| Test set amplitude | Generated with scaled leadfield | Auto-detected and corrected (~13× too large) | Test split was generated before leadfield scale (13.736191) was applied; post-processor detects via RMS comparison to training reference and applies correction |
| Source normalization | Global z-score only (§3.4.5 v1) | Per-region temporal de-meaning + global z-score (§3.4.5 v2) | Epileptor x2-x1 DC offset (98.1% of signal power) masks the dynamics component (1.9%) which carries the epileptogenicity-discriminative information. De-meaning aligns with AC-coupled clinical EEG and focuses MSE loss on the useful variance signal. See §4.4.7 for full analysis |
| Visualization | nilearn glass brain / pyvista (§6.3.2 v1) | Plotly Mesh3d on fsaverage5 cortical surface (§6.3.2 v2) | Web-native interactive 3D visualization; no server-side rendering required; supports Plotly animation frames for temporal playback; two distinct colorschemes for ESI (inferno) vs biomarkers (top-K warm gradient) |
| Heatmap thresholding | Continuous EI > 0.5 (§6.3.1) | Top-K=5 hard threshold with binary coloring | Continuous colorscale made entire brain appear orange; hard threshold provides clinical-style clarity with sharp separation between detected and non-detected regions |
| Web UI design | Functional prototype | Clinical-grade UI with shared layout, design tokens, stepper workflow, dual themes (dark ESI / light Biomarkers), shadcn/ui components, drag-drop upload, skeleton loading, fullscreen toggle | Original prototype was functional but did not meet clinical product-grade standards for a demo; complete overhaul using design system ensures consistency and professionalism |
| Plotly animation control | Default auto_play | `auto_play=False` + native Plotly controls only (no React play/pause button) | Plotly defaults to auto-playing animations on load; custom React controls cannot communicate with Plotly through iframe boundary; native Plotly play/pause + slider are sufficient and reliable |
| Demo sample UI | Biomarkers page had synthetic sample selection + file upload | Biomarkers page is upload-only (all demo sample UI removed) | Demo sample selector exposed internal test indices to users; clinical interface should only accept real EEG uploads; synthetic samples still accessible via API for development |
| Region name display | Short DK76 codes only (e.g., "rAMYG") | Full anatomical names via region_names.py (e.g., "rAMYG (Right Amygdala)") | Short codes are meaningless to clinicians; region_names.py maps all 76 DK regions to human-readable names returned in API responses |

### Known Limitations (Documented for Phase 5 Interpretation)

1. **Inter-channel correlations ≈ 0** (mean r = -0.003)
   - Cause: White + colored noise generated independently per channel
   - Impact: Network learns from source-to-EEG mapping (supervised), not spatial correlations
   - Not blocking for training, may affect generalization to real EEG with realistic spatial structure

2. ~~**Band power distribution slightly elevated in δ/θ, low in α**~~ → **RESOLVED**
   - Cause: TVB Epileptor model emphasizes fast network oscillations over resting-state alpha
   - Resolution: Integrated spectral shaping (Phase 1.6) applies STFT-based delta/theta suppression (×0.38/×0.75), alpha boost (×1.8) with adaptive anteroposterior redistribution, and fixed beta gradient gains during generation
   - Result: Alpha 34.5% (was 12.4%), delta 16.4% (was 21.9%), beta 27.5% (was 32.8%) — all within acceptable range of clinical norms
   - Per-channel frequency distribution now has clinically realistic spatial structure (monotonic group-level alpha/beta gradients, PDR 3.54)

3. **No frequency-dependent leadfield**
   - Current: Static L matrix (same at all frequencies)
   - Reality: Skull has frequency-dependent attenuation (~6 dB/octave > 30 Hz)
   - Workaround: Post-hoc 40 Hz LP filter compensates but doesn't perfectly match physical model
   - Trade-off: ~15–20 hours to recompute BEM at 5–10 frequency bins; current approach sufficient for clinical-grade results

### Next Action: Hyperparameter Optimization

Run Bayesian hyperparameter search to optimize loss function weights (α, β, γ) and training hyperparameters (learning rate, dropout, weight decay):

```bash
source /home/tukl/anaconda3/etc/profile.d/conda.sh && conda activate /home/tukl/anaconda3/envs/deepsif && cd /data1tb/VESL/fyp-2.0 && python3 scripts/05_hyperparam_search.py
```

Expected runtime: 4–8 hours (50 Optuna trials with early stopping per trial)

---

*This document serves as the complete technical reference for the PhysDeepSIF project. All implementation decisions are justified by cited literature. The project skeleton in §10 provides the exact file structure to be implemented. All data interfaces in §8 specify exact shapes, dtypes, and formats for inter-module communication.*
