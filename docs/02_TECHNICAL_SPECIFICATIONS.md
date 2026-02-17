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
9. [Project Directory Structure](#9-project-directory-structure)
10. [Academic References and Justifications](#10-academic-references-and-justifications)

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
7. Validate output: reject any simulation where source_activity contains NaN or Inf (diverged simulations).

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

1. **EEG**: Each window is z-scored independently (zero mean, unit variance across time for each channel). This is per-sample normalization.
2. **Source activity**: Each window is scaled by the global maximum absolute value across all regions and time points within that window, so values lie in $[-1, 1]$.

These normalizations are applied on-the-fly in the PyTorch `Dataset.__getitem__()` method to preserve the raw data in HDF5.

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

The heatmap is rendered using `nilearn.plotting.plot_glass_brain` or `nilearn.plotting.view_surf`, mapping the 76-region EI values onto the fsaverage cortical surface via the DK atlas labels.

```python
from nilearn import plotting, datasets

# Map 76-region EI to FreeSurfer annotation
# Use nilearn's fetch_atlas_destrieux_2009 or equivalent DK atlas
# Create a NIfTI volume or surface annotation with EI values
plotting.plot_glass_brain(ei_stat_map, title='Epileptogenicity Index',
                          colorbar=True, cmap='hot', threshold=0.3)
```

Output files:
- `epileptogenicity_heatmap.png` — 2D glass brain projection
- `epileptogenicity_3d.html` — Interactive 3D surface visualization
- `epileptogenicity_values.csv` — Table of (region_name, EI_value, classification)

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

# 9. Project Directory Structure

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

# 10. Academic References and Justifications

## 10.1 Core Methodological References

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

## 10.2 Justification for Key Design Decisions

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

## 10.3 Key Equations Summary

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

*This document serves as the complete technical reference for the PhysDeepSIF project. All implementation decisions are justified by cited literature. The project skeleton in §9 provides the exact file structure to be implemented. All data interfaces in §8 specify exact shapes, dtypes, and formats for inter-module communication.*
