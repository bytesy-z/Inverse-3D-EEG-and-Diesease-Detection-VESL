# PhysDeepSIF Inverse Solver + Patient-Specific Epileptogenicity Mapping

## Plain Language Description of the Project

---

## 1. What This Project Is

This project builds a system that can look at a patient's scalp EEG recording—the kind of brainwave test done routinely in any neurology clinic—and produce a map showing which regions of the brain are most likely to be the source of epileptic activity. We call this map an **epileptogenicity heatmap**.

The key innovation is that we do not rely on classical inverse-solution mathematics alone (which are notoriously underdetermined—infinitely many source configurations can produce the same scalp EEG). Instead, we train a deep neural network to solve the inverse problem, and we constrain it with the actual physics of how brain signals propagate. Then, on a per-patient basis, we run an optimization loop that adjusts a computational brain model until its simulated EEG matches the patient's real EEG. The parameters that achieve that match tell us which brain regions had to be "over-excitable" to reproduce the patient's seizure patterns—and those regions are flagged as epileptogenic.

---

## 2. The NMT Dataset and Its Role

The **NMT Scalp EEG Dataset** is our source of real clinical EEG data. It consists of 2,417 EEG recordings from unique patients, collected using the standard international 10-20 electrode montage with linked ear reference (electrodes A1 and A2 tied together as the reference). All recordings are sampled at 200 Hz and stored in EDF format. Each recording is labeled as either **normal** or **abnormal** by qualified neurologists.

In our pipeline, the NMT dataset serves as the **real patient EEG** that we want to analyze. Specifically:

- We take the **abnormal** recordings (those flagged as having epileptiform activity).
- We preprocess them (bandpass filter, artifact reject, re-verify linked-ear reference, extract event-related segments).
- We feed those segments through our trained inverse solver to estimate which brain regions were active.
- We then run the patient-specific optimization loop on those segments to produce epileptogenicity heatmaps.

The NMT dataset is **not** used to train the inverse solver. Training happens entirely on synthetic data. The NMT data is used only at inference time (Phases 3–4).

---

## 3. How the System Works, Step by Step

### Step 1: We Build a Virtual Brain Simulator

We set up The Virtual Brain (TVB) software with a 76-region brain parcellation (based on the Desikan-Killiany atlas, subdivided into left and right hemispheres). Each of the 76 regions is modeled as a neural mass using the **Epileptor** model—a mathematical model specifically designed to capture seizure dynamics. The Epileptor has a key parameter called $x_0$ (the excitability parameter): when $x_0$ is high enough (less negative), a region will generate seizure-like activity; when $x_0$ is sufficiently negative, the region stays healthy and quiet.

The 76 regions are connected to each other through a **structural connectivity matrix**—a 76×76 matrix derived from diffusion tensor imaging (DTI) tractography that describes how strongly each pair of brain regions is physically wired together. TVB ships with a default connectivity matrix that we will use.

### Step 2: We Build a Leadfield Matrix

A leadfield matrix describes how electrical activity from each of the 76 brain regions projects to each of the 19 scalp electrodes in the 10-20 montage. Mathematically, if $\mathbf{S}(t)$ is a vector of 76 source amplitudes at time $t$, and $\mathbf{L}$ is the 19×76 leadfield matrix, then the scalp EEG at time $t$ is:

$$\text{EEG}(t) = \mathbf{L} \cdot \mathbf{S}(t)$$

We construct this leadfield using MNE-Python's boundary element method (BEM) forward model, using a template head model (fsaverage), mapping the Desikan-Killiany 76-region parcellation to the 19 electrode positions, and applying the linked-ear reference transformation. This is a one-time computation.

### Step 3: We Generate a Large Synthetic Training Dataset

We run thousands of TVB simulations, each time:

1. **Randomly choosing** which regions are epileptogenic (by setting their $x_0$ to values between −1.6 and −1.2) and which are healthy (setting $x_0$ between −2.2 and −2.1).
2. **Varying** coupling strengths, noise levels, and time constants across physiologically plausible ranges.
3. **Running** the Epileptor simulation to produce 76 source time series.
4. **Projecting** the source time series through the leadfield to get a 19-channel synthetic EEG.
5. **Adding** noise (Gaussian and colored) to the synthetic EEG to make it realistic.

For each simulation, we store:
- The 19-channel synthetic EEG (the network's **input** during training).
- The 76-region source activity (the network's **target** during training).
- The binary label vector marking which regions were set as epileptogenic (used for evaluation).

We generate on the order of 50,000–100,000 such samples, with varied seizure topographies, SNR levels, and timing.

### Step 4: We Train the PhysDeepSIF Inverse Solver

PhysDeepSIF is a deep neural network inspired by the DeepSIF architecture (Sun et al., 2022, PNAS). It has two main modules:

- A **spatial module**: a stack of fully connected layers with skip connections that takes in the 19-channel EEG at each time step and maps it toward the 76-region source space. This module learns to undo the spatial blurring caused by volume conduction.
- A **temporal module**: a stack of bidirectional LSTM layers that processes the time series in the source space to capture temporal dynamics—how source activity evolves over time.

The network is trained using a **composite loss function** that has three terms:

1. **Source reconstruction loss**: How close is the predicted source activity to the true (synthetic) source activity? This is a straightforward mean squared error.
2. **Forward consistency loss**: We take the predicted source activity, multiply it by the leadfield matrix to project it back to scalp EEG, and compare that to the original input EEG. This ensures the solution is physically valid—it must be able to produce the observed scalp data.
3. **Physiological regularization**: Penalties that encourage the solution to be spatially smooth (consistent with brain connectivity), temporally smooth (consistent with neural mass dynamics), and bounded within physiologically realistic amplitude ranges.

The physics-informed loss function is assumed to be already implemented.

### Step 5: We Preprocess Real NMT EEG Data

We take the abnormal recordings from the NMT dataset and:

1. Verify they are in linked-ear reference (the NMT dataset's native format).
2. Bandpass filter between 0.5 and 70 Hz (or 1–40 Hz for epileptiform-focused analysis).
3. Apply notch filtering at 50 Hz (power line frequency in South Asia).
4. Reject or interpolate channels with excessive artifacts.
5. Run ICA-based artifact rejection to remove eye blinks and muscle artifacts.
6. Segment the continuous recording into fixed-length windows (e.g., 2-second epochs) centered around detected epileptiform events or sampled from abnormal segments.
7. Standardize channel ordering to match the 19-channel 10-20 layout expected by the network.

### Step 6: We Run Inference to Get Source Estimates

Each preprocessed EEG segment is fed into the trained PhysDeepSIF network. The output is a 76-region source activity estimate for that time window. We aggregate across multiple segments from the same patient to build a robust picture of which regions are consistently activated.

### Step 7: We Run Patient-Specific Parameter Optimization

This is the novel clinical-value step. For each patient:

1. We initialize a TVB model with the standard 76-region connectivity.
2. We set all regions to a baseline healthy excitability ($x_0 = -2.2$).
3. We run an optimization algorithm (CMA-ES, a gradient-free evolutionary strategy) that adjusts the 76-dimensional $x_0$ vector to minimize the mismatch between:
   - The simulated source activity and the PhysDeepSIF-estimated source activity from the real EEG.
   - The simulated scalp EEG (via leadfield projection) and the actual patient EEG.
4. After convergence, the fitted $x_0$ vector tells us how excitable each region had to be to reproduce the patient's observed patterns.

### Step 8: We Build the Epileptogenicity Heatmap

The fitted $x_0$ values are transformed into an **Epileptogenicity Index (EI)** for each region:

$$\text{EI}_i = \frac{x_{0,i} - \min(x_0)}{\max(x_0) - \min(x_0)}$$

Regions with the highest EI values (those that required the highest excitability) are flagged as epileptogenic. This is visualized as a color-coded brain map.

---

## 4. What the System Produces

For each patient analyzed, the system produces:

1. **Source activity time series**: Estimated neural activity in each of the 76 brain regions during the EEG recording.
2. **Fitted model parameters**: The $x_0$ excitability vector that best explains the patient's EEG within the TVB framework.
3. **Epileptogenicity heatmap**: A 76-region color-coded map showing which regions are most likely to be epileptogenic.
4. **Confidence metrics**: Consistency of the heatmap across different EEG segments from the same patient, goodness-of-fit statistics, and comparison to classical inverse solutions.

---

## 5. What Makes This Approach Different

- **Training on synthetic data**: We do not need labeled source-level ground truth from real patients (which is almost impossible to obtain). We train entirely on physics-based simulations.
- **Physics-informed constraints**: The inverse solver is not a pure black box. The forward consistency loss forces it to produce solutions that are physically plausible.
- **Mechanistic parameter fitting**: The epileptogenicity heatmap is not just a statistical correlation—it comes from fitting a biophysically grounded brain model. The regions flagged as epileptogenic are those that the model says must be pathologically excitable to produce the observed EEG.
- **Compatibility with clinical EEG**: The system works with standard 19-channel 10-20 EEG with linked-ear reference—the exact setup used in the NMT dataset and in most clinical neurology labs worldwide.

---

## 6. Assumptions and Limitations

- We assume the 76-region Desikan-Killiany parcellation is sufficient spatial resolution for clinical epileptogenicity mapping. This is a region-level, not vertex-level, approach.
- The leadfield matrix is computed from a template head model, not individualized MRI. This introduces some forward model error, which we mitigate by noise augmentation during training.
- The Epileptor model captures seizure-like dynamics but is a phenomenological model. It does not represent every cellular mechanism.
- The NMT dataset labels are binary (normal/abnormal) at the recording level. There are no annotations of specific epileptiform events or their spatial extent. Event detection will be automated or heuristic-based.
- The physics-informed loss function is assumed to be already implemented and validated.

---

## 7. Summary of the Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (Offline, One-Time)                          │
│                                                                 │
│  TVB Epileptor Simulations ──► Source Activity (76 regions)      │
│           │                                                     │
│           ▼                                                     │
│  Leadfield Projection ──► Synthetic EEG (19 channels)           │
│           │                           │                         │
│           ▼                           ▼                         │
│  Ground Truth Source          Input EEG + Noise                  │
│           │                           │                         │
│           └───────────┬───────────────┘                         │
│                       ▼                                         │
│              Train PhysDeepSIF                                   │
│         (Source Loss + Forward Loss + Physics Reg.)              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  INFERENCE PIPELINE (Per Patient)                               │
│                                                                 │
│  NMT EEG Recording ──► Preprocessing ──► EEG Segments           │
│                                              │                  │
│                                              ▼                  │
│                                     PhysDeepSIF Inference        │
│                                              │                  │
│                                              ▼                  │
│                                    Source Estimates (76 regions)  │
│                                              │                  │
│                                              ▼                  │
│                              TVB Parameter Optimization (CMA-ES) │
│                                              │                  │
│                                              ▼                  │
│                              Fitted x₀ Vector (76 values)        │
│                                              │                  │
│                                              ▼                  │
│                              Epileptogenicity Heatmap            │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document provides a plain-language overview of the full pipeline. For exact implementation details, data formats, library specifications, and academic justifications, see the companion Technical Specifications Document.*
