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

1. **Randomly choosing** how many regions (0 to 8) are epileptogenic. About 11% of samples have zero epileptogenic regions (k=0), meaning all 76 regions are healthy. These healthy-only samples ensure the network also learns normal brain activity patterns, making it useful for general-purpose EEG source imaging on any patient, not just epileptic ones. For epileptic samples (k≥1), regions are set with $x_0$ between −1.8 and −1.2 (epileptogenic), while healthy regions get $x_0$ between −2.2 and −2.05 (stable).
2. **Varying** coupling strengths, noise levels, and time constants across physiologically plausible ranges.
3. **Running** the Epileptor simulation at high temporal resolution (dt=0.1 ms) to produce 76 source time series. The raw output is at 20,000 Hz and is anti-aliased down to 200 Hz using a proper FIR decimation filter (not a simple average, which would cause spectral aliasing artifacts).
4. **Projecting** the source time series through the leadfield to get a 19-channel synthetic EEG.
5. **Adding** noise (white Gaussian noise at 5–30 dB SNR plus colored 1/f noise at 10–30% of signal amplitude) to the synthetic EEG to make it realistic.

For each simulation, we store:
- The 19-channel synthetic EEG (the network's **input** during training).
- The 76-region source activity (the network's **target** during training).
- The binary label vector marking which regions were set as epileptogenic (used for evaluation; all-zero for healthy-only samples).

We generate on the order of 50,000–100,000 such samples, with varied seizure topographies, SNR levels, and timing. The dataset includes both healthy-only brain activity (~11%) and mixed healthy+epileptic activity (~89%), reflecting a realistic distribution that makes the network robust for both general-purpose source imaging and epileptogenicity detection.

### Step 4: We Train the PhysDeepSIF Inverse Solver

PhysDeepSIF is a deep neural network inspired by the DeepSIF architecture (Sun et al., 2022, PNAS). It is trained on **both healthy and epileptic synthetic data simultaneously**. This is deliberate: the network's primary job is to solve the EEG inverse problem (recovering brain sources from scalp EEG), which requires understanding both normal and abnormal brain dynamics. The network does NOT first learn on healthy data and then fine-tune on epileptic data—it learns the full distribution from the start.

The network has two main modules:

- A **spatial module**: a stack of fully connected layers with skip connections that takes in the 19-channel EEG at each time step and maps it toward the 76-region source space. This module learns to undo the spatial blurring caused by volume conduction.
- A **temporal module**: a stack of bidirectional LSTM layers that processes the time series in the source space to capture temporal dynamics—how source activity evolves over time.

The network is trained using a **composite loss function** that has three terms:

1. **Source reconstruction loss**: How close is the predicted source activity to the true (synthetic) source activity? This is a straightforward mean squared error.
2. **Physiological regularization**: Penalties that encourage the solution to be temporally smooth (consistent with neural mass dynamics) and bounded within physiologically realistic amplitude ranges. No spatial smoothness penalty is imposed — the leadfield-constrained EEG forward pass naturally produces spatially coherent sources, and the DC offset retained in both EEG and sources provides an implicit spatial prior.
3. **Epileptogenicity classification loss**: Helps the network distinguish epileptogenic from healthy regions. Note: the forward consistency loss (β=0.0) is disabled in the final configuration — the spatial prior is provided by the leadfield matrix implicitly rather than by an explicit forward loss term.

The full training pipeline and configuration are detailed in `docs/30thaprplan.md` and `docs/02_TECHNICAL_SPECIFICATIONS.md`.

### Step 5: We Run Inference to Get Source Estimates

A preprocessed EEG segment (real clinical recording or synthetic test sample) is fed into the trained PhysDeepSIF network. The output is a 76-region source activity estimate for that time window.

### Step 6: We Run Instantaneous Biomarker Detection

Source activity estimates are fed into a heuristic biomarker detector that identifies regions with the strongest transient power, producing an instantaneous epileptogenicity ranking (top-10 regions).

### Step 7: We Run Patient-Specific Parameter Optimization

This is the biophysical validation step. For each EEG sample:

1. A TVB forward model is initialized with the 76-region connectivity matrix and baseline healthy excitability ($x_0 = -2.2$).
2. CMA-ES (a gradient-free evolutionary strategy) optimises the 76 $x_0$ values to minimize the PSD mismatch between simulated EEG and the real EEG at scalp level.
3. After convergence, the fitted $x_0$ vector tells us which regions had to be pathologically excitable to produce the observed EEG patterns.

### Step 8: We Build the Epileptogenicity Heatmap

The fitted $x_0$ values are fed through a sigmoid-based biophysical epileptogenicity index (EI) that maps each region to a 0–1 score. Regions with the highest EI values are flagged as most likely epileptogenic. A concordance score is computed between the instantaneous biomarker ranking and the biophysical EI ranking — high overlap gives a HIGH concordance tier (strong evidence), moderate overlap gives MODERATE, and low overlap gives LOW.

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
