"""
Module: phase3_inference
Phase: 3 - Clinical EEG Inference & NMT Preprocessing
Purpose: Preprocessing, inference, and source aggregation for real patient EEG data

This package orchestrates clinical EEG analysis using the pretrained PhysDeepSIF model.
It handles the complete pipeline from raw EDF files through preprocessing, neural network
inference, and source localization, ultimately enabling epileptogenicity mapping for
clinical decision support.

Key Components:
    - nmt_preprocessor: Six-step EEG preprocessing (load, filter, ICA, segment, normalize)
      per Tech Specs §5.2. Transforms raw 19-channel EDF → 2s windowed epochs (n_epochs, 19, 400)
    
    - inference_engine: Full patient inference orchestrator. Chains NMT preprocessing →
      PhysDeepSIF neural network → source aggregation. Returns per-region statistics
      (mean power, variance, activation consistency).
    
    - source_aggregator: Aggregates window-level source estimates into patient-level
      epileptogenicity profiles using multiple features (power, variance, peak-to-peak,
      activation consistency). Provides input to Phase 4 CMA-ES inversion.

Input Data Format:
    - EDF file (21-channel raw EEG from NMT dataset)
    - Sampling rate: 200 Hz (verified in Tech Specs §3.2)
    - Duration: Variable (30 min to hours)
    - Channels: 19 standard 10-20 montage + 2 ear reference (A1, A2)
    - Expected linking: Linked-ear reference (average of A1, A2)

Output Data Format:
    - preprocessed_eeg: ndarray (n_epochs, 19, 400), dtype=float32
      - Per-channel normalized (z-scored using training stats)
      - No DC offset per-channel (temporal de-meaning applied before global normalization)
    
    - source_estimates: ndarray (n_epochs, 76, 400), dtype=float32
      - PhysDeepSIF predictions in original scale (denormalized)
      - 76 Desikan-Killiany regions, 400 samples per 2s epoch @ 200 Hz
    
    - epileptogenicity_index: ndarray (76,), dtype=float32 ∈ [0, 1]
      - Per-region probability of epileptogenicity
      - Computed via Source Aggregator or Phase 4 CMA-ES

Processing Pipeline (Conceptual Flow):
    
    [NMT EDF File]
        ↓ (load via MNE)
    [Raw mne.io.Raw, 21 ch, 200 Hz]
        ↓ (verify/rename channels)
    [Verified EEG, standardized 10-20 names]
        ↓ (bandpass + notch + ICA)
    [Artifact-free EEG]
        ↓ (segment into 2s windows)
    [EEG epochs, (n_epochs, 19, 400)]
        ↓ (per-channel de-mean + global z-score)
    [Preprocessed epochs, normalized]
        ↓ (PhysDeepSIF forward pass)
    [Source estimates, (n_epochs, 76, 400)]
        ↓ (aggregate across epochs)
    [Per-region statistics: power, variance, consistency]
        ↓ (heuristic epileptogenicity or CMA-ES inversion)
    [Epileptogenicity Index per region]
        ↓ (clinical reporting)
    [Epileptogenic zone hypothesis]

Normalization Invariance (Critical):
    The PhysDeepSIF model was trained on data with this exact preprocessing sequence:
        eeg_denorm = eeg - eeg.mean(axis=-1, keepdims=True)  # Per-channel de-meaning
        eeg_normalized = (eeg_denorm - eeg_mean) / (eeg_std + 1e-7)  # Global z-score
    
    This sequence must be applied identically during inference. The normalization_stats.json
    (eeg_mean, eeg_std) were computed on de-meaned training data. Deviating from this
    order will cause distribution shift and poor inference performance.
    
    The de-meaning step is critical because:
        - Real EEG amplifiers are AC-coupled (HPF ≥ 0.1 Hz), so model never sees DC
        - Epileptor LFP proxy (x2-x1) has large DC offset that varies with x0 (excitability)
        - This DC offset masks the useful discriminative dynamics in variance
        - Removing DC per-channel isolates the neural temporal dynamics
        - See Tech Specs §4.4.7 (DC Offset Root Cause Analysis)

API Reference:
    from src.phase3_inference import NMTPreprocessor, run_patient_inference, aggregate_sources
    
    # Option 1: Full pipeline via inference_engine
    results = run_patient_inference(
        edf_path='data/samples/0001082.edf',
        model=model,
        norm_stats=norm_stats,
        device='cpu'
    )
    # Returns: {"source_estimates", "mean_source_power", "preprocessed_eeg", ...}
    
    # Option 2: Manual control via NMTPreprocessor
    preprocessor = NMTPreprocessor()
    preprocessed_data = preprocessor.run('data/samples/0001082.edf', norm_stats)
    # Then pass to PhysDeepSIF manually for custom processing

Dependencies (all in physdeepsif conda env):
    - mne: EDF loading, filtering, ICA
    - scipy: Signal processing (decimation, filtering)
    - torch: PhysDeepSIF inference
    - numpy: Array operations
    - h5py: Loading normalization statistics

Tech Specs References:
    - Section 5.2: NMT Preprocessing Pipeline (6-step specification)
    - Section 4.4.7: DC Offset Root Cause and De-Meaning Justification
    - Section 8.1: Interface Specification (data shapes and dtypes)
    - Section 3.2: EEG Data Characteristics (19 ch, 200 Hz, 10-20 montage)

Known Limitations:
    - ICA artifact removal may fail on very short recordings (<30s) — graceful fallback provided
    - EDF channel names vary across sites — requires mapping via normalize_channel_names()
    - A1/A2 reference channels must be present for linked-ear calculation

Author: Shahliza Ahmad
Generated: April 27, 2026 (Day 1 of final sprint)
"""

# Package version (tied to overall PhysDeepSIF version)
__version__ = "2.0.0"

# Expose key classes and functions at package level for convenience
# These will be imported once the modules are created
# (Commented out until submodules exist)

# from .nmt_preprocessor import NMTPreprocessor
# from .inference_engine import run_patient_inference
# from .source_aggregator import aggregate_sources

__all__ = [
    "NMTPreprocessor",
    "run_patient_inference",
    "aggregate_sources",
]
