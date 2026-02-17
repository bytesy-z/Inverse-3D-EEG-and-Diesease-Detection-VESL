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

**CURRENT PHASE: Phase 2 — PhysDeepSIF Network Architecture and Training**

**CURRENT SUBTASK: 2.1 — PhysDeepSIF Network Architecture**

**COMPLETION STATUS:**
- ✅ 1.1: Source Space Definition — COMPLETE
  - TVB connectivity (76×76) loaded and preprocessed
  - Region centroids (76×3) in MNI space
  - Region labels (76 names) available
  
- ✅ 1.2: Neural Mass Model Configuration — COMPLETE
  - Epileptor simulator wrapper implemented
  - Parameter randomization implemented
  - Batch simulation support ready
  
- ✅ 1.3: Leadfield Matrix Construction — COMPLETE
  - BEM forward model computed (3-layer, oct-6)
  - Voronoi vertex-to-region mapping implemented (71 cortical + 5 fallback)
  - Linked-ear reference applied
  - Leadfield validated: shape (19, 76), rank 18, condition number 33.1
  - Data file: `data/leadfield_19x76.npy`
  
- ✅ 1.4: Synthetic Dataset Generation — COMPLETE
  - Full pipeline: TVB simulation → anti-aliased decimation → leadfield projection → noise → HDF5
  - Anti-aliased decimation: Raw monitor + scipy.signal.decimate (FIR, 2-stage 10×10)
  - TVB 2× rate factor handled: dt=0.1ms → 20,000 Hz → decimate to 200 Hz
  - Noise structure: [D,D,0,D,D,0] — no noise on slow variables (z, g)
  - Healthy-only samples (k=0): ~11% of dataset for general inverse solving
  - Mixed samples (k=1-8): ~89% for epileptogenicity detection
  - NaN/Inf detection and rejection for diverged simulations (~5-10% failure rate)
  - HDF5 schema matches Section 3.4.4 exactly
  - Output files: `data/synthetic/{train,val,test}_dataset.h5`
  - Script: `scripts/02_generate_synthetic_data.py`
  - All shapes/dtypes validated: eeg (N,19,400) float32, source (N,76,400) float32, etc.
  - Spectral validation: peak freq ~10 Hz, >60% power <30 Hz, lag-1 autocorr ~0.5-0.6

**Specific Instructions for Current Subtask (2.1):**
- Implement PhysDeepSIF network architecture per Section 4.1 of technical specs
- File: `src/phase2_network/physdeepsif.py`
- Spatial module: 5 FC layers with skip connections and BatchNorm (Section 4.1.2)
- Temporal module: 2 BiLSTM layers with output projection + skip (Section 4.1.3)
- Register leadfield and connectivity Laplacian as non-trainable buffers
- Target: ~355,000 parameters total

**Key Implementation Requirements for Phase 2:**
- Spatial module processes each time step independently (weight-shared across time)
- Layer dimensions: 19→128→256→256→128→76 with skip connections
- BiLSTM: input_size=76, hidden_size=76, bidirectional=True, 2 layers
- Output projection: Linear(152, 76) + skip from spatial module
- Loss function implements exact formulas from Section 4.2 (composite: source + forward + physics)
- Physics loss has three sub-components: Laplacian, temporal smoothness, amplitude bound

**Next Subtasks (after 2.1):**
- 2.2: Physics-Informed Loss Functions
- 2.3: Training Loop Implementation
- 3.1: NMT Preprocessing Pipeline

**Update Instructions:**
This "Current Phase Context" section will be manually updated after each subtask completion. When a subtask is completed:
1. Mark it as complete in project tracking
2. Update "CURRENT SUBTASK" to the next item
3. Update "Specific Instructions" with details from the technical specifications
4. Update "Key Constraints" with validation criteria for the new subtask

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
- Purpose: Orchestrate TVB simulation + leadfield projection + noise addition
- Must generate HDF5 files with exact schema from Section 3.4.4
- Must implement parallel simulation using joblib
- Must apply normalization only in PyTorch Dataset, not in HDF5 storage
- NaN/Inf detection: check source_activity after run_simulation(); discard diverged sims (~5-10%)
- Forward projection: EEG = leadfield (19×76) @ source_activity (76×T)
- Noise model: white noise (target SNR in dB) + colored noise (fraction of signal RMS)
- Colored noise: lowpass-filtered Gaussian, separate spatial correlation via leadfield
- Healthy-only samples (k=0): ~11% of dataset, all x0 values set to -2.2 (normal range)
- Simulation yields 10 seconds at 200 Hz = 2000 time points, segmented into 5 windows of 400

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

---

**END OF COPILOT INSTRUCTIONS**

*This file should be referenced by GitHub Copilot for all code generation in the PhysDeepSIF project. Update the "Current Phase Context" section as the project progresses through phases and subtasks.*
