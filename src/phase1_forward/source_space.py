"""
Module: source_space.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Load, preprocess, and validate the TVB default structural connectivity
         and Desikan-Killiany parcellation data that forms the anatomical
         foundation for the entire PhysDeepSIF pipeline.

This module is the very first step of the pipeline. It extracts from TVB:
  - The 76×76 structural connectivity matrix (fiber tract densities)
  - The 76×3 region centroid coordinates in MNI space
  - The 76 region label names
  - The 76×76 tract length matrix (for conduction delays)

All downstream modules depend on these data files — the Epileptor simulator
uses them for network coupling, the leadfield builder uses region centers for
source localization, and the neural network uses the connectivity Laplacian
as a physics constraint in the loss function.

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 3.1 for full specifications.

Key dependencies:
- tvb-library: Provides the default DK-76 connectivity dataset
- numpy: Array operations and file I/O
- json: Saving region label list

Input data format:
- TVB internal connectivity object (loaded from TVB's default dataset)

Output data format:
- connectivity_76.npy: (76, 76) float64, preprocessed connectivity weights
- region_centers_76.npy: (76, 3) float64, MNI coordinates in mm
- region_labels_76.json: list of 76 string labels
- tract_lengths_76.npy: (76, 76) float64, tract lengths in mm
"""

# Standard library imports
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# TVB imports
from tvb.datatypes.connectivity import Connectivity

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Section 3.1)
# ---------------------------------------------------------------------------
N_REGIONS = 76  # Desikan-Killiany parcellation: 68 cortical + 8 subcortical
PARCELLATION_NAME = "desikan_killiany"

# Configure module-level logger
logger = logging.getLogger(__name__)


def load_tvb_connectivity() -> Connectivity:
    """
    Load the TVB default structural connectivity dataset.

    TVB ships with a default connectivity dataset based on averaged DTI
    tractography across healthy subjects, parcellated into the 76-region
    Desikan-Killiany atlas. This function loads that default dataset using
    TVB's built-in data loading mechanism.

    The connectivity object contains:
      - weights: fiber tract densities between all region pairs
      - centres: 3D centroid coordinates of each region (MNI space)
      - region_labels: human-readable names for each region
      - tract_lengths: physical distances along white matter tracts (mm)

    Returns:
        Connectivity: TVB connectivity object with all 76 regions loaded.

    Raises:
        RuntimeError: If TVB cannot load its default connectivity dataset
            (e.g., if tvb-data is not installed or the data files are missing).

    References:
        Technical Specs Section 3.1.2
        Sanz-Leon et al. (2013) — TVB simulator framework
    """
    logger.info("Loading TVB default connectivity dataset...")

    try:
        # from_file() with no arguments loads TVB's default connectivity,
        # which is the DK-76 parcellation derived from averaged DTI data
        connectivity = Connectivity.from_file()

        # TVB requires explicit configuration before the data is accessible
        connectivity.configure()

    except Exception as e:
        raise RuntimeError(
            f"Failed to load TVB default connectivity: {e}. "
            f"Ensure tvb-library and tvb-data are properly installed. "
            f"Try: pip install tvb-library tvb-data"
        ) from e

    # Validate that we got the expected 76-region parcellation
    n_regions = connectivity.number_of_regions
    if n_regions != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} regions (Desikan-Killiany parcellation), "
            f"but TVB loaded a connectivity with {n_regions} regions. "
            f"Ensure you are using TVB's default connectivity dataset."
        )

    logger.info(
        f"Successfully loaded TVB connectivity: {n_regions} regions, "
        f"weights shape {connectivity.weights.shape}, "
        f"centres shape {connectivity.centres.shape}"
    )

    return connectivity


def preprocess_connectivity(
    raw_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Preprocess the raw structural connectivity weights matrix.

    The raw connectivity weights from TVB have a heavy-tailed distribution
    because fiber tract densities vary over several orders of magnitude.
    We apply standard preprocessing (Sanz-Leon et al., 2013; Proix et al., 2017):

    Algorithm:
        1. Log-transform: W' = log10(W + 1)
           - The +1 prevents log(0) for unconnected region pairs
           - log10 compresses the heavy-tailed distribution into a usable range
        2. Normalize by maximum: W'' = W' / max(W')
           - Scales all weights to [0, 1] range
           - Ensures the connectivity strength is relative, not absolute

    After preprocessing, the diagonal is zero (no self-connections), and the
    matrix remains symmetric. The resulting matrix is used by:
      - TVB simulator for inter-region coupling
      - Physics-informed loss function for the graph Laplacian constraint

    Args:
        raw_weights: The raw 76×76 connectivity weight matrix from TVB.
            Shape (76, 76), dtype float64. Values are non-negative fiber
            tract densities. The matrix should be symmetric.

    Returns:
        NDArray[np.float64]: Preprocessed connectivity matrix.
            Shape (76, 76), dtype float64. Values in [0, 1].

    Raises:
        ValueError: If the input matrix is not square, not 76×76,
            or contains negative values.

    References:
        Technical Specs Section 3.1.2 (Preprocessing steps 1-2)
    """
    # --- Input validation ---
    if raw_weights.ndim != 2:
        raise ValueError(
            f"Connectivity matrix must be 2D, got {raw_weights.ndim}D."
        )

    if raw_weights.shape != (N_REGIONS, N_REGIONS):
        raise ValueError(
            f"Connectivity matrix must be ({N_REGIONS}, {N_REGIONS}), "
            f"got {raw_weights.shape}."
        )

    if np.any(raw_weights < 0):
        raise ValueError(
            "Raw connectivity weights contain negative values, which is "
            "not physically meaningful for fiber tract densities."
        )

    logger.info("Preprocessing connectivity weights...")

    # Step 1: Log-transform to compress heavy-tailed distribution
    # Adding 1 before log ensures log(0+1) = 0 for unconnected pairs,
    # preserving the sparsity pattern while compressing large values
    log_weights = np.log10(raw_weights + 1.0)
    logger.debug(
        f"After log-transform: min={log_weights.min():.4f}, "
        f"max={log_weights.max():.4f}"
    )

    # Step 2: Normalize by maximum value to scale into [0, 1]
    max_val = np.max(log_weights)
    if max_val < 1e-10:
        # Edge case: if all weights are zero (extremely unlikely with TVB
        # default data, but handle defensively)
        logger.warning(
            "All connectivity weights are effectively zero after "
            "log-transform. Returning zero matrix."
        )
        return log_weights

    normalized_weights = log_weights / max_val
    logger.debug(
        f"After normalization: min={normalized_weights.min():.4f}, "
        f"max={normalized_weights.max():.4f}"
    )

    # --- Output validation ---
    # Ensure the diagonal is zero (no self-connections)
    np.fill_diagonal(normalized_weights, 0.0)

    # Verify symmetry (connectivity should be undirected)
    if not np.allclose(normalized_weights, normalized_weights.T, atol=1e-10):
        logger.warning(
            "Connectivity matrix is not perfectly symmetric after "
            "preprocessing. Forcing symmetry by averaging with transpose."
        )
        normalized_weights = (normalized_weights + normalized_weights.T) / 2.0

    logger.info(
        f"Preprocessing complete. Non-zero entries: "
        f"{np.count_nonzero(normalized_weights)} / "
        f"{N_REGIONS * N_REGIONS}"
    )

    return normalized_weights


def extract_region_centers(
    connectivity: Connectivity,
) -> NDArray[np.float64]:
    """
    Extract the 3D centroid coordinates of each brain region from TVB.

    These coordinates are in MNI space (millimeters) and represent the
    geometric center of each Desikan-Killiany parcel. They are used for:
      - Leadfield construction (mapping region positions to electrode
        sensitivity via the BEM forward model)
      - Dipole Localization Error (DLE) metric computation
      - Spatial Dispersion (SD) metric computation
      - Epileptogenicity heatmap visualization

    Args:
        connectivity: A configured TVB Connectivity object that has been
            loaded via load_tvb_connectivity(). Must have .centres attribute
            with shape (76, 3).

    Returns:
        NDArray[np.float64]: Region centroid coordinates in MNI space.
            Shape (76, 3), dtype float64. Columns are (x, y, z) in mm.

    Raises:
        ValueError: If the centres array doesn't have the expected shape.

    References:
        Technical Specs Section 3.1.3
    """
    centres = np.array(connectivity.centres, dtype=np.float64)

    # Validate shape: expect exactly 76 regions × 3 spatial dimensions
    if centres.shape != (N_REGIONS, 3):
        raise ValueError(
            f"Region centres must have shape ({N_REGIONS}, 3), "
            f"got {centres.shape}. This indicates the wrong parcellation."
        )

    # Sanity check: MNI coordinates should be within reasonable brain bounds
    # The human brain in MNI space is roughly:
    #   x: [-80, 80] mm (left-right)
    #   y: [-120, 80] mm (posterior-anterior)
    #   z: [-60, 90] mm (inferior-superior)
    coord_min = centres.min(axis=0)
    coord_max = centres.max(axis=0)
    logger.info(
        f"Region centres range: "
        f"x=[{coord_min[0]:.1f}, {coord_max[0]:.1f}], "
        f"y=[{coord_min[1]:.1f}, {coord_max[1]:.1f}], "
        f"z=[{coord_min[2]:.1f}, {coord_max[2]:.1f}] mm"
    )

    return centres


def extract_region_labels(
    connectivity: Connectivity,
) -> list:
    """
    Extract the human-readable names for all 76 brain regions.

    These labels come from the Desikan-Killiany atlas and follow the
    naming convention used by FreeSurfer (e.g., "lh-bankssts",
    "rh-superiorfrontal", etc.). They are used for:
      - Labeling axes in visualizations
      - Mapping between region indices and anatomical names
      - Generating the epileptogenicity values CSV output

    Args:
        connectivity: A configured TVB Connectivity object loaded via
            load_tvb_connectivity(). Must have .region_labels attribute.

    Returns:
        list: A list of 76 strings, each being the name of a brain region
            in the Desikan-Killiany parcellation.

    Raises:
        ValueError: If the number of labels doesn't match N_REGIONS.

    References:
        Technical Specs Section 3.1.1
    """
    # TVB stores labels as a numpy array of strings; convert to Python list
    labels = [str(label) for label in connectivity.region_labels]

    if len(labels) != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} region labels, got {len(labels)}. "
            f"This indicates the wrong parcellation was loaded."
        )

    logger.info(
        f"Extracted {len(labels)} region labels. "
        f"First 5: {labels[:5]}, Last 5: {labels[-5:]}"
    )

    return labels


def extract_tract_lengths(
    connectivity: Connectivity,
) -> NDArray[np.float64]:
    """
    Extract the 76×76 tract length matrix from TVB connectivity.

    Tract lengths represent the physical distance (in mm) along white matter
    fiber tracts between each pair of brain regions. Combined with conduction
    speed (m/s), they determine the signal propagation delays in the TVB
    simulation:
        delay_ij = tract_length_ij / conduction_speed

    These delays are critical for realistic neural dynamics because they
    determine the phase relationships between oscillating regions.

    Args:
        connectivity: A configured TVB Connectivity object loaded via
            load_tvb_connectivity(). Must have .tract_lengths attribute
            with shape (76, 76).

    Returns:
        NDArray[np.float64]: Tract length matrix in millimeters.
            Shape (76, 76), dtype float64. Symmetric, with zeros on diagonal.

    Raises:
        ValueError: If tract lengths don't have the expected shape.

    References:
        Technical Specs Section 3.1.2 (step 3 — conduction delays)
    """
    tract_lengths = np.array(connectivity.tract_lengths, dtype=np.float64)

    if tract_lengths.shape != (N_REGIONS, N_REGIONS):
        raise ValueError(
            f"Tract lengths must have shape ({N_REGIONS}, {N_REGIONS}), "
            f"got {tract_lengths.shape}."
        )

    # Log basic statistics for sanity checking
    # Typical inter-region tract lengths in the human brain: 10–200 mm
    nonzero_lengths = tract_lengths[tract_lengths > 0]
    if len(nonzero_lengths) > 0:
        logger.info(
            f"Tract lengths (non-zero): "
            f"min={nonzero_lengths.min():.1f} mm, "
            f"max={nonzero_lengths.max():.1f} mm, "
            f"mean={nonzero_lengths.mean():.1f} mm"
        )
    else:
        logger.warning("All tract lengths are zero — this is unexpected.")

    return tract_lengths


def save_source_space_data(
    connectivity: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save all source space data files to disk in the expected formats.

    This function saves the four data files that define the anatomical
    source space for the entire pipeline. All downstream modules load
    these files from the data/ directory.

    Files saved (matching config.yaml and Section 9 directory structure):
      - connectivity_76.npy: Preprocessed connectivity weights
      - region_centers_76.npy: Region centroid coordinates in MNI space
      - region_labels_76.json: List of region names
      - tract_lengths_76.npy: Inter-region tract lengths

    Args:
        connectivity: Preprocessed connectivity matrix.
            Shape (76, 76), dtype float64.
        region_centers: Region centroid coordinates.
            Shape (76, 3), dtype float64.
        region_labels: List of 76 region name strings.
        tract_lengths: Tract length matrix.
            Shape (76, 76), dtype float64.
        output_dir: Directory where files will be saved (typically "data/").
            Created if it doesn't exist.

    Returns:
        Dict[str, Path]: Mapping from descriptive keys to the file paths
            of each saved file, for downstream reference.

    Raises:
        ValueError: If any of the input arrays have incorrect shapes.
        OSError: If the output directory cannot be created or files
            cannot be written.

    References:
        Technical Specs Section 9 (Project Directory Structure)
    """
    # Validate all inputs before writing anything to disk
    _validate_source_space_data(
        connectivity, region_centers, region_labels, tract_lengths
    )

    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving source space data to: {output_dir}")

    # Define file paths matching config.yaml naming conventions
    paths = {
        "connectivity": output_dir / "connectivity_76.npy",
        "region_centers": output_dir / "region_centers_76.npy",
        "region_labels": output_dir / "region_labels_76.json",
        "tract_lengths": output_dir / "tract_lengths_76.npy",
    }

    # Save connectivity matrix as numpy binary
    np.save(paths["connectivity"], connectivity)
    logger.info(
        f"Saved connectivity: {paths['connectivity']} "
        f"(shape {connectivity.shape}, dtype {connectivity.dtype})"
    )

    # Save region centers as numpy binary
    np.save(paths["region_centers"], region_centers)
    logger.info(
        f"Saved region centers: {paths['region_centers']} "
        f"(shape {region_centers.shape}, dtype {region_centers.dtype})"
    )

    # Save region labels as JSON (human-readable string list)
    with open(paths["region_labels"], "w") as f:
        json.dump(region_labels, f, indent=2)
    logger.info(
        f"Saved region labels: {paths['region_labels']} "
        f"({len(region_labels)} labels)"
    )

    # Save tract lengths as numpy binary
    np.save(paths["tract_lengths"], tract_lengths)
    logger.info(
        f"Saved tract lengths: {paths['tract_lengths']} "
        f"(shape {tract_lengths.shape}, dtype {tract_lengths.dtype})"
    )

    return paths


def _validate_source_space_data(
    connectivity: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
) -> None:
    """
    Validate all source space data arrays before saving.

    This internal validation function performs comprehensive checks on all
    four data objects to ensure they conform to the specifications in
    Section 8.1 (Interface Specification Table) before they are written
    to disk. Catching errors here prevents corrupted data files from
    propagating through the pipeline.

    Args:
        connectivity: Expected shape (76, 76), dtype float64, symmetric.
        region_centers: Expected shape (76, 3), dtype float64.
        region_labels: Expected length 76, all strings.
        tract_lengths: Expected shape (76, 76), dtype float64, symmetric.

    Raises:
        ValueError: If any validation check fails.
    """
    # --- Connectivity matrix ---
    if connectivity.shape != (N_REGIONS, N_REGIONS):
        raise ValueError(
            f"Connectivity shape must be ({N_REGIONS}, {N_REGIONS}), "
            f"got {connectivity.shape}."
        )
    if connectivity.dtype != np.float64:
        raise ValueError(
            f"Connectivity dtype must be float64, got {connectivity.dtype}. "
            f"See Technical Specs Section 8.1."
        )
    if not np.allclose(connectivity, connectivity.T, atol=1e-10):
        raise ValueError(
            "Connectivity matrix must be symmetric. "
            f"Max asymmetry: {np.max(np.abs(connectivity - connectivity.T))}"
        )

    # --- Region centers ---
    if region_centers.shape != (N_REGIONS, 3):
        raise ValueError(
            f"Region centers shape must be ({N_REGIONS}, 3), "
            f"got {region_centers.shape}."
        )
    if region_centers.dtype != np.float64:
        raise ValueError(
            f"Region centers dtype must be float64, got {region_centers.dtype}."
        )

    # --- Region labels ---
    if len(region_labels) != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} region labels, got {len(region_labels)}."
        )
    if not all(isinstance(label, str) for label in region_labels):
        raise ValueError("All region labels must be strings.")

    # --- Tract lengths ---
    if tract_lengths.shape != (N_REGIONS, N_REGIONS):
        raise ValueError(
            f"Tract lengths shape must be ({N_REGIONS}, {N_REGIONS}), "
            f"got {tract_lengths.shape}."
        )
    if tract_lengths.dtype != np.float64:
        raise ValueError(
            f"Tract lengths dtype must be float64, got {tract_lengths.dtype}."
        )

    logger.debug("All source space data validation checks passed.")


def build_source_space(
    output_dir: str = "data",
) -> Dict[str, Path]:
    """
    Complete pipeline to load, preprocess, validate, and save source space data.

    This is the main entry point for Subtask 1.1 of Phase 1. It orchestrates
    the full sequence of operations:
      1. Load TVB's default 76-region connectivity
      2. Preprocess connectivity weights (log-transform + normalize)
      3. Extract region centroid coordinates
      4. Extract region labels
      5. Extract tract lengths
      6. Validate everything
      7. Save to disk

    After running this function, the data/ directory will contain the four
    foundational data files that all downstream pipeline stages depend on.

    Args:
        output_dir: Directory path where data files will be saved.
            Defaults to "data" (relative to project root).

    Returns:
        Dict[str, Path]: Mapping from descriptive keys to saved file paths.
            Keys: "connectivity", "region_centers", "region_labels",
                  "tract_lengths".

    References:
        Technical Specs Sections 3.1.1–3.1.3, 9 (directory structure)
    """
    logger.info("=" * 60)
    logger.info("BUILDING SOURCE SPACE (Phase 1, Subtask 1.1)")
    logger.info("=" * 60)

    # Step 1: Load the raw connectivity from TVB
    connectivity_obj = load_tvb_connectivity()

    # Step 2: Preprocess the connectivity weights
    # Raw weights have a heavy-tailed distribution; log-transform + normalize
    # makes them suitable for use as coupling weights in the simulator
    raw_weights = np.array(connectivity_obj.weights, dtype=np.float64)
    processed_weights = preprocess_connectivity(raw_weights)

    # Step 3: Extract the 3D region centroid coordinates (MNI space, mm)
    region_centers = extract_region_centers(connectivity_obj)

    # Step 4: Extract human-readable region names
    region_labels = extract_region_labels(connectivity_obj)

    # Step 5: Extract inter-region tract lengths (mm)
    tract_lengths = extract_tract_lengths(connectivity_obj)

    # Step 6–7: Validate and save everything to disk
    output_path = Path(output_dir)
    saved_paths = save_source_space_data(
        connectivity=processed_weights,
        region_centers=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
        output_dir=output_path,
    )

    logger.info("=" * 60)
    logger.info("SOURCE SPACE BUILD COMPLETE")
    logger.info(f"Files saved to: {output_path.resolve()}")
    logger.info("=" * 60)

    return saved_paths


def load_source_space_data(
    data_dir: str = "data",
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list,
    NDArray[np.float64],
]:
    """
    Load previously saved source space data from disk.

    This convenience function loads all four source space data files that
    were saved by build_source_space(). It is used by downstream modules
    (epileptor_simulator, leadfield_builder, synthetic_dataset, loss_functions)
    to obtain the anatomical foundation data.

    Args:
        data_dir: Directory containing the saved data files.
            Defaults to "data" (relative to project root).

    Returns:
        Tuple containing:
        - connectivity: (76, 76) float64, preprocessed weight matrix
        - region_centers: (76, 3) float64, MNI coordinates in mm
        - region_labels: list of 76 region name strings
        - tract_lengths: (76, 76) float64, tract lengths in mm

    Raises:
        FileNotFoundError: If any required data file is missing.
        ValueError: If loaded data doesn't pass validation checks.

    References:
        Technical Specs Section 9 (expected file locations)
    """
    data_path = Path(data_dir)

    # Define expected file paths
    conn_file = data_path / "connectivity_76.npy"
    centers_file = data_path / "region_centers_76.npy"
    labels_file = data_path / "region_labels_76.json"
    tracts_file = data_path / "tract_lengths_76.npy"

    # Check all files exist before loading any
    for filepath in [conn_file, centers_file, labels_file, tracts_file]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Source space data file not found: {filepath}. "
                f"Run build_source_space() first to generate these files."
            )

    # Load each file
    connectivity = np.load(conn_file)
    region_centers = np.load(centers_file)
    tract_lengths = np.load(tracts_file)

    with open(labels_file, "r") as f:
        region_labels = json.load(f)

    # Validate loaded data
    _validate_source_space_data(
        connectivity, region_centers, region_labels, tract_lengths
    )

    logger.info(
        f"Loaded source space data from {data_path}: "
        f"connectivity {connectivity.shape}, "
        f"centers {region_centers.shape}, "
        f"{len(region_labels)} labels, "
        f"tract_lengths {tract_lengths.shape}"
    )

    return connectivity, region_centers, region_labels, tract_lengths
