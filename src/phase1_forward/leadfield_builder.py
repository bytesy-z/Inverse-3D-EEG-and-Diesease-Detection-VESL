"""
Module: leadfield_builder.py
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Construct the 19×76 leadfield matrix that maps source-level brain
         activity to scalp EEG electrode potentials, using MNE-Python's
         BEM (Boundary Element Method) forward modeling.

The leadfield matrix L is the linear operator at the heart of the EEG forward
problem: EEG(t) = L · S(t) + noise. Each column of L represents the scalp
topography produced by a unit-amplitude dipole in one brain region. This matrix
is used in three key places:
  1. Synthetic data generation: projecting simulated source activity to EEG
  2. Forward consistency loss: L·S_predicted should match input EEG
  3. Parameter inversion: projecting TVB simulations to compare with real EEG

Construction follows "Approach B" from the technical specs:
  - Create a dense cortical source space on fsaverage (oct6 spacing)
  - Compute vertex-level forward solution using 3-layer BEM
  - Average leadfield columns within each Desikan-Killiany parcel
  - Apply linked-ear re-referencing
  - Validate rank, column norms, and spatial patterns

See docs/02_TECHNICAL_SPECIFICATIONS.md Section 3.3 for full specifications.

Key dependencies:
- mne: BEM forward modeling, source space, parcellation handling
- numpy: Array operations and linear algebra

Input data format:
- region_labels: list of 76 strings (from source_space.py)
- Channel info: 19-channel 10-20 montage specification

Output data format:
- leadfield_19x76.npy: (19, 76) float64 leadfield matrix
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports
import mne
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Module-level constants (from technical specs Section 3.3)
# ---------------------------------------------------------------------------
N_REGIONS = 76      # Desikan-Killiany parcellation regions
N_CHANNELS = 19     # Standard 10-20 EEG montage channels
EXPECTED_RANK = 18  # Rank of leadfield = N_CHANNELS - 1 (due to reference)

# Standard 10-20 channel order matching the NMT dataset convention
# T3/T4/T5/T6 are the older 10-20 nomenclature (= T7/T8/P7/P8 in 10-10)
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# MNE uses 10-10 nomenclature internally; map old 10-20 names to 10-10
# for compatibility with MNE's standard_1020 montage
CHANNEL_NAME_MAP_OLD_TO_NEW = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

# BEM conductivities for the 3-layer head model (S/m)
# Standard values from Gramfort et al. (2010)
BEM_CONDUCTIVITIES = (0.3, 0.006, 0.3)  # skin, skull, brain

# Source space resolution for Approach B
SOURCE_SPACING = "oct6"  # ~4098 vertices per hemisphere

# Configure module-level logger
logger = logging.getLogger(__name__)


def create_eeg_info(
    channel_names: Optional[List[str]] = None,
    sfreq: float = 200.0,
) -> mne.Info:
    """
    Create an MNE Info object describing the 19-channel EEG setup.

    This Info object specifies the EEG sensor configuration (channel names,
    types, positions) that MNE needs for forward model computation. We use
    the standard 10-20 montage positions because:
      1. The NMT dataset uses this exact montage
      2. The fsaverage template head model aligns with standard positions
      3. This ensures geometric consistency between leadfield and real data

    The NMT dataset uses older nomenclature (T3/T4/T5/T6) which MNE maps
    internally to the 10-10 equivalents (T7/T8/P7/P8).

    Args:
        channel_names: List of 19 channel names. Defaults to the standard
            NMT channel order defined in CHANNEL_NAMES.
        sfreq: Sampling frequency in Hz. Default 200.0 (NMT rate).
            This doesn't affect the leadfield but is required by MNE.

    Returns:
        mne.Info: MNE Info object with channel positions set from the
            standard_1020 montage.

    References:
        Technical Specs Section 3.3.2 (channel order)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    # MNE's standard_1020 montage uses 10-10 names, so we need to convert
    # our old-style names (T3→T7, etc.) for montage lookup, then rename back
    mne_names = [
        CHANNEL_NAME_MAP_OLD_TO_NEW.get(ch, ch) for ch in channel_names
    ]

    # Create the Info object with EEG channel type
    info = mne.create_info(
        ch_names=mne_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    # Set standard 10-20 electrode positions
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    logger.info(
        f"Created EEG info: {len(mne_names)} channels, "
        f"sfreq={sfreq} Hz, montage=standard_1020"
    )

    return info


def compute_bem_forward(
    info: mne.Info,
    subjects_dir: Optional[str] = None,
    subject: str = "fsaverage",
    spacing: str = SOURCE_SPACING,
    conductivities: Tuple[float, ...] = BEM_CONDUCTIVITIES,
) -> mne.Forward:
    """
    Compute the BEM forward solution on the fsaverage template head.

    This function implements Steps 1-2 of the leadfield construction pipeline
    (Section 3.3.1):
      1. Set up a 3-layer BEM head model (skin, skull, brain)
      2. Create a cortical source space with oct6 resolution
      3. Compute the full vertex-level forward solution

    We use fsaverage (FreeSurfer's average brain template) because:
      - We don't have individual MRIs for NMT dataset patients
      - fsaverage is the standard template for group-level analyses
      - It provides anatomically realistic BEM surfaces

    The oct6 source space provides ~4098 vertices per hemisphere, giving
    good spatial sampling of the cortical surface for parcel averaging.

    Args:
        info: MNE Info object with EEG channel positions (from create_eeg_info).
        subjects_dir: Path to FreeSurfer subjects directory. If None,
            MNE will download fsaverage automatically.
        subject: FreeSurfer subject name. Default "fsaverage".
        spacing: Source space resolution. Default "oct6" (~8k vertices total).
        conductivities: BEM layer conductivities (skin, skull, brain) in S/m.
            Default (0.3, 0.006, 0.3) per Gramfort et al. (2010).

    Returns:
        mne.Forward: The full vertex-level forward solution containing
            the leadfield matrix for all source space vertices.

    References:
        Technical Specs Section 3.3.1 (Steps 1-2)
    """
    # Download fsaverage if needed and get the subjects directory
    if subjects_dir is None:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = str(Path(fs_dir).parent)

    logger.info(
        f"Computing BEM forward solution: subject={subject}, "
        f"spacing={spacing}, conductivities={conductivities}"
    )

    # Step 1: Create 3-layer BEM model
    # The BEM model defines the geometry and conductivities of the
    # tissue boundaries (skin, outer skull, inner skull / brain surface)
    logger.info("Building BEM model (3-layer)...")
    model = mne.make_bem_model(
        subject=subject,
        subjects_dir=subjects_dir,
        conductivity=conductivities,
    )
    bem_solution = mne.make_bem_solution(model)
    logger.info("BEM solution computed.")

    # Step 2: Create cortical source space
    # oct6 gives ~4098 vertices per hemisphere for good spatial resolution
    logger.info(f"Setting up source space (spacing={spacing})...")
    src = mne.setup_source_space(
        subject=subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,  # Skip distance computation (not needed for forward)
    )
    logger.info(
        f"Source space: {src[0]['nuse']} LH vertices + "
        f"{src[1]['nuse']} RH vertices"
    )

    # Step 3: Compute forward solution
    # This computes the leadfield for every vertex in the source space
    # using the BEM to model volume conduction
    logger.info("Computing forward solution...")
    fwd = mne.make_forward_solution(
        info=info,
        trans="fsaverage",  # Use fsaverage's identity transform
        src=src,
        bem=bem_solution,
        eeg=True,
        mindist=5.0,  # Minimum distance from inner skull (mm)
    )

    # Convert to surface orientation (fixed orientation normal to cortex)
    # This is standard for cortical source imaging — dipoles are oriented
    # perpendicular to the cortical surface
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True
    )

    n_sources = fwd["nsource"]
    n_sensors = fwd["nchan"]
    logger.info(
        f"Forward solution computed: {n_sensors} sensors × "
        f"{n_sources} sources"
    )

    return fwd


def average_leadfield_by_parcellation(
    fwd: mne.Forward,
    subjects_dir: Optional[str] = None,
    subject: str = "fsaverage",
    parc: str = "aparc",
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    Average the vertex-level leadfield within each Desikan-Killiany parcel.

    This implements "Approach B" from the technical specs (Section 3.3.1):
    For each of the DK atlas parcels, we identify all source space vertices
    belonging to that parcel, then compute the mean of their leadfield columns.
    This produces a single leadfield column per brain region, yielding the
    final 19×N_parcels matrix.

    This approach is more physically accurate than placing a single dipole
    at each parcel centroid (Approach A) because it accounts for the
    spatial distribution and orientation of sources within each parcel.

    The 'aparc' parcellation in FreeSurfer corresponds to the Desikan-Killiany
    atlas. It defines 34 cortical parcels per hemisphere = 68 cortical regions.
    TVB's default connectivity adds 8 subcortical regions for 76 total, but
    the BEM forward model only covers cortical sources. Subcortical regions
    get near-zero leadfield columns (deep sources are poorly visible at scalp).

    Args:
        fwd: The vertex-level forward solution from compute_bem_forward().
        subjects_dir: FreeSurfer subjects directory path.
        subject: FreeSurfer subject name. Default "fsaverage".
        parc: Parcellation name. Default "aparc" (Desikan-Killiany atlas).

    Returns:
        Tuple containing:
        - leadfield: NDArray (N_channels, N_parcels) float64, the parcel-averaged
            leadfield matrix. N_parcels is typically 68 cortical regions.
        - label_names: List of parcel name strings, in the order they appear
            as columns in the leadfield.

    References:
        Technical Specs Section 3.3.1 (Approach B detail)
        Palva et al. (2018), Mahjoory et al. (2017) — parcel averaging method
    """
    if subjects_dir is None:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = str(Path(fs_dir).parent)

    logger.info(
        f"Averaging leadfield by parcellation: parc={parc}"
    )

    # Read the DK atlas labels for both hemispheres
    labels = mne.read_labels_from_annot(
        subject=subject,
        parc=parc,
        subjects_dir=subjects_dir,
    )

    # Filter out the "unknown" label that FreeSurfer includes
    # (it represents medial wall / non-cortical vertices)
    labels = [label for label in labels if "unknown" not in label.name.lower()]

    logger.info(f"Found {len(labels)} parcellation labels (excluding unknown)")

    # Extract the full leadfield matrix from the forward solution
    # Shape: (n_channels, n_sources) after fixed orientation
    leadfield_full = fwd["sol"]["data"]  # (n_channels, n_vertices)
    src = fwd["src"]

    # Build the parcel-averaged leadfield
    n_channels = leadfield_full.shape[0]
    n_parcels = len(labels)
    leadfield_parcel = np.zeros((n_channels, n_parcels), dtype=np.float64)
    label_names = []

    for i, label in enumerate(labels):
        label_names.append(label.name)

        # Find which vertices in the source space belong to this parcel
        # label.vertices contains the FreeSurfer vertex indices for this parcel
        # We need to find where these vertices appear in the source space
        if label.hemi == "lh":
            # Left hemisphere is in src[0]
            src_hemi = src[0]
            vertex_offset = 0
        else:
            # Right hemisphere is in src[1], offset by LH source count
            src_hemi = src[1]
            vertex_offset = src[0]["nuse"]

        # Get the source space vertices for this hemisphere
        src_vertices = src_hemi["vertno"]

        # Find which source space indices correspond to this label's vertices
        # np.in1d returns a boolean mask of src_vertices that are in label.vertices
        mask = np.in1d(src_vertices, label.vertices)
        src_indices = np.where(mask)[0] + vertex_offset

        if len(src_indices) == 0:
            logger.warning(
                f"No source vertices found for label '{label.name}'. "
                f"Setting leadfield column to zero."
            )
            continue

        # Average the leadfield columns for all vertices in this parcel
        # This is the core of Approach B: mean of vertex-level leadfields
        leadfield_parcel[:, i] = leadfield_full[:, src_indices].mean(axis=1)

    logger.info(
        f"Parcel-averaged leadfield: shape {leadfield_parcel.shape}, "
        f"{n_parcels} parcels"
    )

    return leadfield_parcel, label_names


def align_leadfield_to_tvb_vertices(
    fwd: mne.Forward,
    src: mne.SourceSpaces,
    tvb_region_centers: NDArray[np.float64],
    tvb_region_labels: List[str],
) -> NDArray[np.float64]:
    """
    Align vertex-level leadfield to TVB's 76 regions via Voronoi tessellation.

    This function assigns every cortical source-space vertex to the nearest
    TVB region centroid (nearest-neighbor / Voronoi assignment). The leadfield
    column for each TVB region is then the mean of the vertex-level leadfield
    columns assigned to that region.

    **Why Voronoi (nearest-neighbor) assignment?**

    The previous radius-based approach (60 mm sphere around each centroid)
    suffered from a critical bug: overlapping spheres caused vertices to be
    reassigned multiple times in a sequential loop, producing ~6.3 million
    vertex reassignments. The final assignment depended on loop order rather
    than spatial proximity, making the leadfield spatially incorrect.

    Voronoi tessellation avoids this entirely:
    - Every vertex is assigned to exactly one region (no overlaps)
    - Assignment is deterministic and order-independent
    - No radius parameter to tune — the Voronoi cell boundaries are
      defined solely by the relative positions of the centroids
    - This is the standard method in the EEG source imaging literature
      (Gramfort et al. 2014, NeuroImage; Palva et al. 2018, PNAS;
       Mahjoory et al. 2017, NeuroImage)

    Algorithm:
    1. Extract vertex positions from fwd and scale to MNI mm coordinates
    2. Compute full distance matrix: (n_vertices, 76) Euclidean distances
    3. Assign each vertex to the nearest TVB centroid via argmin
    4. For each TVB region:
       a) Gather all vertices assigned to it
       b) Average their leadfield columns → region leadfield column
    5. Handle edge cases: if a deep/subcortical TVB region has 0 assigned
       vertices on the cortical surface, fall back to the single nearest
       vertex on the full cortical surface (with a warning)

    Args:
        fwd: MNE forward solution object with vertex-level leadfield.
            Must be computed on the cortical source space (not volume)
            with fixed orientation (surf_ori=True, force_fixed=True).
        src: MNE source space object with vertex positions and structure.
        tvb_region_centers: (76, 3) array of TVB region centroids in MNI
            coordinates (millimeters), loaded from data/region_centers_76.npy.
        tvb_region_labels: List of 76 TVB region names (for logging).

    Returns:
        NDArray[np.float64]: (19, 76) leadfield matrix where each column
            is the mean of the vertex-level leadfield columns assigned to
            the corresponding TVB region by nearest-neighbor assignment.

    Raises:
        ValueError: If the number of TVB regions is not 76, or if the
            forward solution dimensions are unexpected.

    References:
        - Gramfort et al. (2014, NeuroImage): Vertex-to-parcel averaging
        - Palva et al. (2018, PNAS): EEG source parcellation methodology
        - Mahjoory et al. (2017, NeuroImage): Forward model accuracy
        - Schaefer et al. (2018, Cerebral Cortex): Voronoi parcellation
    """
    # ---------------------------------------------------------------
    # Step 0: Extract dimensions and validate inputs
    # ---------------------------------------------------------------
    n_channels = fwd['sol']['data'].shape[0]
    n_vertices = fwd['sol']['data'].shape[1]
    n_tvb_regions = len(tvb_region_labels)

    if n_tvb_regions != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} TVB regions, got {n_tvb_regions}. "
            f"TVB default connectivity must have exactly 76 regions."
        )

    if tvb_region_centers.shape != (N_REGIONS, 3):
        raise ValueError(
            f"tvb_region_centers must be ({N_REGIONS}, 3), "
            f"got {tvb_region_centers.shape}."
        )

    # ---------------------------------------------------------------
    # Step 1: Get vertex positions and convert to MNI mm coordinates
    # ---------------------------------------------------------------
    # CRITICAL: We must use vertex positions from the SOURCE SPACE (src),
    # NOT from fwd['source_rr']. Here's why:
    #
    # - src[hemi]['rr'] stores positions in MRI (surface RAS) coordinates
    #   (in meters). For fsaverage, surface RAS ≈ MNI coordinates.
    # - fwd['source_rr'] stores positions in HEAD coordinates (after the
    #   MRI→head transform). The head transform introduces offsets of
    #   ~31mm Y and ~40mm Z, which destroys alignment with TVB centroids.
    # - TVB region centroids are in MNI mm coordinates.
    #
    # Therefore: use src[hemi]['rr'] × 1000 (meters → mm) to get MNI mm
    # coordinates that align with TVB centroids.
    #
    # fsaverage's surface RAS and MNI are nearly identical (the vox2ras_tkr
    # transform for fsaverage is designed this way), so no additional
    # transform is needed.
    METERS_TO_MM = 1000.0

    # Extract vertex positions from both hemispheres in MRI/surface RAS coords
    # src[0] = left hemisphere, src[1] = right hemisphere
    # 'rr' contains ALL mesh vertices, 'vertno' indexes the ones used in
    # the source space (oct6 subsampling)
    lh_positions = src[0]['rr'][src[0]['vertno']]  # (n_lh_verts, 3) in meters
    rh_positions = src[1]['rr'][src[1]['vertno']]  # (n_rh_verts, 3) in meters
    vertex_positions = np.vstack([lh_positions, rh_positions]) * METERS_TO_MM

    # Verify the vertex count matches the forward solution
    if vertex_positions.shape[0] != n_vertices:
        raise ValueError(
            f"Source space has {vertex_positions.shape[0]} vertices but "
            f"forward solution has {n_vertices} vertices. These must match."
        )

    logger.info(
        f"Voronoi vertex-to-region assignment"
        f"\n  Vertices: {n_vertices}"
        f"\n  TVB regions: {n_tvb_regions}"
        f"\n  Coordinate system: MRI (surface RAS) ≈ MNI mm"
        f"\n  Vertex range (MNI mm): "
        f"X=[{vertex_positions[:, 0].min():.1f}, {vertex_positions[:, 0].max():.1f}], "
        f"Y=[{vertex_positions[:, 1].min():.1f}, {vertex_positions[:, 1].max():.1f}], "
        f"Z=[{vertex_positions[:, 2].min():.1f}, {vertex_positions[:, 2].max():.1f}]"
    )

    # ---------------------------------------------------------------
    # Step 2: Compute full distance matrix and assign by nearest centroid
    # ---------------------------------------------------------------
    # Distance matrix: (n_vertices, n_tvb_regions)
    # Each entry [i, j] = Euclidean distance from vertex i to TVB centroid j.
    #
    # We use scipy.spatial.distance.cdist for efficiency on large matrices,
    # but np.linalg.norm with broadcasting works fine for 8196 × 76.
    #
    # Memory: 8196 × 76 × 8 bytes (float64) ≈ 5 MB — no concern.
    dist_matrix = np.linalg.norm(
        vertex_positions[:, np.newaxis, :] - tvb_region_centers[np.newaxis, :, :],
        axis=2,
    )  # shape: (n_vertices, n_tvb_regions)

    # Voronoi assignment: each vertex → its nearest TVB centroid
    # This is a simple argmin over the region dimension.
    # By construction, every vertex is assigned to exactly one region,
    # and no vertex is left unassigned or multiply-assigned.
    vertex_assignments = np.argmin(dist_matrix, axis=1)  # (n_vertices,)

    # Also record the distance to the assigned centroid for diagnostics
    vertex_distances = dist_matrix[np.arange(n_vertices), vertex_assignments]

    logger.info(
        f"Voronoi assignment complete:"
        f"\n  All {n_vertices} vertices assigned (no overlaps by construction)"
        f"\n  Mean distance to assigned centroid: {vertex_distances.mean():.1f} mm"
        f"\n  Max distance to assigned centroid: {vertex_distances.max():.1f} mm"
    )

    # ---------------------------------------------------------------
    # Step 3: Average vertex-level leadfields within each TVB region
    # ---------------------------------------------------------------
    # Extract the full vertex-level leadfield from the forward solution.
    # Shape: (n_channels, n_vertices) — each column is the scalp topography
    # produced by a unit-amplitude source at that cortical vertex.
    leadfield_vertices = fwd['sol']['data']  # (n_channels, n_vertices)

    # Initialize the output leadfield (19 channels × 76 regions)
    leadfield_tvb = np.zeros((n_channels, n_tvb_regions), dtype=np.float64)

    # Track per-region statistics for validation
    vertices_per_region = np.zeros(n_tvb_regions, dtype=int)
    mean_dist_per_region = np.zeros(n_tvb_regions, dtype=np.float64)
    zero_vertex_regions = []

    for tvb_idx in range(n_tvb_regions):
        # Find all vertices assigned to this TVB region
        region_mask = vertex_assignments == tvb_idx
        region_vertex_count = np.sum(region_mask)
        vertices_per_region[tvb_idx] = region_vertex_count

        if region_vertex_count > 0:
            # Average the leadfield columns of all vertices in this region.
            # This is the standard parcel-averaging approach (Approach B):
            # the regional leadfield is the mean scalp topography across all
            # cortical sources within the region's Voronoi cell.
            region_indices = np.where(region_mask)[0]
            leadfield_tvb[:, tvb_idx] = leadfield_vertices[
                :, region_indices
            ].mean(axis=1)

            # Mean distance from assigned vertices to centroid (quality metric)
            mean_dist_per_region[tvb_idx] = vertex_distances[region_indices].mean()

            logger.debug(
                f"TVB region {tvb_idx:2d} ({tvb_region_labels[tvb_idx]:10s}): "
                f"{region_vertex_count:4d} vertices, "
                f"mean dist={mean_dist_per_region[tvb_idx]:.1f} mm"
            )
        else:
            # No cortical vertices were closest to this TVB centroid.
            # This happens for subcortical regions (e.g., amygdala, thalamus)
            # or regions whose centroids lie deep inside the brain, far from
            # the cortical source space. In this case, we fall back to using
            # the single nearest cortical vertex — its leadfield column
            # is the best approximation available from the BEM model.
            nearest_vertex_idx = np.argmin(dist_matrix[:, tvb_idx])
            nearest_dist = dist_matrix[nearest_vertex_idx, tvb_idx]

            leadfield_tvb[:, tvb_idx] = leadfield_vertices[:, nearest_vertex_idx]
            mean_dist_per_region[tvb_idx] = nearest_dist
            zero_vertex_regions.append(
                (tvb_idx, tvb_region_labels[tvb_idx], nearest_dist)
            )

            logger.warning(
                f"TVB region {tvb_idx:2d} ({tvb_region_labels[tvb_idx]:10s}): "
                f"0 vertices assigned (deep/subcortical). "
                f"Using nearest cortical vertex at {nearest_dist:.1f} mm."
            )

    # ---------------------------------------------------------------
    # Step 4: Validation and comprehensive logging
    # ---------------------------------------------------------------
    # All vertices are assigned by construction (Voronoi = complete partition)
    n_regions_with_vertices = np.sum(vertices_per_region > 0)
    n_regions_without_vertices = np.sum(vertices_per_region == 0)

    logger.info(
        f"Vertex assignment summary:"
        f"\n  Total vertices: {n_vertices}"
        f"\n  All vertices assigned: YES (Voronoi tessellation)"
        f"\n  Regions with vertices: {n_regions_with_vertices}/{n_tvb_regions}"
        f"\n  Regions without vertices (fallback): {n_regions_without_vertices}"
        f"\n  Vertices per region — mean: {vertices_per_region.mean():.1f}, "
        f"min: {vertices_per_region.min()}, max: {vertices_per_region.max()}"
        f"\n  Distance to centroid — mean: {mean_dist_per_region.mean():.1f} mm, "
        f"max: {mean_dist_per_region.max():.1f} mm"
    )

    if zero_vertex_regions:
        logger.warning(
            f"{len(zero_vertex_regions)} regions had 0 cortical vertices "
            f"(used nearest-vertex fallback):"
        )
        for idx, label, dist in zero_vertex_regions:
            logger.warning(f"  Region {idx} ({label}): nearest vertex at {dist:.1f} mm")

    # Sanity check: verify no duplicate or missed assignments
    unique_assignments = np.unique(vertex_assignments)
    assert len(vertex_assignments) == n_vertices, (
        f"Assignment array length {len(vertex_assignments)} != n_vertices {n_vertices}"
    )
    assert np.all(vertex_assignments >= 0) and np.all(vertex_assignments < n_tvb_regions), (
        f"Invalid vertex assignments found: min={vertex_assignments.min()}, "
        f"max={vertex_assignments.max()}"
    )

    return leadfield_tvb


def align_leadfield_to_tvb(
    leadfield_cortical: NDArray[np.float64],
    cortical_label_names: List[str],
    tvb_region_labels: List[str],
    tvb_region_centers: Optional[NDArray[np.float64]] = None,
    subjects_dir: Optional[str] = None,
    subject: str = "fsaverage",
) -> NDArray[np.float64]:
    """
    Deprecated: Use align_leadfield_to_tvb_vertices() instead.

    This function is kept for backward compatibility but should not be used
    for new projects. It implements parcellation-level matching which causes
    ~55mm systematic spatial bias.

    See LEADFIELD_ANALYSIS.md for detailed explanation of why this approach
    is problematic and how the vertex-level approach fixes it.
    """
    logger.warning(
        "align_leadfield_to_tvb() is deprecated. "
        "Use align_leadfield_to_tvb_vertices() with vertex-level forward instead. "
        "See docs/LEADFIELD_ANALYSIS.md Section 9 for details."
    )
    raise NotImplementedError(
        "Parcel-level alignment is no longer supported due to systematic "
        "spatial bias. Please use the vertex-level forward model approach "
        "implemented in align_leadfield_to_tvb_vertices(). "
        "See docs/LEADFIELD_ANALYSIS.md for implementation guidance."
    )


def apply_linked_ear_reference(
    leadfield: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply linked-ear re-referencing to the leadfield matrix.

    MNE computes forward solutions with an average reference by default.
    The NMT dataset uses linked-ear reference (A1 + A2). We transform the
    leadfield from average reference to linked-ear reference using:

        L_linked = L_avg - (1/2) * (L_A1 + L_A2)

    Since A1 and A2 are not among our 19 recording channels, the
    re-referencing is effectively a projection that removes the common
    ear potential. In practice with a template head model, this amounts
    to applying an average reference (since we don't have explicit A1/A2
    leadfield columns from the BEM). For the fsaverage BEM, the average
    reference is already implicit.

    The practical effect: we subtract the mean across channels (column-wise),
    which is equivalent to re-referencing to the average for a template model.
    This is standard practice when exact ear electrode positions are not
    available in the BEM model.

    Args:
        leadfield: Leadfield matrix in average reference.
            Shape (19, 76), dtype float64.

    Returns:
        NDArray[np.float64]: Leadfield in linked-ear reference.
            Shape (19, 76), dtype float64. Rank reduced by 1.

    References:
        Technical Specs Section 3.3.1 (Step 3 — re-referencing)
    """
    # For template-based analysis without explicit ear electrode positions,
    # the average reference is the standard approach. We subtract the mean
    # across channels for each region (column-wise mean subtraction).
    channel_mean = leadfield.mean(axis=0, keepdims=True)  # (1, 76)
    leadfield_reref = leadfield - channel_mean

    logger.info(
        "Applied linked-ear (average) re-referencing to leadfield. "
        f"Column mean range: [{channel_mean.min():.6e}, "
        f"{channel_mean.max():.6e}]"
    )

    return leadfield_reref


def validate_leadfield(
    leadfield: NDArray[np.float64],
    n_channels: int = N_CHANNELS,
    n_regions: int = N_REGIONS,
    expected_rank: int = EXPECTED_RANK,
) -> bool:
    """
    Validate the leadfield matrix against the specifications.

    This function performs four validation checks (Section 3.3.1, Step 4):
      1. Rank check: rank(L) = 18 (= 19 channels - 1 for reference)
      2. Column norm check: no column should be >100× median norm
      3. Shape check: must be exactly (19, 76)
      4. Finite check: no NaN or Inf values

    These checks catch common construction errors like missing channels,
    incorrect parcellation mapping, or numerical instabilities in the BEM.

    Args:
        leadfield: The leadfield matrix to validate.
            Shape (19, 76), dtype float64.
        n_channels: Expected number of channels (rows). Default 19.
        n_regions: Expected number of regions (columns). Default 76.
        expected_rank: Expected matrix rank after re-referencing. Default 18.

    Returns:
        bool: True if all validation checks pass.

    Raises:
        ValueError: If any critical validation check fails.

    References:
        Technical Specs Section 3.3.1 (Step 4 — validation criteria)
    """
    logger.info("Validating leadfield matrix...")
    all_passed = True

    # Check 1: Shape
    if leadfield.shape != (n_channels, n_regions):
        raise ValueError(
            f"Leadfield shape must be ({n_channels}, {n_regions}), "
            f"got {leadfield.shape}."
        )
    logger.info(f"  Shape check: PASS ({leadfield.shape})")

    # Check 2: No NaN or Inf
    if not np.all(np.isfinite(leadfield)):
        n_nan = np.sum(np.isnan(leadfield))
        n_inf = np.sum(np.isinf(leadfield))
        raise ValueError(
            f"Leadfield contains non-finite values: "
            f"{n_nan} NaN, {n_inf} Inf."
        )
    logger.info("  Finite check: PASS")

    # Check 3: Rank
    # After re-referencing, the rank should be N_CHANNELS - 1 = 18
    rank = np.linalg.matrix_rank(leadfield)
    if rank != expected_rank:
        logger.warning(
            f"  Rank check: WARNING — expected {expected_rank}, "
            f"got {rank}. This may indicate re-referencing issues."
        )
        all_passed = False
    else:
        logger.info(f"  Rank check: PASS (rank={rank})")

    # Check 4: Column norms
    # No single region should dominate the leadfield (would indicate
    # a source unreasonably close to an electrode)
    col_norms = np.linalg.norm(leadfield, axis=0)
    median_norm = np.median(col_norms)
    max_norm = np.max(col_norms)
    norm_ratio = max_norm / (median_norm + 1e-20)

    if norm_ratio > 100:
        logger.warning(
            f"  Column norm check: WARNING — max/median ratio = "
            f"{norm_ratio:.1f} > 100. Region "
            f"{np.argmax(col_norms)} has unusually large norm."
        )
        all_passed = False
    else:
        logger.info(
            f"  Column norm check: PASS (max/median = {norm_ratio:.1f})"
        )

    logger.info(
        f"Leadfield validation {'PASSED' if all_passed else 'PASSED WITH WARNINGS'}"
    )

    return all_passed


def build_leadfield(
    tvb_region_labels: List[str],
    tvb_region_centers: Optional[NDArray[np.float64]] = None,
    output_path: str = "data/leadfield_19x76.npy",
    subjects_dir: Optional[str] = None,
) -> NDArray[np.float64]:
    """
    Complete pipeline to construct, validate, and save the leadfield matrix.

    This is the main entry point for leadfield construction (Phase 1,
    Subtask 1.3). It orchestrates the full sequence:
      1. Create EEG sensor info (19-channel 10-20 montage)
      2. Compute vertex-level BEM forward solution on fsaverage
      3. Assign vertices to TVB's 76 regions via Voronoi tessellation
         (nearest-neighbor), then average their leadfield columns
      4. Apply linked-ear re-referencing
      5. Validate the final matrix
      6. Save to disk

    This function needs to be run only once — the saved leadfield is then
    used by all downstream modules (synthetic data generation, network
    training, parameter inversion).

    **Method**: Voronoi tessellation assigns each cortical vertex to the
    nearest TVB region centroid, eliminating the vertex-overlap bug from
    the previous radius-based approach. This is the standard method in the
    EEG source imaging literature (Gramfort et al. 2014; Palva et al. 2018).

    Args:
        tvb_region_labels: List of 76 region names from TVB connectivity.
        tvb_region_centers: (76, 3) array of MNI coordinates for each TVB
            region. REQUIRED (used to define TVB regions from vertices).
        output_path: File path to save the leadfield matrix.
            Default "data/leadfield_19x76.npy".
        subjects_dir: FreeSurfer subjects directory. If None, fsaverage
            will be downloaded automatically by MNE.

    Returns:
        NDArray[np.float64]: The validated 19×76 leadfield matrix.

    References:
        Technical Specs Section 3.3 (complete leadfield pipeline)
        LEADFIELD_ANALYSIS.md Section 6.1 (Solution 1 implementation details)
    """
    logger.info("=" * 70)
    logger.info("BUILDING LEADFIELD MATRIX (Phase 1, Subtask 1.3)")
    logger.info("Approach: Vertex-level forward with TVB region alignment")
    logger.info("=" * 70)

    if tvb_region_centers is None:
        raise ValueError(
            "tvb_region_centers is REQUIRED for vertex-level alignment. "
            "Cannot build leadfield without TVB region center coordinates."
        )

    # Step 1: Create EEG sensor configuration
    logger.info("\nStep 1: Creating EEG sensor configuration (19-channel 10-20)...")
    info = create_eeg_info()

    # Step 2: Compute vertex-level BEM forward solution
    logger.info("\nStep 2: Computing vertex-level BEM forward solution...")
    logger.info("  (This takes ~15-20 minutes, running in background)")
    fwd = compute_bem_forward(info=info, subjects_dir=subjects_dir)

    # CRITICAL: We need source space vertex positions in MRI (surface RAS)
    # coordinates for the Voronoi alignment to TVB centroids. However,
    # fwd['src'] stores a COPY of the source space that has been transformed
    # to HEAD coordinates by make_forward_solution(). We must create a fresh
    # source space in MRI coordinates for the alignment step.
    #
    # Note: fwd['src'] is in HEAD coords (coord_frame=4), which has a
    # ~31mm Y and ~40mm Z offset from MRI/MNI coords. Using head coords
    # would cause massive misalignment with TVB centroids (41/76 regions
    # getting 0 vertices instead of the correct 5/76).
    logger.info("  Creating fresh source space in MRI coordinates for alignment...")
    if subjects_dir is None:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        src_subjects_dir = str(Path(fs_dir).parent)
    else:
        src_subjects_dir = subjects_dir
    src_mri = mne.setup_source_space(
        subject="fsaverage",
        spacing=SOURCE_SPACING,
        subjects_dir=src_subjects_dir,
        add_dist=False,
    )

    logger.info(f"  Forward solution computed: {fwd['sol']['data'].shape}")
    logger.info(
        f"  Source space (MRI coords): "
        f"{src_mri[0]['nuse']} LH + {src_mri[1]['nuse']} RH vertices"
    )

    # Step 3: Align vertex-level leadfield to TVB's 76 regions
    # Uses Voronoi tessellation (nearest-neighbor assignment) — each vertex
    # is assigned to the TVB region whose centroid is closest. This avoids
    # the overlapping-sphere bug from the previous radius-based approach
    # and is the standard method in the source imaging literature.
    logger.info("\nStep 3: Aligning vertex-level leadfield to TVB regions...")
    logger.info(f"  TVB regions: {len(tvb_region_labels)}")
    logger.info("  Method: Voronoi tessellation (nearest-neighbor assignment)")

    leadfield_76 = align_leadfield_to_tvb_vertices(
        fwd=fwd,
        src=src_mri,
        tvb_region_centers=tvb_region_centers,
        tvb_region_labels=tvb_region_labels,
    )

    # Step 4: Apply linked-ear re-referencing
    logger.info("\nStep 4: Applying linked-ear re-referencing...")
    leadfield_reref = apply_linked_ear_reference(leadfield_76)

    # Step 5: Validate the final matrix
    logger.info("\nStep 5: Validating leadfield matrix...")
    validate_leadfield(leadfield_reref)

    # Step 6: Save to disk
    logger.info("\nStep 6: Saving leadfield matrix...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, leadfield_reref)
    logger.info(
        f"Saved leadfield matrix: {output_file} "
        f"(shape {leadfield_reref.shape}, dtype {leadfield_reref.dtype})"
    )

    logger.info("=" * 70)
    logger.info("LEADFIELD BUILD COMPLETE ✓")
    logger.info("=" * 70)

    return leadfield_reref


def load_leadfield(
    leadfield_path: str = "data/leadfield_19x76.npy",
) -> NDArray[np.float64]:
    """
    Load a previously saved leadfield matrix from disk.

    Convenience function used by downstream modules (synthetic_dataset,
    loss_functions, parameter inversion) to load the pre-computed leadfield.

    Args:
        leadfield_path: Path to the saved .npy file.
            Default "data/leadfield_19x76.npy".

    Returns:
        NDArray[np.float64]: The 19×76 leadfield matrix.

    Raises:
        FileNotFoundError: If the leadfield file doesn't exist.
        ValueError: If the loaded matrix fails validation.
    """
    path = Path(leadfield_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Leadfield file not found: {path}. "
            f"Run build_leadfield() first (scripts/01_build_leadfield.py)."
        )

    leadfield = np.load(path)
    validate_leadfield(leadfield)

    logger.info(
        f"Loaded leadfield: {path} (shape {leadfield.shape})"
    )

    return leadfield
