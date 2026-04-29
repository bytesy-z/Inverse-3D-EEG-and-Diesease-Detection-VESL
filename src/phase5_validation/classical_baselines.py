import logging
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

import mne
from mne.forward import Forward as ForwardClass
from mne.source_space import SourceSpaces
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv
from mne.io.constants import FIFF

from src.phase1_forward.leadfield_builder import create_eeg_info



def _make_info_with_ref():
    """Create MNE Info with average EEG reference projector."""
    info = create_eeg_info(sfreq=SFREQ)
    dummy = mne.io.RawArray(np.zeros((N_CHANNELS, 400)), info)
    dummy.set_eeg_reference("average", projection=True)
    return dummy.info



logger = logging.getLogger(__name__)

N_CHANNELS = 19
N_REGIONS = 76
SFREQ = 200.0
LAMBDA2 = 1.0 / 9.0

FIXED_ORI = FIFF.FIFFV_MNE_FIXED_ORI
MRI_COORD = FIFF.FIFFV_COORD_MRI


def _build_source_spaces(region_centers: NDArray) -> SourceSpaces:
    """Build discrete MNE SourceSpaces from 76 region center coordinates.

    Each region center becomes a single dipole source with fixed orientation.

    Args:
        region_centers: (76, 3) MNI coordinates in mm.

    Returns:
        SourceSpaces with one discrete source per region.
    """
    n = region_centers.shape[0]
    if region_centers.shape != (N_REGIONS, 3):
        raise ValueError(f"Expected ({N_REGIONS}, 3) got {region_centers.shape}.")

    pos = np.ascontiguousarray(region_centers, dtype=np.float64)
    norms = np.zeros((n, 3), dtype=np.float64)
    norms[:, 2] = 1.0

    src_dict = {
        "type": "discrete",
        "rr": pos,
        "nn": norms,
        "nuse": n,
        "vertno": np.arange(n, dtype=np.int32),
        "inuse": np.ones(n, dtype=np.int32),
        "coord_frame": MRI_COORD,
        "id": 101,
        "segmented": False,
        "selection": np.arange(n, dtype=np.int32),
        "select": np.ones(n, dtype=bool),
    }

    src = SourceSpaces([src_dict])

    logger.info(
        f"_build_source_spaces — {n} discrete sources, "
        f"coord range x=[{pos[:,0].min():.1f},{pos[:,0].max():.1f}] "
        f"y=[{pos[:,1].min():.1f},{pos[:,1].max():.1f}] "
        f"z=[{pos[:,2].min():.1f},{pos[:,2].max():.1f}] mm"
    )
    return src


def _build_mne_forward(
    leadfield: NDArray,
    info: mne.Info,
    src: SourceSpaces,
    region_centers: NDArray,
) -> ForwardClass:
    """Construct MNE Forward object from our custom 19x76 leadfield matrix.

    Wraps the pre-computed leadfield into an MNE Forward so that MNE's
    inverse operators (dSPM, MNE, eLORETA) can work with it.

    Args:
        leadfield: (19, 76) float64 leadfield matrix.
        info: MNE Info with 19 EEG channels.
        src: SourceSpaces with 76 discrete sources.
        region_centers: (76, 3) source positions in mm.

    Returns:
        mne.Forward object ready for make_inverse_operator.
    """
    n_chan, n_src = leadfield.shape

    if n_chan != N_CHANNELS or n_src != N_REGIONS:
        raise ValueError(
            f"Leadfield shape must be ({N_CHANNELS}, {N_REGIONS}), "
            f"got ({n_chan}, {n_src})."
        )

    source_nn = np.zeros((n_src, 3), dtype=np.float64)
    source_nn[:, 2] = 1.0

    fwd_dict = {
        "info": info,
        "sol": {
            "data": np.ascontiguousarray(leadfield, dtype=np.float64),
            "row_names": info["ch_names"],
        },
        "source_rr": np.ascontiguousarray(region_centers, dtype=np.float64),
        "source_ori": FIXED_ORI,
        "source_nn": source_nn,
        "nsource": n_src,
        "nchan": n_chan,
        "src": src,
        "surf_ori": False,
        "coord_frame": MRI_COORD,
        "sol_grad": None,
        "_orig_source_projection": np.eye(n_src),
        "_orig_sol": np.ascontiguousarray(leadfield, dtype=np.float64),
        "mri_head_t": np.eye(4, dtype=np.float64),
    }

    fwd = ForwardClass(fwd_dict)

    logger.info(
        f"_build_mne_forward — Forward: {fwd['nchan']} channels x "
        f"{fwd['nsource']} sources, fixed_ori=True, "
        f"leadfield range=[{leadfield.min():.4f}, {leadfield.max():.4f}]"
    )
    return fwd


def _make_noise_cov(info: mne.Info) -> mne.Covariance:
    """Create a minimal diagonal noise covariance for synthetic data.

    Args:
        info: MNE Info with channel names.

    Returns:
        mne.Covariance with diagonal (19, 19).
    """
    n_chan = len(info["ch_names"])
    data = np.eye(n_chan, dtype=np.float64)
    eigvals, eigvecs = linalg.eigh(data)

    noise_cov = mne.Covariance(
        data,
        info["ch_names"],
        info["bads"],
        info["projs"],
        nfree=int(1e10),
        eig=eigvals,
        eigvec=eigvecs,
    )
    logger.debug(f"_make_noise_cov — shape ({n_chan}, {n_chan}), identity diagonal")
    return noise_cov


def _make_data_cov(eeg: NDArray, ch_names: list) -> mne.Covariance:
    """Compute data covariance from an EEG window.

    Args:
        eeg: (19, 400) EEG data array.
        ch_names: list of channel name strings.

    Returns:
        mne.Covariance for LCMV beamformer.
    """
    data_cov = np.cov(eeg).astype(np.float64)
    n_chan = data_cov.shape[0]
    eigvals, eigvecs = linalg.eigh(data_cov)

    if np.any(eigvals < 0):
        logger.warning(
            f"Data covariance has {np.sum(eigvals < 0)} negative eigenvalues. "
            "Adding regularization."
        )
        data_cov += 1e-6 * np.eye(n_chan, dtype=np.float64)
        eigvals, eigvecs = linalg.eigh(data_cov)

    cov = mne.Covariance(
        data_cov,
        ch_names,
        [],
        [],
        nfree=eeg.shape[1],
        eig=eigvals,
        eigvec=eigvecs,
    )
    return cov


def _check_eeg(eeg: NDArray, label: str):
    """Validate EEG input shape and values. Logs diagnostics."""
    if eeg.ndim != 3:
        raise ValueError(
            f"{label}: expected 3-D (batch, 19, 400), got "
            f"{eeg.ndim}-D {eeg.shape}."
        )
    if eeg.shape[1:] != (N_CHANNELS, 400):
        raise ValueError(
            f"{label}: expected (batch, {N_CHANNELS}, 400), got {eeg.shape}."
        )
    if not np.all(np.isfinite(eeg)):
        n_bad = np.sum(~np.isfinite(eeg))
        logger.warning(f"{label}: {n_bad} non-finite values found.")


def _apply_inverse_to_batch(
    eeg_batch: NDArray,
    info: mne.Info,
    inv_op,
    method: str,
    label: str,
) -> NDArray:
    """Apply inverse operator to each sample in batch.

    Args:
        eeg_batch: (batch, 19, 400) EEG windows.
        info: MNE Info object.
        inv_op: InverseOperator from make_inverse_operator.
        method: 'MNE', 'dSPM', or 'eLORETA'.
        label: Method name for logging.

    Returns:
        NDArray (batch, 76, 400) source estimates.
    """
    batch_size = eeg_batch.shape[0]
    results = []

    for i in range(batch_size):
        evoked = mne.EvokedArray(eeg_batch[i].astype(np.float64), info)
        try:
            stc = apply_inverse(evoked, inv_op, lambda2=LAMBDA2, method=method)
            src_data = stc.data.astype(np.float32)
        except Exception as e:
            logger.error(f"{label} sample {i}/{batch_size} failed: {e}")
            raise RuntimeError(f"{label} failed at sample {i}: {e}") from e

        if not np.all(np.isfinite(src_data)):
            n_bad = np.sum(~np.isfinite(src_data))
            logger.warning(
                f"{label} sample {i}: {n_bad}/{src_data.size} non-finite values "
                f"in output."
            )
        results.append(src_data)

    out = np.stack(results, axis=0)
    logger.info(
        f"{label} — output shape={out.shape}, "
        f"range=[{out.min():.4f}, {out.max():.4f}]"
    )
    return out


def _build_inverse_operator(
    info: mne.Info, fwd: ForwardClass, label: str
) -> "mne.minimum_norm.InverseOperator":
    """Build shared inverse operator (used by MNE, dSPM, eLORETA)."""
    noise_cov = _make_noise_cov(info)
    try:
        inv_op = make_inverse_operator(
            info, fwd, noise_cov, loose=0.0, depth=None, fixed=True, verbose=False
        )
    except Exception as e:
        logger.error(f"{label} — make_inverse_operator failed: {e}")
        logger.debug(
            f"Forward fields: {[k for k in fwd.keys()]}, "
            f"sol/data shape: {fwd['sol']['data'].shape}, "
            f"source_rr shape: {fwd['source_rr'].shape}, "
            f"source_ori: {fwd['source_ori']}, "
            f"nchan: {fwd['nchan']}, nsource: {fwd['nsource']}"
        )
        raise
    logger.info(f"{label} — inverse operator built (fixed ori, loose=0)")
    return inv_op


def compute_eloreta(
    eeg: NDArray,
    fwd: ForwardClass,
    info: mne.Info,
) -> NDArray:
    """eLORETA inverse solution.

    Args:
        eeg: (batch, 19, 400) EEG windows.
        fwd: MNE Forward from _build_mne_forward.
        info: MNE Info.

    Returns:
        ndarray (batch, 76, 400) source estimates.
    """
    _check_eeg(eeg, "eLORETA")
    inv_op = _build_inverse_operator(info, fwd, "eLORETA")
    return _apply_inverse_to_batch(eeg, info, inv_op, "eLORETA", "eLORETA")


def compute_mne(
    eeg: NDArray,
    fwd: ForwardClass,
    info: mne.Info,
) -> NDArray:
    """MNE (minimum norm) inverse solution.

    Args:
        eeg: (batch, 19, 400) EEG windows.
        fwd: MNE Forward.
        info: MNE Info.

    Returns:
        ndarray (batch, 76, 400) source estimates.
    """
    _check_eeg(eeg, "MNE")
    inv_op = _build_inverse_operator(info, fwd, "MNE")
    return _apply_inverse_to_batch(eeg, info, inv_op, "MNE", "MNE")


def compute_dspm(
    eeg: NDArray,
    fwd: ForwardClass,
    info: mne.Info,
) -> NDArray:
    """dSPM (dynamic statistical parametric mapping) inverse.

    Args:
        eeg: (batch, 19, 400) EEG windows.
        fwd: MNE Forward.
        info: MNE Info.

    Returns:
        ndarray (batch, 76, 400) source estimates.
    """
    _check_eeg(eeg, "dSPM")
    inv_op = _build_inverse_operator(info, fwd, "dSPM")
    return _apply_inverse_to_batch(eeg, info, inv_op, "dSPM", "dSPM")


def compute_lcmv(
    eeg: NDArray,
    fwd: ForwardClass,
    info: mne.Info,
) -> NDArray:
    """LCMV beamformer.

    Args:
        eeg: (batch, 19, 400) EEG windows.
        fwd: MNE Forward.
        info: MNE Info.

    Returns:
        ndarray (batch, 76, 400) source estimates.
    """
    _check_eeg(eeg, "LCMV")
    batch_size = eeg.shape[0]
    ch_names = info["ch_names"]
    results = []

    for i in range(batch_size):
        evoked = mne.EvokedArray(eeg[i].astype(np.float64), info)
        data_cov = _make_data_cov(eeg[i], ch_names)
        try:
            filters = make_lcmv(
                info,
                fwd,
                data_cov,
                reg=0.05,
                noise_cov=None,
                pick_ori=None,
                weight_norm=None,
                verbose=False,
            )
            stc = apply_lcmv(evoked, filters, verbose=False)
            src_data = stc.data.astype(np.float32)
        except Exception as e:
            logger.error(f"LCMV sample {i}/{batch_size} failed: {e}")
            raise RuntimeError(f"LCMV failed at sample {i}: {e}") from e
        results.append(src_data)

    out = np.stack(results, axis=0)
    logger.info(
        f"LCMV — output shape={out.shape}, "
        f"range=[{out.min():.4f}, {out.max():.4f}]"
    )
    return out


def compute_all_baselines(
    eeg: NDArray,
    leadfield: NDArray,
    region_centers: NDArray,
    methods: Optional[list] = None,
) -> Dict[str, Optional[NDArray]]:
    """Run all 4 classical baselines on EEG data.

    Pipeline:
        1. Build MNE Info (channel config) — reused from Phase 1.
        2. Build discrete SourceSpaces from region_centers.
        3. Construct Forward from leadfield + SourceSpaces.
        4. Run eLORETA, MNE, dSPM, LCMV.
        5. Return dict {method: (batch, 76, 400)}.

    Args:
        eeg: (batch, 19, 400) EEG windows.
        leadfield: (19, 76) leadfield matrix.
        region_centers: (76, 3) MNI coordinates in mm.
        methods: Optional list of methods to run.
            Default: ["eloreta", "mne", "dspm", "lcmv"].

    Returns:
        dict with keys matching method names, values (batch, 76, 400) float32.
    """
    if methods is None:
        methods = ["eloreta", "mne", "dspm", "lcmv"]

    _mne_logger = logging.getLogger("mne")
    _mne_logger.setLevel(logging.WARNING)
    logger.info(
        f"compute_all_baselines — ENTER: eeg={eeg.shape}, "
        f"leadfield={leadfield.shape}, centers={region_centers.shape}, "
        f"methods={methods}"
    )
    _check_eeg(eeg, "baselines")

    logger.info("Step 1/3: Building MNE Info...")
    info = _make_info_with_ref()

    logger.info("Step 2/3: Building discrete SourceSpaces...")
    src = _build_source_spaces(region_centers)

    logger.info("Step 3/3: Constructing MNE Forward from leadfield...")
    fwd = _build_mne_forward(leadfield, info, src, region_centers)

    method_map = {
        "eloreta": compute_eloreta,
        "mne": compute_mne,
        "dspm": compute_dspm,
        "lcmv": compute_lcmv,
    }

    results = {}
    for method in methods:
        if method not in method_map:
            logger.warning(f"Unknown method '{method}', skipping.")
            continue
        logger.info(f"Running {method}...")
        try:
            out = method_map[method](eeg, fwd, info)
            results[method] = out
        except Exception as e:
            logger.error(f"{method} failed: {e}")
            results[method] = None

    logger.info(
        f"compute_all_baselines — DONE: {len(results)} methods, "
        f"keys={list(results.keys())}"
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("mne").setLevel(logging.WARNING)
    print("Testing classical_baselines...")

    np.random.seed(42)
    eeg = np.random.randn(2, N_CHANNELS, 400).astype(np.float32)
    leadfield = np.load("data/leadfield_19x76.npy")
    region_centers = np.load("data/region_centers_76.npy")
    info = _make_info_with_ref()
    src = _build_source_spaces(region_centers)
    fwd = _build_mne_forward(leadfield, info, src, region_centers)
    print("  _build_mne_forward:  PASS (Forward: 19 ch x 76 src)")

    out_eloreta = compute_eloreta(eeg, fwd, info)
    assert out_eloreta.shape == (2, 76, 400)
    print(
        f"  compute_eloreta:     PASS "
        f"({out_eloreta.shape}, range=[{out_eloreta.min():.4f}, "
        f"{out_eloreta.max():.4f}])"
    )

    out_mne = compute_mne(eeg, fwd, info)
    assert out_mne.shape == (2, 76, 400)
    print(
        f"  compute_mne:         PASS "
        f"({out_mne.shape}, range=[{out_mne.min():.4f}, "
        f"{out_mne.max():.4f}])"
    )

    out_dspm = compute_dspm(eeg, fwd, info)
    assert out_dspm.shape == (2, 76, 400)
    print(
        f"  compute_dspm:        PASS "
        f"({out_dspm.shape}, range=[{out_dspm.min():.4f}, "
        f"{out_dspm.max():.4f}])"
    )

    out_lcmv = compute_lcmv(eeg, fwd, info)
    assert out_lcmv.shape == (2, 76, 400)
    print(
        f"  compute_lcmv:        PASS "
        f"({out_lcmv.shape}, range=[{out_lcmv.min():.4f}, "
        f"{out_lcmv.max():.4f}])"
    )

    baselines = compute_all_baselines(eeg, leadfield, region_centers)
    assert set(baselines.keys()) == {"eloreta", "mne", "dspm", "lcmv"}
    print("  compute_all_baselines: PASS (keys:", list(baselines.keys()), ")")

    print("\nAll classical baseline tests passed!")
