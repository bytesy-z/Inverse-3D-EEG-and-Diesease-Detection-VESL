import logging
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import decimate, welch
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Difference
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.monitors import Raw
from tvb.simulator.noise import Additive
from tvb.simulator.simulator import Simulator

from src.phase1_forward.epileptor_simulator import (
    _factorize_decimation,
    build_tvb_connectivity,
    configure_epileptor,
)

logger = logging.getLogger(__name__)


def compute_eeg_psd(
    eeg: NDArray[np.float64],
    fs: float = 200.0,
) -> NDArray[np.float64]:
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]
    elif eeg.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got {eeg.ndim}D")

    n_channels, n_times = eeg.shape
    _, psd = welch(eeg, fs=fs, nperseg=128, noverlap=64, axis=-1)
    return psd


def _run_simulation_cached(
    x0_vector: NDArray[np.float64],
    connectivity: Connectivity,
    global_coupling: float = 1.0,
    noise_intensity: float = 0.001,
    conduction_speed: float = 4.0,
    Iext1: float = 3.1,
    Iext2: float = 0.45,
    tau0: float = 2857.0,
    tau2: float = 10.0,
    simulation_length_ms: float = 6000.0,
    dt: float = 0.1,
) -> NDArray[np.float64]:
    params: Dict = {
        "x0_vector": x0_vector,
        "global_coupling": global_coupling,
        "noise_intensity": noise_intensity,
        "conduction_speed": conduction_speed,
        "Iext1": Iext1,
        "Iext2": Iext2,
        "tau0": tau0,
        "tau2": tau2,
    }

    model = configure_epileptor(params)

    nsig = np.array([
        noise_intensity, noise_intensity, 0.0,
        noise_intensity, noise_intensity, 0.0,
    ])
    noise = Additive(nsig=nsig)
    noise.random_stream.seed(
        int(np.abs(hash(tuple(x0_vector[:5]))) % (2**31))
    )
    integrator = HeunStochastic(dt=dt, noise=noise)
    coupling = Difference(a=np.array([global_coupling]))
    monitor = Raw()

    simulator = Simulator(
        model=model,
        connectivity=connectivity,
        coupling=coupling,
        integrator=integrator,
        monitors=[monitor],
        simulation_length=simulation_length_ms,
    )
    simulator.configure()

    raw_output = []
    for (time_array, data_array), in simulator():
        raw_output.append(data_array)
    full_output = np.concatenate(raw_output, axis=0)

    if full_output.ndim == 3 and full_output.shape[2] == 1:
        lfp_raw = full_output[:, :, 0]
    elif full_output.ndim == 4:
        x1 = full_output[:, 0, :, 0]
        x2 = full_output[:, 3, :, 0]
        lfp_raw = x2 - x1
    else:
        raise ValueError(f"Unexpected output shape: {full_output.shape}")

    actual_raw_fs = 2.0 * (1000.0 / dt)
    total_decimation = int(actual_raw_fs / 200.0)
    stages = _factorize_decimation(total_decimation)

    n_raw, n_reg = lfp_raw.shape
    expected_len = n_raw
    for s in stages:
        expected_len //= s

    lfp_dec = np.zeros((expected_len, n_reg), dtype=np.float64)
    for r in range(n_reg):
        sig = lfp_raw[:, r].astype(np.float64)
        for s in stages:
            sig = decimate(sig, s, ftype="fir")
        lfp_dec[:, r] = sig

    transient = int(2000.0 / 1000.0 * 200.0)
    src = lfp_dec[transient:, :].T

    return src.astype(np.float64)


def build_objective(
    target_eeg: NDArray[np.float64],
    leadfield: NDArray[np.float64],
    connectivity_weights: NDArray[np.float64],
    region_centers: NDArray[np.float64],
    region_labels: list,
    tract_lengths: NDArray[np.float64],
    fs: float = 200.0,
    w_reg: float = 0.1,
    healthy_baseline: float = -2.2,
) -> Callable[[NDArray[np.float64]], float]:
    if target_eeg.ndim == 1:
        pass
    elif target_eeg.ndim != 2:
        raise ValueError(f"target_eeg must be 1D or 2D, got {target_eeg.ndim}D")

    connectivity = build_tvb_connectivity(
        weights=connectivity_weights,
        centres=region_centers,
        region_labels=region_labels,
        tract_lengths=tract_lengths,
        conduction_speed=4.0,
    )

    target_psd = compute_eeg_psd(target_eeg, fs=fs)
    if target_psd.ndim == 2:
        target_psd = target_psd.mean(axis=0)

    window_samples = int(2.0 * fs)
    n_windows_from_4s = int((4.0 * fs) / window_samples)

    def objective(x0_candidate: NDArray[np.float64]) -> float:
        x0_candidate = np.asarray(x0_candidate, dtype=np.float64).ravel()
        if x0_candidate.size != connectivity_weights.shape[0]:
            raise ValueError(
                f"x0_candidate size {x0_candidate.size} != "
                f"{connectivity_weights.shape[0]} regions"
            )

        try:
            source = _run_simulation_cached(x0_candidate, connectivity)
        except Exception as e:
            logger.debug(f"Simulation failed for x0 candidate: {e}")
            return 1e10

        if source.shape[1] < window_samples * n_windows_from_4s:
            n_avail = source.shape[1] // window_samples
            psd_windows = []
            for i in range(n_avail):
                start = i * window_samples
                seg = source[:, start:start + window_samples]
                eeg_seg = leadfield @ seg
                psd_w = compute_eeg_psd(eeg_seg, fs=fs)
                if psd_w.ndim == 2:
                    psd_w = psd_w.mean(axis=0)
                psd_windows.append(psd_w)
        else:
            psd_windows = []
            for i in range(n_windows_from_4s):
                start = i * window_samples
                seg = source[:, start:start + window_samples]
                eeg_seg = leadfield @ seg
                psd_w = compute_eeg_psd(eeg_seg, fs=fs)
                if psd_w.ndim == 2:
                    psd_w = psd_w.mean(axis=0)
                psd_windows.append(psd_w)

        psd_sim = np.mean(psd_windows, axis=0)

        mse = float(np.mean((psd_sim - target_psd) ** 2))
        reg = float(w_reg * np.sum((x0_candidate - healthy_baseline) ** 2))
        return mse + reg

    return objective
