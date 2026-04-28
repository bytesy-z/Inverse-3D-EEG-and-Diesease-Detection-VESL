"""
CMA-ES objective function for fitting x0 (excitability) per patient.

J(x0) = w_source * J_source(x0) + w_eeg * J_eeg(x0) + w_reg * J_reg(x0)

Where:
  J_source: MSE between PhysDeepSIF-predicted source activity and TVB-simulated source
  J_eeg:    PSD-based error between simulated EEG and real patient EEG
  J_reg:    Sparsity regularizer ||x0 - x0_healthy||² (push toward healthy baseline)

Each CMA-ES evaluation runs a 4s TVB simulation (~1s compute on CPU).
"""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
import torch

from ..phase1_forward.epileptor_simulator import run_simulation
from ..phase1_forward.parameter_sampler import sample_simulation_parameters
from ..phase1_forward.synthetic_dataset import (
    project_to_eeg,
    apply_skull_attenuation_filter,
    apply_spectral_shaping,
    add_measurement_noise,
)
from ..phase2_network.physdeepsif import build_physdeepsif


def _compute_psd(eeg: NDArray, sfreq: int = 200, fmin: float = 0.5, fmax: float = 70.0) -> NDArray:
    """Compute average PSD across 19 EEG channels using Welch's method."""
    n_channels, n_samples = eeg.shape
    psds = []
    for ch_idx in range(n_channels):
        freqs, psd = welch(eeg[ch_idx], sfreq=sfreq, nperseg=min(256, n_samples))
        mask = (freqs >= fmin) & (freqs <= fmax)
        psds.append(psd[mask])
    return np.mean(psds, axis=0)


def objective(
    x0: NDArray,
    patient_eeg_psd: NDArray,
    leadfield: NDArray,
    connectivity: NDArray,
    region_centers: NDArray,
    region_labels: list,
    tract_lengths: NDArray,
    config: dict,
    model: torch.nn.Module,
    norm_stats: dict,
    device: str = "cpu",
    w_source: float = 0.4,
    w_eeg: float = 0.4,
    w_reg: float = 0.2,
) -> float:
    # --------------------------
    # 1. Run TVB simulation with candidate x0
    # --------------------------
    sim_params = sample_simulation_parameters()
    sim_params["x0"] = x0.astype(np.float64)
    sim_length_ms = config.get("parameter_inversion", {}).get("sim_length_ms", 4000)
    
    source_activity = run_simulation(
        connectivity=connectivity,
        tract_lengths=tract_lengths,
        simulation_length_ms=sim_length_ms,
        params=sim_params,
    )  # Returns (76, n_time) source activity

    # --------------------------
    # 2. Project source to EEG and match patient EEG characteristics
    # --------------------------
    sim_eeg = project_to_eeg(source_activity, leadfield)
    sim_eeg = apply_skull_attenuation_filter(sim_eeg)
    sim_eeg = apply_spectral_shaping(sim_eeg, sfreq=200)
    sim_eeg = add_measurement_noise(sim_eeg, snr_db=config.get("synthetic_data", {}).get("snr_db", 20))

    # --------------------------
    # 3. J_source: MSE between PhysDeepSIF output and simulated source
    # --------------------------
    # Match training preprocessing: per-channel de-mean + global z-score
    sim_eeg_demeaned = sim_eeg - sim_eeg.mean(axis=-1, keepdims=True)
    sim_eeg_norm = (sim_eeg_demeaned - norm_stats["eeg_mean"]) / (norm_stats["eeg_std"] + 1e-8)

    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.from_numpy(sim_eeg_norm).float().to(device).unsqueeze(0)
        pred_source = model(eeg_tensor).cpu().numpy().squeeze(0)  # (76, n_time)

    # Denormalize to original scale
    pred_source = pred_source * (norm_stats["src_std"] + 1e-8) + norm_stats["src_mean"]
    J_source = np.mean((pred_source - source_activity) ** 2)

    # --------------------------
    # 4. J_eeg: PSD error between simulated and patient EEG
    # --------------------------
    sim_psd = _compute_psd(sim_eeg, sfreq=200)
    min_len = min(sim_psd.shape[0], patient_eeg_psd.shape[0])
    J_eeg = np.mean((sim_psd[:min_len] - patient_eeg_psd[:min_len]) ** 2)

    # --------------------------
    # 5. J_reg: Regularize toward healthy x0 baseline (-2.2 per Epileptor literature)
    # --------------------------
    healthy_x0 = config.get("parameter_inversion", {}).get("healthy_x0", -2.2)
    x0_healthy = np.full(76, healthy_x0)
    J_reg = np.sum((x0 - x0_healthy) ** 2)

    return float(w_source * J_source + w_eeg * J_eeg + w_reg * J_reg)