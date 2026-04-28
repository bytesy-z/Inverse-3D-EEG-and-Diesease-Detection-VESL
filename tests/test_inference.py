import numpy as np
import torch


def test_ei_computation(model, synthetic_sample, normalization_stats):
    """Epileptogenicity index returns valid scores."""
    eeg_raw, mask = synthetic_sample
    # Apply preprocessing
    eeg_ac = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
    eps = 1e-7
    eeg_norm = (eeg_ac - normalization_stats["eeg_mean"]) / (normalization_stats["eeg_std"] + eps)

    with torch.no_grad():
        sources = model(eeg_norm)

    # Simple EI: region power
    power = sources.pow(2).mean(dim=-1).numpy().flatten()
    assert power.shape == (76,)
    assert np.all(power >= 0), "Power should be non-negative"
    assert power.sum() > 0, "Total power should be positive"


def test_source_activity_range(model, synthetic_sample, normalization_stats):
    """Predicted source activity is within reasonable range after denorm."""
    eeg_raw, _ = synthetic_sample
    eeg_ac = eeg_raw - eeg_raw.mean(dim=-1, keepdim=True)
    eps = 1e-7
    eeg_norm = (eeg_ac - normalization_stats["eeg_mean"]) / (normalization_stats["eeg_std"] + eps)

    with torch.no_grad():
        sources = model(eeg_norm)

    # Denormalize: reverse z-score
    src_denorm = sources * (normalization_stats["src_std"] + eps) + normalization_stats["src_mean"]
    assert src_denorm.abs().max() < 10.0, f"Denormalized source too large: {src_denorm.abs().max()}"
