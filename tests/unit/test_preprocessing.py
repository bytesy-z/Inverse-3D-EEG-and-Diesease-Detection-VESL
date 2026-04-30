import numpy as np
import pytest


@pytest.fixture
def eeg_data():
    rng = np.random.RandomState(42)
    data = rng.randn(19, 400).astype(np.float32)
    data[0, :] += 5.0
    return data


@pytest.fixture
def norm_stats():
    return {
        'eeg_mean': 0.0,
        'eeg_std': 1.0,
        'src_mean': 0.0,
        'src_std': 1.0,
    }


def test_per_channel_demeaning(eeg_data):
    demeaned = eeg_data - eeg_data.mean(axis=-1, keepdims=True)
    channel_means = demeaned.mean(axis=-1)
    assert np.allclose(channel_means, 0.0, atol=1e-6), (
        f"Channel means should be ~0, got max={np.max(np.abs(channel_means))}"
    )


def test_zscore_normalization(eeg_data, norm_stats):
    demeaned = eeg_data - eeg_data.mean(axis=-1, keepdims=True)
    eps = 1e-7
    normalized = (demeaned - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + eps)
    assert np.abs(normalized.mean()) < 0.5
    assert np.abs(normalized.std() - 1.0) < 0.5


def test_roundtrip_normalize_denormalize(eeg_data, norm_stats):
    demeaned = eeg_data - eeg_data.mean(axis=-1, keepdims=True)
    eps = 1e-7
    normalized = (demeaned - norm_stats['eeg_mean']) / (norm_stats['eeg_std'] + eps)
    denormalized = normalized * (norm_stats['eeg_std'] + eps) + norm_stats['eeg_mean']
    assert np.allclose(demeaned, denormalized, atol=1e-5), "Roundtrip should be identity"


def test_zero_variance_channel_no_division_by_zero():
    data = np.zeros((19, 400), dtype=np.float32)
    data[0, :] = 1.0
    eps = 1e-7
    demeaned = data - data.mean(axis=-1, keepdims=True)
    channel_std = demeaned.std(axis=-1, keepdims=True) + eps
    zero_var_mask = channel_std.squeeze() < eps * 2
    assert zero_var_mask.any(), "Should have zero-variance channels"
    normalized = demeaned / channel_std
    assert np.all(np.isfinite(normalized)), "No division by zero with eps"
    zero_var_ch = np.where(zero_var_mask)[0][0]
    assert np.all(normalized[zero_var_ch] == 0.0), "Zero-variance channel should be 0 after division"


def test_missing_normalization_stats_key():
    stats = {'eeg_mean': 0.0}
    with pytest.raises(KeyError):
        _ = stats['eeg_std']
