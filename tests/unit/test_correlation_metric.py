import numpy as np
import pytest
from src.phase2_network.metrics import compute_temporal_correlation

N_REGIONS = 76
N_TIMES = 400


def test_identical_signals_corr_one():
    rng = np.random.RandomState(0)
    sources = rng.randn(2, N_REGIONS, N_TIMES).astype(np.float32)
    corr = compute_temporal_correlation(sources, sources)
    assert corr == pytest.approx(1.0, abs=1e-6), f"Expected Corr=1.0, got {corr}"


def test_anti_correlated_signals_corr_neg_one():
    rng = np.random.RandomState(1)
    sources = rng.randn(2, N_REGIONS, N_TIMES).astype(np.float32)
    inverted = -sources.copy()
    corr = compute_temporal_correlation(sources, inverted)
    assert corr == pytest.approx(-1.0, abs=1e-6), f"Expected Corr=-1.0, got {corr}"


def test_orthogonal_signals_corr_zero():
    rng = np.random.RandomState(2)
    true = rng.randn(2, N_REGIONS, N_TIMES).astype(np.float32)
    pred = rng.randn(2, N_REGIONS, N_TIMES).astype(np.float32)
    corr = compute_temporal_correlation(pred, true)
    assert abs(corr) < 0.15, f"Expected Corr≈0, got {corr}"


def test_dc_offset_does_not_affect_corr():
    rng = np.random.RandomState(3)
    base = rng.randn(1, N_REGIONS, N_TIMES).astype(np.float32)
    offset = base.copy() + 100.0
    corr = compute_temporal_correlation(base, offset)
    assert corr == pytest.approx(1.0, abs=1e-6), f"DC offset should not affect correlation, got {corr}"


def test_shape_mismatch_raises_value_error():
    pred = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true_wrong = np.zeros((1, 76, 200), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_temporal_correlation(pred, true_wrong)
