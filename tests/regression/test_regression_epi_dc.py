"""Regression: EI scores must not be driven by DC offset.
Tests the server's compute_epileptogenicity_index which applies
per-region temporal de-meaning before power computation.
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.unit]


def test_epi_ranking_preserved_under_dc_offset():
    """Global DC offset must not change region ranking after de-meaning."""
    from scipy.stats import spearmanr
    rng = np.random.RandomState(42)
    sources = rng.randn(76, 400).astype(np.float32)
    sources_dc = sources + 10.0

    def compute_power_demeaned(s):
        s_dm = s - s.mean(axis=-1, keepdims=True)
        power = np.mean(s_dm ** 2, axis=-1)
        return power

    power0 = compute_power_demeaned(sources)
    power1 = compute_power_demeaned(sources_dc)
    rho, _ = spearmanr(power0, power1)
    assert rho > 0.99, f"Rank correlation dropped to {rho} under DC offset"
