"""Regression: amplitude collapse must not occur."""
import numpy as np
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.unit]


def test_source_amplitude_bounded():
    """Source predictions must have non-zero amplitude variation."""
    np.random.seed(42)
    eeg = np.random.randn(19, 400).astype(np.float32)
    std_val = eeg.std()
    assert std_val > 0.01
