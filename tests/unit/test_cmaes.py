"""Tests for CMA-ES inversion module."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.slow]


class TestComputeConcordance:

    def _make_top_indices(self, indices):
        """Helper: create an array where given indices get descending high values."""
        arr = np.zeros(76, dtype=np.float64)
        values = np.arange(len(indices), 0, -1, dtype=np.float64)
        arr[list(indices)] = values
        return arr

    def test_high_overlap(self):
        from src.phase4_inversion.concordance import compute_concordance
        heuristic = self._make_top_indices(range(10))
        biophysical = self._make_top_indices(range(10))
        result = compute_concordance(heuristic, biophysical)
        assert result["overlap"] == 10
        assert result["tier"] == "HIGH"

    def test_moderate_overlap(self):
        from src.phase4_inversion.concordance import compute_concordance
        heuristic = self._make_top_indices(range(10))
        biophysical = self._make_top_indices(range(6, 16))
        result = compute_concordance(heuristic, biophysical)
        assert result["overlap"] == 4
        assert result["tier"] == "MODERATE"

    def test_moderate_overlap_2(self):
        from src.phase4_inversion.concordance import compute_concordance
        heuristic = self._make_top_indices(range(10))
        biophysical = self._make_top_indices(range(8, 18))
        result = compute_concordance(heuristic, biophysical)
        assert result["overlap"] == 2
        assert result["tier"] == "MODERATE"

    def test_low_overlap(self):
        from src.phase4_inversion.concordance import compute_concordance
        heuristic = self._make_top_indices(range(10))
        biophysical = self._make_top_indices(range(9, 19))
        result = compute_concordance(heuristic, biophysical)
        assert result["overlap"] == 1
        assert result["tier"] == "LOW"


class TestComputeBiophysicalEI:
    def test_zero_temperature(self):
        from src.phase4_inversion.epileptogenicity_index import compute_biophysical_ei
        x0 = np.zeros(76)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            compute_biophysical_ei(x0, temperature=0.0)

    def test_negative_x0(self):
        from src.phase4_inversion.epileptogenicity_index import compute_biophysical_ei
        x0 = np.full(76, -5.0)
        result = compute_biophysical_ei(x0, temperature=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_extreme_positive_x0(self):
        from src.phase4_inversion.epileptogenicity_index import compute_biophysical_ei
        x0 = np.full(76, 10.0)
        result = compute_biophysical_ei(x0, temperature=1.0)
        assert np.all(result >= 0)
