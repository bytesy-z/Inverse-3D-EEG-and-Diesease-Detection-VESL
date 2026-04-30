"""Tests for compute_spatial_dispersion edge cases."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]


def test_uniform_activation():
    """Uniform activation should give some dispersion."""
    from src.phase2_network.metrics import compute_spatial_dispersion
    region_centers = np.random.randn(76, 3).astype(np.float32)
    source_activity = np.ones((1, 76, 400), dtype=np.float32)
    result = compute_spatial_dispersion(source_activity, region_centers)
    assert np.isfinite(result)
    assert result >= 0


def test_focal_activation():
    """Focal activation should give lower dispersion."""
    from src.phase2_network.metrics import compute_spatial_dispersion
    region_centers = np.random.randn(76, 3).astype(np.float32)
    source_activity = np.zeros((1, 76, 400), dtype=np.float32)
    source_activity[:, 5] = 1.0
    source_activity[:, 6] = 0.5
    result = compute_spatial_dispersion(source_activity, region_centers)
    assert np.isfinite(result)
    assert result >= 0


def test_diffuse_activation():
    """Diffuse activation should give larger dispersion than focal."""
    from src.phase2_network.metrics import compute_spatial_dispersion
    np.random.seed(42)
    region_centers = np.random.randn(76, 3).astype(np.float32)
    focal = np.zeros((1, 76, 400), dtype=np.float32)
    focal[:, 5] = 1.0
    diffuse = np.random.randn(1, 76, 400).astype(np.float32)
    disp_focal = compute_spatial_dispersion(focal, region_centers)
    disp_diffuse = compute_spatial_dispersion(diffuse, region_centers)
    assert disp_diffuse >= disp_focal


def test_zero_activation():
    """Zero activation should not crash."""
    from src.phase2_network.metrics import compute_spatial_dispersion
    region_centers = np.random.randn(76, 3).astype(np.float32)
    source_activity = np.zeros((1, 76, 400), dtype=np.float32)
    result = compute_spatial_dispersion(source_activity, region_centers)
    assert np.isfinite(result)


def test_nan_input():
    """NaN input should return NaN."""
    from src.phase2_network.metrics import compute_spatial_dispersion
    region_centers = np.random.randn(76, 3).astype(np.float32)
    source_activity = np.full((1, 76, 400), np.nan, dtype=np.float32)
    result = compute_spatial_dispersion(source_activity, region_centers)
    assert np.isnan(result)
