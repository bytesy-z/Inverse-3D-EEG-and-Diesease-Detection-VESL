import numpy as np
import pytest
from src.phase2_network.metrics import compute_dipole_localization_error

N_REGIONS = 76
N_TIMES = 400


@pytest.fixture
def region_centers():
    rng = np.random.RandomState(0)
    return rng.randn(N_REGIONS, 3).astype(np.float32) * 50


@pytest.fixture
def single_epi_mask():
    mask = np.zeros((1, N_REGIONS), dtype=bool)
    mask[0, 15] = True
    return mask


def test_perfect_prediction(region_centers, single_epi_mask):
    rng = np.random.RandomState(1)
    sources = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    sources[:, 15, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    dle = compute_dipole_localization_error(
        sources, sources, region_centers, epileptogenic_mask=single_epi_mask
    )
    assert dle < 1.0, f"Expected DLE≈0 when power concentrated in epi region, got {dle}"


def test_worst_prediction(region_centers, single_epi_mask):
    rng = np.random.RandomState(2)
    true_sources = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true_sources[:, 15, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    pred_sources = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    pred_sources[:, 61, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    centers = region_centers
    dle = compute_dipole_localization_error(
        pred_sources, true_sources, centers, epileptogenic_mask=single_epi_mask
    )
    expected_dist = float(np.linalg.norm(centers[61] - centers[15]))
    assert dle == pytest.approx(expected_dist, rel=0.1), f"Expected DLE≈{expected_dist}, got {dle}"


def test_healthy_only_fallback(region_centers):
    rng = np.random.RandomState(3)
    true_sources = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true_sources[:, 30, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    pred_sources = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    pred_sources[:, 30, :] = rng.randn(N_TIMES).astype(np.float32) * 4.0
    healthy_mask = np.zeros((1, N_REGIONS), dtype=bool)
    dle = compute_dipole_localization_error(
        pred_sources, true_sources, region_centers, epileptogenic_mask=healthy_mask
    )
    assert dle < 1.0, f"Expected DLE≈0 for healthy fallback, got {dle}"


def test_single_region_epi(region_centers):
    rng = np.random.RandomState(4)
    mask = np.zeros((1, N_REGIONS), dtype=bool)
    mask[0, 7] = True
    pred = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    pred[:, 7, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    true = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true[:, 7, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    dle = compute_dipole_localization_error(pred, true, region_centers, epileptogenic_mask=mask)
    assert dle < 1.0, f"Expected DLE≈0 for single-region epi, got {dle}"


def test_multi_focal_power_weighted_centroid(region_centers):
    rng = np.random.RandomState(5)
    mask = np.zeros((1, N_REGIONS), dtype=bool)
    mask[0, [10, 20]] = True
    pred = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    pred[:, 5, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    true = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true[:, 10, :] = rng.randn(N_TIMES).astype(np.float32) * 5.0
    true[:, 20, :] = rng.randn(N_TIMES).astype(np.float32) * 4.0
    dle = compute_dipole_localization_error(pred, true, region_centers, epileptogenic_mask=mask)
    pred_centroid = region_centers[5]
    w10, w20 = 25.0, 16.0  # power = amp^2
    true_centroid = (w10 * region_centers[10] + w20 * region_centers[20]) / (w10 + w20)
    expected = float(np.linalg.norm(pred_centroid - true_centroid))
    assert dle == pytest.approx(expected, rel=0.1), f"Expected DLE≈{expected:.1f}, got {dle:.1f}"


def test_zero_variance_prediction_no_nan(region_centers, single_epi_mask):
    pred = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true[:, 15, :] = 1.0
    dle = compute_dipole_localization_error(pred, true, region_centers, epileptogenic_mask=single_epi_mask)
    assert np.isfinite(dle), f"Expected finite DLE, got {dle}"


def test_shape_mismatch_raises_value_error(region_centers, single_epi_mask):
    pred = np.zeros((1, N_REGIONS, N_TIMES), dtype=np.float32)
    true_wrong = np.zeros((1, 76, 200), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_dipole_localization_error(pred, true_wrong, region_centers, epileptogenic_mask=single_epi_mask)
