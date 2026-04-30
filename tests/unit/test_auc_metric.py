import numpy as np
import pytest
from src.phase2_network.metrics import compute_auc_epileptogenicity

N_REGIONS = 76
N_TIMES = 400


@pytest.fixture
def epi_mask():
    mask = np.zeros((4, N_REGIONS), dtype=bool)
    mask[:, 10:15] = True
    return mask


def test_perfect_classifier_auc_one(epi_mask):
    rng = np.random.RandomState(0)
    sources = np.zeros((4, N_REGIONS, N_TIMES), dtype=np.float32)
    sources[:, 10:15, :] = 10.0
    sources[:, :10, :] = rng.randn(4, 10, N_TIMES).astype(np.float32) * 0.01
    sources[:, 15:, :] = rng.randn(4, 61, N_TIMES).astype(np.float32) * 0.01
    auc = compute_auc_epileptogenicity(sources, epi_mask)
    assert auc == pytest.approx(1.0, abs=0.01), f"Expected AUC≈1.0, got {auc}"


def test_random_classifier_auc_half(epi_mask):
    rng = np.random.RandomState(1)
    sources = rng.randn(4, N_REGIONS, N_TIMES).astype(np.float32)
    auc = compute_auc_epileptogenicity(sources, epi_mask)
    assert auc == pytest.approx(0.5, abs=0.15), f"Expected AUC≈0.5, got {auc}"


def test_inverted_classifier_auc_zero(epi_mask):
    sources = np.zeros((4, N_REGIONS, N_TIMES), dtype=np.float32)
    sources[:, 10:15, :] = 0.01
    sources[:, :10, :] = 10.0
    sources[:, 15:, :] = 10.0
    auc = compute_auc_epileptogenicity(sources, epi_mask)
    assert auc == pytest.approx(0.0, abs=0.01), f"Expected AUC≈0.0, got {auc}"


def test_all_healthy_returns_half():
    sources = np.random.RandomState(2).randn(4, N_REGIONS, N_TIMES).astype(np.float32)
    all_healthy = np.zeros((4, N_REGIONS), dtype=bool)
    auc = compute_auc_epileptogenicity(sources, all_healthy)
    assert auc == 0.5, f"Expected 0.5 for all-healthy, got {auc}"


def test_all_epi_returns_half():
    sources = np.random.RandomState(3).randn(4, N_REGIONS, N_TIMES).astype(np.float32)
    all_epi = np.ones((4, N_REGIONS), dtype=bool)
    auc = compute_auc_epileptogenicity(sources, all_epi)
    assert auc == 0.5, f"Expected 0.5 for all-epi, got {auc}"
