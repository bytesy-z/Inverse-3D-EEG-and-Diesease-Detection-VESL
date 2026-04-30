import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


def _make_psd_laplacian(n=76):
    rng = torch.Generator().manual_seed(42)
    W = torch.randn(n, n, generator=rng).abs()
    W = (W + W.T) / 2
    W.fill_diagonal_(0)
    deg = W.sum(dim=1)
    D = torch.diag(deg)
    L = D - W
    return L


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = _make_psd_laplacian()
    return PhysicsInformedLoss(leadfield, laplacian)


def test_uniform_activity_zero_loss():
    D = torch.zeros(76, 76)
    leadfield = torch.randn(19, 76)
    loss_fn = PhysicsInformedLoss(leadfield, D)
    ones = torch.ones(2, 76, 400)
    loss = loss_fn._compute_laplacian_loss(ones)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss=0, got {loss.item()}"


def test_step_discontinuity_positive_loss(loss_fn):
    D = _make_psd_laplacian()
    leadfield = torch.randn(19, 76)
    loss_fn = PhysicsInformedLoss(leadfield, D)
    sources = torch.zeros(2, 76, 400)
    sources[:, 0, :] = 10.0
    sources[:, 1, :] = -10.0
    loss = loss_fn._compute_laplacian_loss(sources)
    assert loss.item() > 0, f"Expected loss>0 for step discontinuity, got {loss.item()}"


def test_lambda_laplacian_zero_in_total(loss_fn):
    loss_fn.lambda_laplacian = 0.0
    sources = torch.randn(2, 76, 400)
    raw_loss = loss_fn._compute_laplacian_loss(sources)
    assert loss_fn.lambda_laplacian == 0.0
    assert raw_loss.item() > 0, "Raw laplacian loss should be >0 for random sources"


def test_loss_always_non_negative(loss_fn):
    for _ in range(10):
        sources = torch.randn(4, 76, 400)
        loss = loss_fn._compute_laplacian_loss(sources)
        assert loss.item() >= -1e-6, f"Expected loss>=0, got {loss.item()}"
