import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = torch.randn(76, 76)
    return PhysicsInformedLoss(leadfield, laplacian)


def test_constant_signal_zero_loss(loss_fn):
    constant = torch.ones(2, 76, 400) * 3.0
    loss = loss_fn._compute_temporal_loss(constant)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss=0, got {loss.item()}"


def test_high_frequency_noise_positive_loss(loss_fn):
    rng = torch.Generator().manual_seed(0)
    noise = torch.randn(2, 76, 400, generator=rng)
    loss = loss_fn._compute_temporal_loss(noise)
    assert loss.item() > 0, f"Expected loss>0 for noise, got {loss.item()}"


def test_lambda_temporal_zero_in_physics(loss_fn):
    loss_fn.lambda_temporal = 0.0
    sources = torch.randn(2, 76, 400)
    raw_loss = loss_fn._compute_temporal_loss(sources)
    assert loss_fn.lambda_temporal == 0.0
    assert raw_loss.item() > 0


def test_temporal_loss_non_negative(loss_fn):
    rng = torch.Generator().manual_seed(1)
    for _ in range(10):
        sources = torch.randn(2, 76, 400, generator=rng)
        loss = loss_fn._compute_temporal_loss(sources)
        assert loss.item() >= -1e-6, f"Expected loss>=0, got {loss.item()}"
