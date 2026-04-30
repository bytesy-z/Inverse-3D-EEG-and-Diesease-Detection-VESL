import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = torch.randn(76, 76)
    return PhysicsInformedLoss(leadfield, laplacian)


def test_activity_below_bound_zero_loss(loss_fn):
    small = torch.ones(2, 76, 400) * 0.5
    loss = loss_fn._compute_amplitude_loss(small)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss=0, got {loss.item()}"


def test_activity_above_bound_positive_loss(loss_fn):
    large = torch.ones(2, 76, 400) * 5.0
    loss = loss_fn._compute_amplitude_loss(large)
    assert loss.item() > 0, f"Expected loss>0 for activity above bound, got {loss.item()}"


def test_loss_proportional_to_excess(loss_fn):
    slightly_above = torch.ones(2, 76, 400) * 4.0
    far_above = torch.ones(2, 76, 400) * 10.0
    loss_small = loss_fn._compute_amplitude_loss(slightly_above)
    loss_large = loss_fn._compute_amplitude_loss(far_above)
    assert loss_large.item() > loss_small.item(), (
        "Larger excess should produce larger loss"
    )


def test_amplitude_loss_non_negative(loss_fn):
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        sources = torch.randn(2, 76, 400, generator=rng) * 5.0
        loss = loss_fn._compute_amplitude_loss(sources)
        assert loss.item() >= -1e-6, f"Expected loss>=0, got {loss.item()}"
