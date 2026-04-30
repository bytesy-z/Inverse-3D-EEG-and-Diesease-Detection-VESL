import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = torch.randn(76, 76)
    return PhysicsInformedLoss(leadfield, laplacian)


def test_perfect_forward_consistency_loss_zero(loss_fn):
    rng = torch.Generator().manual_seed(0)
    sources = torch.randn(2, 76, 400, generator=rng)
    eeg = torch.einsum('ij,bjt->bit', loss_fn.leadfield, sources)
    loss = loss_fn._compute_forward_loss(sources, eeg)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss≈0, got {loss.item()}"


def test_random_prediction_positive_loss(loss_fn):
    rng = torch.Generator().manual_seed(1)
    pred = torch.randn(2, 76, 400, generator=rng)
    eeg = torch.randn(2, 19, 400, generator=rng)
    loss = loss_fn._compute_forward_loss(pred, eeg)
    assert loss.item() > 0, f"Expected loss>0, got {loss.item()}"


def test_beta_zero_still_returns_raw_component(loss_fn):
    loss_fn.beta = 0.0
    pred = torch.randn(2, 76, 400)
    eeg = torch.randn(2, 19, 400)
    loss = loss_fn._compute_forward_loss(pred, eeg)
    assert loss.item() > 0, "Raw forward loss should be >0 even when beta=0"


def test_eeg_demeaning_inside_forward_loss(loss_fn):
    rng = torch.Generator().manual_seed(2)
    sources = torch.randn(2, 76, 400, generator=rng)
    eeg = torch.einsum('ij,bjt->bit', loss_fn.leadfield, sources)
    eeg_with_dc = eeg + 5.0
    loss_no_dc = loss_fn._compute_forward_loss(sources, eeg)
    loss_with_dc = loss_fn._compute_forward_loss(sources, eeg_with_dc)
    assert loss_with_dc.item() == pytest.approx(loss_no_dc.item(), abs=0.5), (
        "DC offset should have minor effect due to variance-normalised denominator"
    )


def test_gradient_flows_forward_loss(loss_fn):
    pred = torch.randn(2, 76, 400, requires_grad=True)
    eeg = torch.randn(2, 19, 400)
    loss = loss_fn._compute_forward_loss(pred, eeg)
    loss.backward()
    assert pred.grad is not None, "Gradient did not flow"
    assert pred.grad.abs().sum().item() > 0, "Gradient is zero"
