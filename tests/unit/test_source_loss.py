import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = torch.randn(76, 76)
    laplacian = (laplacian + laplacian.T) / 2
    return PhysicsInformedLoss(leadfield, laplacian)


def test_zero_error_zero_loss(loss_fn):
    sources = torch.randn(4, 76, 400)
    loss = loss_fn._compute_source_loss(sources, sources)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected loss=0, got {loss.item()}"


def test_nonzero_error_positive_loss(loss_fn):
    rng = torch.Generator().manual_seed(42)
    pred = torch.randn(4, 76, 400, generator=rng)
    true = torch.randn(4, 76, 400, generator=torch.Generator().manual_seed(99))
    loss = loss_fn._compute_source_loss(pred, true)
    assert loss.item() > 0, f"Expected loss>0, got {loss.item()}"


def test_gradient_flows(loss_fn):
    pred = torch.randn(2, 76, 400, requires_grad=True)
    true = torch.randn(2, 76, 400)
    loss = loss_fn._compute_source_loss(pred, true)
    loss.backward()
    assert pred.grad is not None, "Gradient did not flow"
    assert pred.grad.abs().sum().item() > 0, "Gradient is zero"


def test_no_class_balance_applied(loss_fn):
    pred = torch.randn(2, 76, 400)
    true = torch.randn(2, 76, 400)
    loss_no_mask = loss_fn._compute_source_loss(pred, true)
    mask = torch.zeros(2, 76, dtype=torch.bool)
    mask[:, 10:15] = True
    loss_with_mask = loss_fn._compute_source_loss(pred, true, epileptogenic_mask=mask)
    assert loss_no_mask.item() == pytest.approx(loss_with_mask.item(), abs=1e-6), (
        "Mask should be ignored in source loss"
    )


def test_shape_mismatch_propagates(loss_fn):
    pred = torch.randn(2, 76, 400)
    true_wrong = torch.randn(2, 76, 200)
    with pytest.raises(RuntimeError):
        loss_fn._compute_source_loss(pred, true_wrong)
