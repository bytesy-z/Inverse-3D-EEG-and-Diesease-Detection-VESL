import torch
import pytest
from src.phase2_network.loss_functions import PhysicsInformedLoss


@pytest.fixture
def loss_fn():
    leadfield = torch.randn(19, 76)
    laplacian = torch.randn(76, 76)
    return PhysicsInformedLoss(leadfield, laplacian)


def test_perfect_prediction_loss_approx_zero(loss_fn):
    n_regions = 3
    pred = torch.randn(2, 76, 400) * 0.01
    pred[:, :n_regions, :] = torch.randn(2, n_regions, 400) * 1.0
    mask = torch.zeros(2, 76, dtype=torch.bool)
    mask[:, :n_regions] = True
    loss = loss_fn._compute_epi_loss(pred, mask)
    assert loss.item() == pytest.approx(0.0, abs=0.15), (
        f"Expected loss≈0 for matching epi regions, got {loss.item()}"
    )


def test_class_balanced_healthy_and_epi(loss_fn):
    pred = torch.randn(2, 76, 400)
    pred[:, 10:15, :] += 5.0
    mask = torch.zeros(2, 76, dtype=torch.bool)
    mask[:, 10:15] = True
    loss = loss_fn._compute_epi_loss(pred, mask)
    assert loss.item() > 0, "Expected positive loss with class imbalance"
    assert torch.isfinite(torch.tensor(loss.item())), "Loss should be finite"


def test_all_healthy_no_nan(loss_fn):
    pred = torch.randn(2, 76, 400)
    mask = torch.zeros(2, 76, dtype=torch.bool)
    loss = loss_fn._compute_epi_loss(pred, mask)
    assert torch.isfinite(torch.tensor(loss.item())), "Loss should be finite for all-healthy"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_epi_loss_uses_ac_variance(loss_fn):
    pred_no_dc = torch.randn(2, 76, 400)
    rng = torch.Generator().manual_seed(0)
    dc_offset = torch.randn(2, 76, 1, generator=rng) * 3.0
    pred_with_dc = pred_no_dc + dc_offset
    mask = torch.zeros(2, 76, dtype=torch.bool)
    mask[:, 10:13] = True
    loss_no_dc = loss_fn._compute_epi_loss(pred_no_dc, mask)
    loss_with_dc = loss_fn._compute_epi_loss(pred_with_dc, mask)
    assert loss_no_dc.item() == pytest.approx(loss_with_dc.item(), abs=1e-5), (
        "Epi loss should be based on AC variance, unaffected by DC"
    )


def test_none_mask_returns_zero(loss_fn):
    pred = torch.randn(2, 76, 400)
    loss = loss_fn._compute_epi_loss(pred, None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
