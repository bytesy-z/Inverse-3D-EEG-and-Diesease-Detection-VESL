"""Edge case tests for loss functions."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit]


class TestLossEdgeCases:
    @pytest.fixture
    def loss_fn(self):
        from src.phase2_network.loss_functions import PhysicsInformedLoss
        return PhysicsInformedLoss(
            leadfield=torch.randn(19, 76),
            connectivity_laplacian=torch.randn(76, 76),
        )

    def test_nan_input_source_loss(self, loss_fn):
        source_pred = torch.full((2, 76, 400), float("nan"))
        source_true = torch.randn(2, 76, 400)
        eeg_input = torch.randn(2, 19, 400)
        result = loss_fn(source_pred, source_true, eeg_input)
        assert not torch.isfinite(result["loss_total"])

    def test_inf_input_source_loss(self, loss_fn):
        source_pred = torch.full((2, 76, 400), float("inf"))
        source_true = torch.randn(2, 76, 400)
        eeg_input = torch.randn(2, 19, 400)
        result = loss_fn(source_pred, source_true, eeg_input)
        assert not torch.isfinite(result["loss_total"])

    def test_all_zero_sources(self, loss_fn):
        source_pred = torch.zeros(2, 76, 400)
        source_true = torch.zeros(2, 76, 400)
        eeg_input = torch.randn(2, 19, 400)
        result = loss_fn(source_pred, source_true, eeg_input)
        assert torch.isfinite(result["loss_total"])
        assert result["loss_source"].item() == 0.0

    def test_extreme_dc_offset(self, loss_fn):
        source_true = torch.randn(2, 76, 400)
        source_pred = source_true + 1000.0
        eeg_input = torch.randn(2, 19, 400)
        result = loss_fn(source_pred, source_true, eeg_input)
        assert torch.isfinite(result["loss_total"])
        assert result["loss_total"].item() > 0

    def test_empty_batch_source_loss(self, loss_fn):
        source_pred = torch.empty(0, 76, 400)
        source_true = torch.empty(0, 76, 400)
        eeg_input = torch.empty(0, 19, 400)
        result = loss_fn(source_pred, source_true, eeg_input)
        assert not torch.isfinite(result["loss_total"])

    def test_epi_loss_edge_case(self, loss_fn):
        source_pred = torch.randn(2, 76, 400)
        epi_mask = torch.zeros(2, 76, dtype=torch.bool)
        eeg_input = torch.randn(2, 19, 400)
        source_true = torch.randn(2, 76, 400)
        result = loss_fn(source_pred, source_true, eeg_input, epileptogenic_mask=epi_mask)
        assert torch.isfinite(result["loss_total"])
