"""Tests for PhysDeepSIFTrainer."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

pytestmark = [pytest.mark.unit]


class TestTrainerInit:
    def test_init_with_minimal_model(self, model):
        import copy
        _saved_state = copy.deepcopy(model.state_dict())
        try:
            from src.phase2_network.trainer import PhysDeepSIFTrainer
            from src.phase2_network.loss_functions import PhysicsInformedLoss
            loss_fn = PhysicsInformedLoss(
                leadfield=model.leadfield,
                connectivity_laplacian=model.connectivity_laplacian,
            )
            device = torch.device("cpu")
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = PhysDeepSIFTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    device=device,
                    output_dir=Path(tmpdir),
                    region_centers=np.zeros((76, 3), dtype=np.float32),
                    learning_rate=1e-3,
                )
            assert trainer.model is model
            assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
        finally:
            model.load_state_dict(_saved_state)

    def test_train_epoch_shape_assertions(self, model):
        import copy
        _saved_state = copy.deepcopy(model.state_dict())
        from src.phase2_network.trainer import PhysDeepSIFTrainer
        from src.phase2_network.loss_functions import PhysicsInformedLoss
        loss_fn = PhysicsInformedLoss(
            leadfield=model.leadfield,
            connectivity_laplacian=model.connectivity_laplacian,
        )
        device = torch.device("cpu")
        import tempfile
        from pathlib import Path
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = PhysDeepSIFTrainer(
                    model=model,
                    loss_fn=loss_fn,
                    device=device,
                    output_dir=Path(tmpdir),
                    region_centers=np.zeros((76, 3), dtype=np.float32),
                    learning_rate=1e-3,
                )
                eeg_batch = torch.randn(4, 19, 400)
                src_batch = torch.randn(4, 76, 400)
                epi_mask = torch.zeros(4, 76, dtype=torch.bool)
                epi_mask[:, :3] = True
                dataset = TensorDataset(eeg_batch, src_batch, epi_mask)
                loader = DataLoader(dataset, batch_size=4)
                trainer.current_epoch = 0
                loss = trainer._train_epoch(loader)
                assert np.isfinite(loss)
                assert isinstance(loss, float)
        finally:
            model.load_state_dict(_saved_state)
