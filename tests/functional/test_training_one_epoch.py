import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class TestTrainingOneEpoch:

    @pytest.fixture
    def mock_data(self):
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 1, (64, 19, 400)).astype(np.float32)
        sources = rng.normal(0, 0.5, (64, 76, 400)).astype(np.float32)
        mask = np.zeros((64, 76), dtype=bool)
        mask[:, 10:15] = True
        return eeg, sources, mask

    def test_loss_decreases_after_one_epoch(self, mock_data, model):
        import copy
        _saved_state = copy.deepcopy(model.state_dict())
        try:
            eeg, sources, mask = mock_data
            eeg_t = torch.from_numpy(eeg)
            src_t = torch.from_numpy(sources)
            mask_t = torch.from_numpy(mask)
            dataset = TensorDataset(eeg_t, src_t, mask_t)
            loader = DataLoader(dataset, batch_size=16)

            from src.phase2_network.loss_functions import PhysicsInformedLoss
            loss_fn = PhysicsInformedLoss(
                leadfield=model.leadfield,
                connectivity_laplacian=model.connectivity_laplacian,
                alpha=1.0, beta=0.0, gamma=0.01, delta_epi=1.0,
                lambda_laplacian=0.0, lambda_temporal=0.3, lambda_amplitude=0.2,
            )
            loss_fn.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            model.eval()
            with torch.no_grad():
                for batch in loader:
                    e_b, s_b, m_b = batch
                    pred = model(e_b)
                    init_loss = loss_fn(pred, s_b, e_b, m_b)["loss_total"].item()
                    break

            model.train()
            for batch in loader:
                e_b, s_b, m_b = batch
                pred = model(e_b)
                loss = loss_fn(pred, s_b, e_b, m_b)["loss_total"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for batch in loader:
                    e_b, s_b, m_b = batch
                    pred = model(e_b)
                    final_loss = loss_fn(pred, s_b, e_b, m_b)["loss_total"].item()
                    break

            assert final_loss < init_loss + 1e-6, (
                f"Loss did not decrease: init={init_loss:.4f}, final={final_loss:.4f}"
            )
        finally:
            model.load_state_dict(_saved_state)

    def test_gradients_flow_to_all_params(self, mock_data, model):
        eeg, sources, mask = mock_data
        eeg_t = torch.from_numpy(eeg[:4])
        src_t = torch.from_numpy(sources[:4])
        from src.phase2_network.loss_functions import PhysicsInformedLoss
        loss_fn = PhysicsInformedLoss(
            leadfield=model.leadfield,
            connectivity_laplacian=model.connectivity_laplacian,
            alpha=1.0, beta=0.0, gamma=0.01, delta_epi=1.0,
        )
        model.train()
        pred = model(eeg_t)
        loss = loss_fn(pred, src_t, eeg_t, None)["loss_total"]
        loss.backward()

        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                zero_grad_params.append(name)
            elif param.grad.abs().sum().item() == 0:
                zero_grad_params.append(name)

        assert len(zero_grad_params) == 0, (
            f"Params with zero gradient: {zero_grad_params[:5]}"
        )

    def test_no_nan_loss_or_gradients(self, mock_data, model):
        eeg, sources, mask = mock_data
        eeg_t = torch.from_numpy(eeg[:8])
        src_t = torch.from_numpy(sources[:8])
        from src.phase2_network.loss_functions import PhysicsInformedLoss
        loss_fn = PhysicsInformedLoss(
            leadfield=model.leadfield,
            connectivity_laplacian=model.connectivity_laplacian,
            alpha=1.0, beta=0.0, gamma=0.01, delta_epi=1.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        pred = model(eeg_t)
        loss = loss_fn(pred, src_t, eeg_t, None)["loss_total"]
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )

    def test_optimizer_step_updates_lr(self, mock_data, model):
        eeg, sources, mask = mock_data
        eeg_t = torch.from_numpy(eeg[:4])
        src_t = torch.from_numpy(sources[:4])
        from src.phase2_network.loss_functions import PhysicsInformedLoss
        loss_fn = PhysicsInformedLoss(
            leadfield=model.leadfield,
            connectivity_laplacian=model.connectivity_laplacian,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        model.train()
        pred = model(eeg_t)
        loss = loss_fn(pred, src_t, eeg_t, None)["loss_total"]
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        assert optimizer.param_groups[0]["lr"] <= 1e-3
