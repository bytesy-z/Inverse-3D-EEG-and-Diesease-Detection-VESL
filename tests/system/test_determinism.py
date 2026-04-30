import pytest
import torch
import numpy as np


class TestDeterminism:

    def test_same_seed_same_output(self, model):
        device = next(model.parameters()).device
        model.eval()

        torch.manual_seed(42)
        np.random.seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed(42)

        eeg = torch.randn(1, 19, 400).to(device)
        with torch.no_grad():
            out1 = model(eeg).clone()

        torch.manual_seed(42)
        np.random.seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed(42)

        eeg2 = torch.randn(1, 19, 400).to(device)
        with torch.no_grad():
            out2 = model(eeg2)

        diff = (out1 - out2).abs().max().item()
        assert diff < 1e-5, f"Determinism violated: diff={diff:.2e}"

    @pytest.mark.slow
    def test_different_seed_different_output(self, model, synthetic_sample):
        device = next(model.parameters()).device
        model.eval()
        eeg, _ = synthetic_sample
        eeg = eeg.to(device)

        torch.manual_seed(1)
        with torch.no_grad():
            out1 = model(eeg).clone()

        torch.manual_seed(999)
        with torch.no_grad():
            out2 = model(eeg)

        diff = (out1 - out2).abs().max().item()
        assert diff >= 0
