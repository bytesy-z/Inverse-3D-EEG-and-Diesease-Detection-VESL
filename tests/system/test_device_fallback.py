import pytest
import torch


class TestDeviceFallback:

    def test_cpu_forward_pass(self, model):
        device = torch.device("cpu")
        model.to(device)
        rng = torch.Generator(device="cpu").manual_seed(42)
        eeg = torch.randn(1, 19, 400, generator=rng)
        with torch.no_grad():
            out = model(eeg)
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward_pass(self, model):
        device = torch.device("cuda")
        model.to(device)
        torch.cuda.manual_seed(42)
        eeg = torch.randn(1, 19, 400, device=device)
        with torch.no_grad():
            out = model(eeg)
        assert torch.isfinite(out).all()

    def test_cpu_batch_no_oom(self, model):
        device = torch.device("cpu")
        model.to(device)
        eeg = torch.randn(64, 19, 400)
        with torch.no_grad():
            out = model(eeg)
        assert out.shape == (64, 76, 400)
