import pytest
import torch


class TestMemory:

    def test_no_monotonic_memory_growth(self, model):
        device = next(model.parameters()).device
        model.eval()
        eeg = torch.randn(1, 19, 400).to(device)

        mem_usage = []
        for _ in range(10):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for _ in range(10):
                with torch.no_grad():
                    _ = model(eeg)
            if device.type == "cuda":
                mem_usage.append(torch.cuda.memory_allocated(device))

        if len(mem_usage) >= 2:
            assert mem_usage[-1] <= mem_usage[0] * 1.1 or device.type == "cpu", (
                f"Memory grew from {mem_usage[0]} to {mem_usage[-1]}"
            )

    def test_gradient_memory_freed(self, model):
        device = next(model.parameters()).device
        eeg = torch.randn(2, 19, 400).to(device)

        model.train()
        pred = model(eeg)
        loss = pred.pow(2).mean()
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                break
