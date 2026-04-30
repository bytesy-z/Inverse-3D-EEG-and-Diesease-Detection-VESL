import pytest
import torch
import time


class TestThroughput:

    def test_batch1_latency(self, model):
        device = next(model.parameters()).device
        model.eval()
        eeg = torch.randn(1, 19, 400).to(device)

        for _ in range(5):
            with torch.no_grad():
                _ = model(eeg)

        start = time.time()
        n_iterations = 20
        for _ in range(n_iterations):
            with torch.no_grad():
                _ = model(eeg)
        elapsed = (time.time() - start) / n_iterations * 1000

        assert elapsed < 200, f"Batch=1 latency: {elapsed:.1f}ms (expected <200ms)"

    @pytest.mark.slow
    def test_throughput_batch64(self, model):
        device = next(model.parameters()).device
        model.eval()
        eeg = torch.randn(64, 19, 400).to(device)

        for _ in range(3):
            with torch.no_grad():
                _ = model(eeg)

        start = time.time()
        n_batches = 30
        for _ in range(n_batches):
            with torch.no_grad():
                _ = model(eeg)
        elapsed = time.time() - start
        samples_per_sec = n_batches * 64 / elapsed

        assert samples_per_sec > 10, (
            f"Throughput: {samples_per_sec:.0f} samples/sec (expected >10)"
        )
