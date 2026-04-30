"""Integration: full inference pipeline from EEG to biomarker results."""
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.integration]

def test_inference_on_synthetic_sample(model, synthetic_sample):
    if model is None:
        pytest.skip("No model loaded")
    eeg, _ = synthetic_sample
    model.eval()
    with torch.no_grad():
        output = model(eeg)
    assert output.shape == (1, 76, 400)
    assert torch.isfinite(output).all()
