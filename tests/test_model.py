import torch


def test_model_loads(model):
    """Model loads from checkpoint without error."""
    assert model is not None
    params = model.get_parameter_count()
    assert params["total_trainable"] > 300_000  # ~419k


def test_forward_pass_shape(model, synthetic_sample):
    """Forward pass produces correct output shape."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert output.shape == (1, 76, 400), f"Expected (1,76,400), got {output.shape}"


def test_output_finite(model, synthetic_sample):
    """Model output contains no NaN or Inf."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


def test_output_not_constant(model, synthetic_sample):
    """Model output has non-zero variance (not collapsed)."""
    eeg, _ = synthetic_sample
    with torch.no_grad():
        output = model(eeg)
    assert output.std() > 1e-6, f"Output std too small: {output.std():.2e}"
