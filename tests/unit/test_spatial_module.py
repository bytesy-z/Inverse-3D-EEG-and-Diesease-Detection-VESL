import torch
import pytest
from src.phase2_network.physdeepsif import SpatialModule


@pytest.fixture
def module():
    return SpatialModule()


def test_output_shape_batch_time(module):
    eeg = torch.randn(4, 19, 400)
    out = module(eeg)
    assert out.shape == (4, 76, 400), f"Expected (4, 76, 400), got {out.shape}"


def test_output_shape_single_time(module):
    eeg = torch.randn(4, 19)
    out = module(eeg)
    assert out.shape == (4, 76), f"Expected (4, 76), got {out.shape}"


def test_output_shape_single_sample(module):
    eeg = torch.randn(1, 19, 400)
    out = module(eeg)
    assert out.shape == (1, 76, 400), f"Expected (1, 76, 400), got {out.shape}"


def test_skip_connections_active(module):
    eeg = torch.randn(2, 19, 400)
    out = module(eeg)
    nonzero = out.abs().sum().item()
    assert nonzero > 0, "Output should be non-zero with skip connections"


def test_batchnorm_eval_mode(module):
    module.eval()
    eeg1 = torch.randn(2, 19, 400)
    eeg2 = torch.randn(2, 19, 400)
    with torch.no_grad():
        out1 = module(eeg1)
        out2 = module(eeg2)
    assert out1.shape == out2.shape, "Shapes should match in eval mode"
    assert torch.isfinite(out1).all(), "All outputs should be finite in eval mode"


def test_weight_initialisation_correct_type(module):
    for name, param in module.named_parameters():
        assert param.dtype == torch.float32, f"{name} has dtype {param.dtype}, expected float32"
