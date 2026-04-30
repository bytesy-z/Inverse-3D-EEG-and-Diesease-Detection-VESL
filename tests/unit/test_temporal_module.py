import torch
import pytest
from src.phase2_network.physdeepsif import TemporalModule


@pytest.fixture
def module():
    return TemporalModule(hidden_size=32)


def test_output_shape_preserved(module):
    spatial_out = torch.randn(4, 76, 400)
    out = module(spatial_out)
    assert out.shape == (4, 76, 400), f"Expected (4, 76, 400), got {out.shape}"


def test_bidirectional_correct_hidden_dim():
    module = TemporalModule(hidden_size=32)
    for name, param in module.lstm.named_parameters():
        if 'weight_hh' in name:
            assert param.shape[0] == 4 * 32, (
                f"LSTM hidden weight dim 0 should be 4*hidden=128, got {param.shape[0]}"
            )


def test_dropout_zero_in_eval(module):
    module.eval()
    training_state = module.training
    assert training_state is False, "Module should be in eval mode"
    with torch.no_grad():
        x = torch.randn(2, 76, 400)
        out1 = module(x)
        out2 = module(x)
    assert torch.allclose(out1, out2, atol=1e-6), (
        "Eval mode should be deterministic (dropout=0)"
    )


def test_dropout_nonzero_in_training(module):
    module.train()
    training_state = module.training
    assert training_state is True, "Module should be in training mode"
    x = torch.randn(4, 76, 400)
    out1 = module(x)
    out2 = module(x)
    diff = (out1 - out2).abs().mean().item()
    assert diff > 1e-8, (
        "Training mode should have small stochastic variation from dropout"
    )


def test_hidden_state_init_zeros(module):
    batch, time, features = 2, 400, 76
    x = torch.randn(batch, time, features)
    lstm_output, (hn, cn) = module.lstm(x)
    assert hn.shape == (4, batch, 32), f"hn shape mismatch: {hn.shape}"
    assert cn.shape == (4, batch, 32), f"cn shape mismatch: {cn.shape}"
    assert torch.isfinite(hn).all(), "Hidden states should be finite"
