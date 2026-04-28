import sys
import types
import numpy as np
import torch


def _install_dummy_preprocessor(n_channels=19, n_time=400, n_epochs=4):
    """Insert a dummy module `src.phase3_inference.nmt_preprocessor` into sys.modules
    that provides `NMTPreprocessor` with a `preprocess` method returning
    synthetic EEG windows and metadata.
    """
    mod = types.ModuleType("src.phase3_inference.nmt_preprocessor")

    class NMTPreprocessor:
        def preprocess(self, path):
            eeg = np.random.randn(n_epochs, n_channels, n_time).astype(np.float32)
            return {
                "preprocessed_eeg": eeg,
                "channel_names": [f"Ch{i}" for i in range(n_channels)],
                "epoch_times": [0.0 + i * 2.0 for i in range(n_epochs)],
            }

    mod.NMTPreprocessor = NMTPreprocessor
    sys.modules["src.phase3_inference.nmt_preprocessor"] = mod


class DummyModel(torch.nn.Module):
    def __init__(self, n_regions=76, n_time=400):
        super().__init__()
        self.n_regions = n_regions
        self.n_time = n_time

    def forward(self, x):
        # x: (batch, channels, time)
        b = x.shape[0]
        # return deterministic values so tests are stable
        out = torch.zeros((b, self.n_regions, self.n_time), dtype=torch.float32)
        return out


def test_run_patient_inference_basic_shapes():
    """Smoke test for `run_patient_inference` ensuring outputs have expected shapes
    and keys when using mocked preprocessor and a dummy model.
    """
    _install_dummy_preprocessor(n_channels=19, n_time=400, n_epochs=5)

    # import under test
    from src.phase3_inference.inference_engine import run_patient_inference

    model = DummyModel(n_regions=76, n_time=400)

    out = run_patient_inference(edf_path="/not/used.edf", model=model, norm_stats=None, device="cpu", batch_size=2)

    # Basic presence checks
    expected_keys = {
        'source_estimates', 'mean_source_power', 'variance', 'peak_to_peak',
        'activation_consistency', 'preprocessed_eeg', 'n_epochs', 'epoch_times',
        'channel_names', 'edf_path'
    }
    assert expected_keys.issubset(set(out.keys()))

    n_epochs = out['n_epochs']
    assert n_epochs == 5

    src = out['source_estimates']
    assert isinstance(src, np.ndarray)
    assert src.shape == (n_epochs, 76, 400)

    # Aggregates length
    assert out['mean_source_power'].shape[0] == 76
    assert out['variance'].shape[0] == 76
    assert out['peak_to_peak'].shape[0] == 76
    assert out['activation_consistency'].shape[0] == 76

