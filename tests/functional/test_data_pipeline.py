import pytest
import numpy as np
from pathlib import Path


class TestDataPipeline:

    @pytest.fixture
    def test_dataset(self):
        h5_path = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic3" / "test_dataset.h5"
        if not h5_path.exists():
            pytest.skip("test_dataset.h5 not found")

        import h5py
        with h5py.File(str(h5_path), "r") as f:
            n_samples = min(10, f["eeg"].shape[0])
            samples = []
            for i in range(n_samples):
                eeg = f["eeg"][i]
                sources = f["source_activity"][i]
                mask = f["epileptogenic_mask"][i]
                samples.append((eeg, sources, mask))
        return samples

    def test_all_samples_finite(self, test_dataset):
        for i, (eeg, sources, mask) in enumerate(test_dataset):
            assert np.isfinite(eeg).all(), f"Sample {i} EEG has non-finite values"
            assert np.isfinite(sources).all(), f"Sample {i} sources have non-finite values"

    def test_eeg_shape(self, test_dataset):
        for i, (eeg, sources, mask) in enumerate(test_dataset):
            assert eeg.shape == (19, 400), f"Sample {i}: EEG shape {eeg.shape}"

    def test_sources_variance(self, test_dataset):
        for i, (eeg, sources, mask) in enumerate(test_dataset):
            epi_indices = np.where(mask)[0]
            if len(epi_indices) > 0:
                epi_var = sources[epi_indices].var()
                assert epi_var > 1e-6, (
                    f"Sample {i}: epi region variance too low: {epi_var:.2e}"
                )

    def test_epi_mask_has_positives(self, test_dataset):
        has_epi = sum(mask.any() for (_, _, mask) in test_dataset)
        assert has_epi > 0, "No samples have epileptogenic regions"

    def test_epi_mask_shape(self, test_dataset):
        for i, (eeg, sources, mask) in enumerate(test_dataset):
            assert mask.shape == (76,), f"Sample {i}: mask shape {mask.shape}"
