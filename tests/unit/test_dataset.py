import importlib.util
import numpy as np
import torch
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _import_hdf5_dataset():
    spec = importlib.util.spec_from_file_location(
        "train_network", PROJECT_ROOT / "scripts" / "03_train_network.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HDF5Dataset


@pytest.fixture(scope="module")
def mock_hdf5_path(tmp_path_factory):
    import h5py
    tmp_dir = tmp_path_factory.mktemp("hdf5_data")
    h5_path = tmp_dir / "test_dataset.h5"
    rng = np.random.RandomState(42)
    n_samples = 8
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("eeg", data=rng.randn(n_samples, 19, 400).astype(np.float32))
        f.create_dataset(
            "source_activity",
            data=rng.randn(n_samples, 76, 400).astype(np.float32),
        )
        epi_mask = np.zeros((n_samples, 76), dtype=bool)
        epi_mask[:, 10:15] = True
        f.create_dataset("epileptogenic_mask", data=epi_mask)
    return str(h5_path)


@pytest.fixture
def dataset(mock_hdf5_path):
    HDF5Dataset = _import_hdf5_dataset()
    return HDF5Dataset(
        hdf5_path=mock_hdf5_path,
        batch_size=4,
        shuffle=False,
    )


@pytest.fixture
def dataset_no_mask(mock_hdf5_path):
    import h5py
    import os
    HDF5Dataset = _import_hdf5_dataset()
    no_mask_path = mock_hdf5_path.replace(".h5", "_no_mask.h5")
    with h5py.File(mock_hdf5_path, "r") as src:
        with h5py.File(no_mask_path, "w") as dst:
            dst.create_dataset("eeg", data=src["eeg"][:])
            dst.create_dataset("source_activity", data=src["source_activity"][:])
    ds = HDF5Dataset(hdf5_path=no_mask_path, batch_size=4, shuffle=False)
    yield ds
    os.unlink(no_mask_path)


def test_returns_eeg_sources_mask_tuple(dataset):
    batch = next(iter(dataset))
    assert len(batch) == 3
    eeg, sources, mask = batch
    assert eeg.shape == (4, 19, 400)
    assert sources.shape == (4, 76, 400)
    assert mask.shape == (4, 76)
    assert mask.dtype == torch.bool


def test_indexing_consistent(dataset):
    batches = list(iter(dataset))
    assert len(batches) == 2


def test_len_matches_hdf5_size(dataset):
    assert len(dataset) == 2


def test_dataset_no_mask_returns_tuple(dataset_no_mask):
    batch = next(iter(dataset_no_mask))
    assert len(batch) == 2


def test_missing_hdf5_key_raises_error():
    HDF5Dataset = _import_hdf5_dataset()
    with pytest.raises((KeyError, OSError, FileNotFoundError, RuntimeError)):
        HDF5Dataset(hdf5_path="/nonexistent/file.h5", batch_size=4)
