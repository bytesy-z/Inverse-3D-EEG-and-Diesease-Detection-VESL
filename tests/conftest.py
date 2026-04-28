import pytest
import torch
import numpy as np
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def model():
    """Load PhysDeepSIF model from checkpoint."""
    from src.phase2_network.physdeepsif import build_physdeepsif
    model = build_physdeepsif(
        str(PROJECT_ROOT / "data/leadfield_19x76.npy"),
        str(PROJECT_ROOT / "data/connectivity_76.npy"),
    )
    ckpt = torch.load(
        str(PROJECT_ROOT / "outputs/models/checkpoint_best.pt"),
        map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@pytest.fixture(scope="session")
def normalization_stats():
    with open(PROJECT_ROOT / "outputs/models/normalization_stats.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def synthetic_sample():
    import h5py
    with h5py.File(str(PROJECT_ROOT / "data/synthetic3/test_dataset.h5"), "r") as f:
        eeg = torch.from_numpy(f["eeg"][0:1].astype(np.float32))
        mask = torch.from_numpy(f["epileptogenic_mask"][0:1].astype(bool))
    return eeg, mask


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient for API testing."""
    from fastapi.testclient import TestClient
    import backend.server as server_mod
    return TestClient(server_mod.app)
