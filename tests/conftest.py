import pytest
import torch
import numpy as np
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Mock Data Fixtures ───────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_leadfield():
    rng = np.random.RandomState(42)
    base = rng.randn(19, 10).astype(np.float32)
    leadfield = base @ rng.randn(10, 76).astype(np.float32)
    leadfield += 0.1 * rng.randn(19, 76).astype(np.float32)
    u, s, vt = np.linalg.svd(leadfield, full_matrices=False)
    s = np.exp(-np.arange(len(s)) * 0.5) * s[0]
    leadfield = (u * s) @ vt
    return leadfield

@pytest.fixture(scope="session")
def mock_connectivity():
    rng = np.random.RandomState(42)
    conn = rng.rand(76, 76).astype(np.float32)
    conn = (conn + conn.T) / 2
    conn[conn < 0.85] = 0
    np.fill_diagonal(conn, 0)
    return conn

@pytest.fixture(scope="session")
def mock_laplacian(mock_connectivity):
    rng = np.random.default_rng(7)
    adj = rng.random((76, 76)) < 0.1
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj
    return lap.astype(np.float32)

@pytest.fixture(scope="session")
def mock_region_centers():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 40, (76, 3)).astype(np.float32)
    return centers

@pytest.fixture(scope="session")
def mock_eeg_batch():
    rng = np.random.RandomState(1)
    n_batch, n_ch, n_time = 4, 19, 400
    freqs = np.fft.rfftfreq(n_time)
    spectrum = np.where(freqs > 0, 1.0 / (freqs + 0.1), 0.0)
    eeg = np.zeros((n_batch, n_ch, n_time), dtype=np.float32)
    for b in range(n_batch):
        for c in range(n_ch):
            coeffs = rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
            coeffs *= np.sqrt(spectrum)
            eeg[b, c] = np.fft.irfft(coeffs)[:n_time]
    eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-7)
    return eeg

@pytest.fixture(scope="session")
def mock_source_batch():
    rng = np.random.default_rng(2)
    sources = rng.normal(0, 0.5, (4, 76, 400)).astype(np.float32)
    sources[:, 10:15, :] *= 3.0
    return sources

@pytest.fixture(scope="session")
def mock_epi_mask():
    mask = np.zeros((4, 76), dtype=bool)
    mask[0, 10:13] = True
    mask[1, 25:28] = True
    mask[2, 40:42] = True
    mask[3, 55:58] = True
    return mask

@pytest.fixture(scope="session")
def mock_normalization_stats():
    return {
        "eeg_mean": 0.0, "eeg_std": 1.0,
        "eeg_mean_ac": 0.01, "eeg_std_ac": 1.02,
        "src_mean": 0.0, "src_std": 1.0,
        "src_mean_ac": 0.02, "src_std_ac": 1.05,
    }

# ── Real Data Fixtures (for functional/system tests) ─────────────

@pytest.fixture(scope="session")
def load_leadfield():
    path = PROJECT_ROOT / "data" / "leadfield_19x76.npy"
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    pytest.skip("leadfield_19x76.npy not found")

@pytest.fixture(scope="session")
def load_region_centers():
    path = PROJECT_ROOT / "data" / "region_centers_76.npy"
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    pytest.skip("region_centers_76.npy not found")

@pytest.fixture(scope="session")
def load_connectivity():
    path = PROJECT_ROOT / "data" / "connectivity_76.npy"
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    pytest.skip("connectivity_76.npy not found")

@pytest.fixture(scope="session")
def load_normalization_stats():
    path = PROJECT_ROOT / "outputs" / "models" / "normalization_stats.json"
    if path.exists():
        with open(str(path)) as f:
            stats = json.load(f)
        return stats
    pytest.skip("normalization_stats.json not found")

@pytest.fixture(scope="session")
def model(load_leadfield, load_connectivity, load_region_centers):
    from src.phase2_network.physdeepsif import build_physdeepsif
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_physdeepsif(
        leadfield_path=str(PROJECT_ROOT / "data" / "leadfield_19x76.npy"),
        connectivity_path=str(PROJECT_ROOT / "data" / "connectivity_76.npy"),
        lstm_hidden_size=76,
    )
    model.to(device)
    checkpoint_path = PROJECT_ROOT / "outputs" / "models" / "checkpoint_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    else:
        pytest.skip("checkpoint_best.pt not found")
    model.to(torch.device("cpu"))
    return model

@pytest.fixture(scope="session")
def synthetic_sample():
    rng = np.random.default_rng(42)
    eeg = rng.normal(0, 1, (1, 19, 400)).astype(np.float32)
    mask = np.zeros((1, 76), dtype=bool)
    mask[0, 10:13] = True
    return torch.from_numpy(eeg), mask

# ── Backward-compat aliases for existing tests ────────────────────

@pytest.fixture(scope="session")
def normalization_stats(load_normalization_stats):
    return load_normalization_stats

@pytest.fixture(scope="session")
def test_client(load_normalization_stats):
    """TestClient with model and stats pre-loaded."""
    import backend.server as server_mod
    from fastapi.testclient import TestClient
    import json

    # Load model into server module's global scope
    if server_mod.model is None:
        from src.phase2_network.physdeepsif import build_physdeepsif
        import torch
        device = torch.device("cpu")
        m = build_physdeepsif(
            leadfield_path=str(server_mod.PROJECT_ROOT / "data" / "leadfield_19x76.npy"),
            connectivity_path=str(server_mod.PROJECT_ROOT / "data" / "connectivity_76.npy"),
            lstm_hidden_size=76,
        )
        ckpt_path = server_mod.PROJECT_ROOT / "outputs" / "models" / "checkpoint_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device)
            m.load_state_dict(ckpt["model_state"])
        m.eval()
        server_mod.model = m.to(device)

    # Load normalization stats
    if server_mod.norm_stats is None:
        server_mod.norm_stats = load_normalization_stats

    with TestClient(server_mod.app) as client:
        yield client
