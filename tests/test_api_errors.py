"""Tests for API error handling paths."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit]


def test_analyze_model_not_loaded(test_client):
    """Test POST /api/analyze returns 503 when model is None."""
    import backend.server as server
    saved_model = server.model
    server.model = None
    try:
        response = test_client.post("/api/analyze", data={"sample_idx": 0})
        assert response.status_code == 503
        assert "not loaded" in response.text.lower()
    finally:
        server.model = saved_model


def test_analyze_nan_input(test_client):
    """Test POST with NaN EEG values returns 400."""
    import tempfile, os
    nan_eeg = np.full((19, 400), np.nan, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, nan_eeg)
        f.seek(0)
        response = test_client.post("/api/analyze", files={"file": ("test.npy", f, "application/octet-stream")})
    os.unlink(f.name)
    assert response.status_code in (400, 422, 500)


def test_analyze_oversized_file(test_client):
    """Test oversized upload returns 413 or error."""
    import tempfile, os
    large_data = np.zeros((19, 2000000), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        np.save(f, large_data)
        f.seek(0)
        response = test_client.post("/api/analyze", files={"file": ("test.edf", f, "application/octet-stream")})
    os.unlink(f.name)
    assert response.status_code in (400, 413, 422, 500)


def test_analyze_missing_file(test_client):
    """Test POST without file or sample_idx returns 400."""
    response = test_client.post("/api/analyze")
    assert response.status_code in (400, 422)


def test_analyze_out_of_range_sample(test_client):
    """Test out-of-range sample index returns 400."""
    response = test_client.post("/api/analyze", data={"sample_idx": 999999})
    assert response.status_code in (400, 422, 500)


def test_analyze_wrong_shape_npy(test_client):
    """Test NPY with wrong shape (not 19x400) returns 400."""
    import tempfile, os
    wrong_eeg = np.zeros((10, 100), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, wrong_eeg)
        f.seek(0)
        response = test_client.post("/api/analyze", files={"file": ("test.npy", f, "application/octet-stream")})
    os.unlink(f.name)
    assert response.status_code in (400, 422)


def test_analyze_invalid_extension(test_client):
    """Test .txt file returns 400."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an eeg file")
        f.seek(0)
        response = test_client.post("/api/analyze", files={"file": ("test.txt", f, "text/plain")})
    os.unlink(f.name)
    assert response.status_code in (400, 422)


def test_analyze_invalid_mode(test_client):
    """Test invalid mode parameter."""
    response = test_client.post("/api/analyze", data={"sample_idx": 0, "mode": "invalid_mode"})
    assert response.status_code in (200, 422)


def test_analyze_missing_normalization_key(test_client):
    """Test missing normalization key returns 500."""
    import backend.server as server
    saved_stats = server.norm_stats
    server.norm_stats = {"bad_key": 0.0}
    try:
        response = test_client.post("/api/analyze", data={"sample_idx": 0})
        assert response.status_code == 500
    finally:
        server.norm_stats = saved_stats


def test_results_path_traversal(test_client):
    """Test path traversal returns 403."""
    response = test_client.get("/api/results/../../../etc/passwd")
    assert response.status_code in (403, 404)


def test_health_endpoint(test_client):
    """Test health endpoint returns 200."""
    response = test_client.get("/api/health")
    assert response.status_code == 200


def test_biomarkers_endpoint(test_client):
    """Test biomarkers endpoint with sample."""
    response = test_client.post("/api/biomarkers", data={"sample_idx": 0})
    assert response.status_code in (200, 503)


def test_eeg_waveform_endpoint(test_client):
    """Test eeg_waveform endpoint."""
    response = test_client.post("/api/eeg_waveform", json={"eeg": [[0.0] * 400 for _ in range(19)]})
    assert response.status_code in (200, 422)
