import pytest
import json


def test_health_endpoint(test_client):
    """GET /api/health returns 200 with model_loaded=True."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_analyze_synthetic(test_client):
    """POST /api/analyze with synthetic sample index returns valid result."""
    response = test_client.post(
        "/api/analyze",
        data={"sample_idx": 10, "mode": "source_localization"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "plotHtml" in data


def test_analyze_edf(test_client):
    """POST /api/analyze with EDF file returns 200 when MNE is available."""
    import os
    try:
        import mne  # noqa: F401
    except ImportError:
        pytest.skip("MNE not installed — cannot process EDF files")
    edf_path = "data/samples/0001082.edf"
    if not os.path.exists(edf_path):
        pytest.skip(f"EDF file not found: {edf_path}")
    with open(edf_path, "rb") as f:
        response = test_client.post(
            "/api/analyze",
            files={"file": ("0001082.edf", f, "application/octet-stream")},
            data={"mode": "source_localization", "include_eeg": "true"}
        )
    assert response.status_code == 200, (
        f"EDF analysis failed with status {response.status_code}: {response.text[:500]}"
    )


def test_websocket_polls_job_status(test_client):
    """WebSocket endpoint returns active_job status then cleans up on disconnect."""
    import backend.server as server_mod
    server_mod.active_jobs["ws_test_job"] = {
        "status": "completed", "progress": 100,
        "message": "done", "_ts": 0.0,
    }
    with test_client.websocket_connect("/ws/ws_test_job") as ws:
        data = ws.receive_json()
        assert data["status"] == "completed"
        assert data["progress"] == 100
    # After disconnect, job should be cleaned up
    assert "ws_test_job" not in server_mod.active_jobs


def test_websocket_returns_in_progress_then_completed(test_client):
    """WebSocket polls until job reaches terminal status."""
    import backend.server as server_mod
    server_mod.active_jobs["ws_progress_job"] = {
        "status": "processing", "progress": 50,
        "message": "halfway", "_ts": 0.0,
    }
    with test_client.websocket_connect("/ws/ws_progress_job") as ws:
        msg1 = ws.receive_json()
        assert msg1["status"] == "processing"
        # Update to completed while connected
        server_mod.active_jobs["ws_progress_job"].update(
            {"status": "completed", "progress": 100, "message": "done"}
        )
        msg2 = ws.receive_json()
        assert msg2["status"] == "completed"
    assert "ws_progress_job" not in server_mod.active_jobs
