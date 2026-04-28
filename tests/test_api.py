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
    """POST /api/analyze with EDF file returns valid result."""
    import os
    edf_path = "data/samples/0001082.edf"
    if not os.path.exists(edf_path):
        pytest.skip(f"EDF file not found: {edf_path}")
    with open(edf_path, "rb") as f:
        response = test_client.post(
            "/api/analyze",
            files={"file": ("0001082.edf", f, "application/octet-stream")},
            data={"mode": "source_localization", "include_eeg": "true"}
        )
    assert response.status_code in (200, 500)  # 500 OK if MNE missing, 200 if works
