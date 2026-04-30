"""Integration: backend lifecycle (startup, health, shutdown)."""
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration]

def test_health_endpoint_returns_system_info(test_client):
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
