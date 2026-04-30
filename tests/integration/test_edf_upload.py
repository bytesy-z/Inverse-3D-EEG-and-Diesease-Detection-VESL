"""Integration: EDF upload pipeline."""
import pytest

pytestmark = [pytest.mark.integration]

def test_edf_upload_rejected_for_nonexistent_file(test_client):
    with pytest.raises(Exception):
        with open("nonexistent.edf", "rb") as f:
            test_client.post("/api/analyze", files={"file": ("test.edf", f, "application/octet-stream")})
