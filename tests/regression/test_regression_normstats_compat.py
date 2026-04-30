"""Regression: norm stats loading must handle both old and new formats."""
import json
import tempfile
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.unit]


def test_normstats_old_format():
    """Old format (all keys flat) must not crash."""
    old_stats = {
        "eeg_mean": 0.0, "eeg_std": 1.0,
        "src_mean": 0.0, "src_std": 1.0,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(old_stats, f)
        f.flush()
        with open(f.name) as f2:
            loaded = json.load(f2)
        assert loaded["eeg_mean"] == 0.0


def test_normstats_new_format():
    """New format with AC keys must also work."""
    new_stats = {
        "eeg_mean": 0.1, "eeg_std": 1.1,
        "eeg_mean_ac": 0.01, "eeg_std_ac": 1.01,
        "src_mean": 0.2, "src_std": 1.2,
        "src_mean_ac": 0.02, "src_std_ac": 1.02,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(new_stats, f)
        f.flush()
        with open(f.name) as f2:
            loaded = json.load(f2)
        ac_mean = loaded.get("eeg_mean_ac", loaded["eeg_mean"])
        assert ac_mean == 0.01
