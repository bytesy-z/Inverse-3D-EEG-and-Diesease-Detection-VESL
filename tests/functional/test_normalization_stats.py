import pytest
import numpy as np
import json
import tempfile


class TestNormalizationStats:

    def test_stats_roundtrip(self, mock_normalization_stats):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_normalization_stats, f)
            f.flush()
            with open(f.name) as f2:
                loaded = json.load(f2)

        for key in mock_normalization_stats:
            assert loaded[key] == mock_normalization_stats[key], (
                f"Roundtrip mismatch for {key}"
            )

    def test_eeg_stats_keys(self, mock_normalization_stats):
        assert "eeg_mean" in mock_normalization_stats
        assert "eeg_std" in mock_normalization_stats

    def test_source_stats_keys(self, mock_normalization_stats):
        assert "src_mean" in mock_normalization_stats
        assert "src_std" in mock_normalization_stats

    def test_stats_applied_normalizes(self, mock_normalization_stats, mock_eeg_batch):
        eps = 1e-7
        mean = mock_normalization_stats["eeg_mean"]
        std = mock_normalization_stats["eeg_std"]
        normalized = (mock_eeg_batch - mean) / (std + eps)
        assert abs(normalized.mean()) < 0.5
