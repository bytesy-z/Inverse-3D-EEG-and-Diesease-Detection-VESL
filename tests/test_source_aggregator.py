import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from src.phase3_inference.source_aggregator import aggregate_sources


def test_aggregate_basic_metrics(tmp_path):
    # Construct a small deterministic source_estimates: (n_epochs=2, n_regions=3, n_time=4)
    se = np.zeros((2, 3, 4), dtype=float)
    # epoch 0
    se[0, 0, :] = 0
    se[0, 1, :] = 1
    se[0, 2, :] = np.array([0, 1, 2, 3])
    # epoch 1
    se[1, 0, :] = 0
    se[1, 1, :] = 2
    se[1, 2, :] = np.array([3, 2, 1, 0])

    labels = ["r0", "r1", "r2"]
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    out = aggregate_sources(se, region_labels_path=str(labels_path), top_fraction=0.34)

    assert out['n_epochs'] == 2
    assert out['n_regions'] == 3
    assert out['n_time'] == 4
    assert out['region_names'] == labels

    # mean_power: mean abs across epochs and time
    expected_mean_power = np.mean(np.abs(se), axis=(0, 2))
    np.testing.assert_allclose(out['mean_power'], expected_mean_power)

    # variance
    expected_var = np.var(se, axis=(0, 2))
    np.testing.assert_allclose(out['variance'], expected_var)

    # peak-to-peak
    expected_ptp = np.max(se, axis=(0, 2)) - np.min(se, axis=(0, 2))
    np.testing.assert_allclose(out['peak_to_peak'], expected_ptp)


def test_input_types_and_activation_consistency():
    # Use list input and verify activation_consistency calculation
    se = [
        [[0, 0, 0], [1, 2, 3], [5, 5, 5]],  # epoch 0
        [[0, 0, 0], [2, 2, 2], [0, 1, 0]],  # epoch 1
    ]
    out = aggregate_sources(se, region_labels_path='nonexistent.json', top_fraction=1/3)

    # n_epochs, n_regions, n_time
    assert out['n_epochs'] == 2
    assert out['n_regions'] == 3
    assert out['n_time'] == 3

    # top_fraction=1/3 -> top_k = ceil(1) = 1, so activation_consistency counts epochs where region is top1
    # For epoch0, region2 has highest mean; epoch1, region1 has highest mean (region1 mean=2, region2 mean=0.333)
    # So activation_consistency should be: region0=0.0, region1=0.5, region2=0.5
    np.testing.assert_allclose(out['activation_consistency'], np.array([0.0, 0.5, 0.5]))
