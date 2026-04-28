import pytest
import numpy as np
from src.xai.eeg_occlusion import explain_biomarker


def _make_mock_pipeline(fixed_scores=None):
    """Return a callable that returns fixed scores (76,) regardless of input."""
    if fixed_scores is None:
        fixed_scores = np.random.default_rng(0).random(76).astype(np.float32)
        fixed_scores[5] = 0.95  # make region 5 the top-1

    def pipeline(eeg_window):
        return {"scores": fixed_scores.copy()}

    pipeline.fixed_scores = fixed_scores
    return pipeline


def _make_variance_pipeline():
    """Return a pipeline where score[region_i] = variance(channel_i).

    When explain_biomarker occludes a segment of channel i by replacing
    it with the per-channel mean, the variance of channel i drops
    measurably, producing a positive attribution.  This lets occlusion
    tests verify that occlusion actually affects the score.
    """
    def pipeline(eeg_window):
        ch_var = np.var(eeg_window, axis=1)        # (19,)
        scores = np.zeros(76, dtype=np.float32)
        scores[:19] = ch_var
        return {"scores": scores}
    return pipeline


def test_explain_biomarker_returns_expected_keys():
    """explain_biomarker returns all required keys."""
    pipeline = _make_mock_pipeline()
    eeg = np.random.default_rng(1).normal(size=(19, 400)).astype(np.float32)

    result = explain_biomarker(eeg, target_region_idx=5, run_pipeline_fn=pipeline)

    assert isinstance(result, dict)
    assert "channel_importance" in result
    assert "time_importance" in result
    assert "attribution_map" in result
    assert "top_segments" in result
    assert "target_region_idx" in result
    assert "baseline_score" in result


def test_explain_biomarker_output_shapes():
    """Output arrays have correct shapes for 19-channel, 400-sample EEG."""
    pipeline = _make_mock_pipeline()
    eeg = np.random.default_rng(2).normal(size=(19, 400)).astype(np.float32)

    result = explain_biomarker(
        eeg, target_region_idx=5, run_pipeline_fn=pipeline,
        occlusion_width=40, stride=20,
    )

    n_segments = (400 - 40) // 20 + 1  # = 19
    assert len(result["channel_importance"]) == 19
    assert len(result["time_importance"]) == n_segments
    assert len(result["attribution_map"]) == 19
    assert len(result["attribution_map"][0]) == n_segments
    assert len(result["top_segments"]) == 5


def test_explain_biomarker_occlusion_affects_score():
    """Occluding channel N reduces channel N's variance, producing positive attribution."""
    pipeline = _make_variance_pipeline()
    rng = np.random.default_rng(42)
    eeg = rng.normal(size=(19, 400)).astype(np.float32)

    result = explain_biomarker(eeg, target_region_idx=5, run_pipeline_fn=pipeline)
    attr = np.array(result["attribution_map"])

    assert attr.shape == (19, 19)
    # Channel 5 occlusion reduces channel 5 variance → positive attribution
    assert attr[5].mean() > 0, \
        f"Channel {5} mean attribution should be >0, got {attr[5].mean():.6f}"
    # At least one positive attribution somewhere
    assert attr.max() > 0


def test_explain_biomarker_returns_top_segments():
    """Top-5 segments have sensible time indices within window bounds."""
    pipeline = _make_mock_pipeline()
    eeg = np.random.default_rng(3).normal(size=(19, 400)).astype(np.float32)

    result = explain_biomarker(eeg, target_region_idx=5, run_pipeline_fn=pipeline)

    for seg in result["top_segments"]:
        assert 0 <= seg["channel_idx"] < 19
        assert 0 <= seg["start_sample"] < 400
        assert seg["end_sample"] <= 400
        assert seg["start_sample"] < seg["end_sample"]


def test_explain_biomarker_default_params():
    """Uses default 40-sample width and 20-sample stride without error."""
    pipeline = _make_mock_pipeline()
    eeg = np.random.default_rng(4).normal(size=(19, 400)).astype(np.float32)

    result = explain_biomarker(eeg, target_region_idx=5, run_pipeline_fn=pipeline)
    assert result["baseline_score"] >= 0
