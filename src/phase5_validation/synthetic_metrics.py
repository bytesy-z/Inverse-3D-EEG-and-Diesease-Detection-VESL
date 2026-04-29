import logging
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from src.phase2_network.metrics import (
    compute_auc_epileptogenicity,
    compute_temporal_correlation as _phase2_temporal_corr,
)

logger = logging.getLogger(__name__)

N_REGIONS = 76
N_TIMES = 400


def compute_dle(
    predicted_centers: NDArray[np.float32],
    true_centers: NDArray[np.float32],
) -> float:
    """Distance Localization Error (mm) — mean Euclidean distance between
    predicted and true epileptogenic region centers.

    Args:
        predicted_centers: (batch, 3) or (3,) predicted centroids in mm.
        true_centers: (batch, 3) or (3,) true centroids in mm.
            If 1-D, both are broadcast as single-sample.

    Returns:
        float: Mean DLE across batch in mm.
    """
    predicted_centers = np.atleast_2d(predicted_centers)
    true_centers = np.atleast_2d(true_centers)

    if predicted_centers.shape != true_centers.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_centers.shape} vs "
            f"true {true_centers.shape}. Both must have same shape."
        )
    if predicted_centers.shape[-1] != 3:
        raise ValueError(
            f"Last dimension must be 3 (x,y,z), got {predicted_centers.shape[-1]}."
        )

    dle_per_sample = np.linalg.norm(predicted_centers - true_centers, axis=-1)
    mean_dle = float(np.mean(dle_per_sample))

    logger.info(
        f"compute_dle — mean={mean_dle:.2f}mm, "
        f"per_sample={np.round(dle_per_sample, 2).tolist()}"
    )

    if mean_dle > 100.0:
        logger.warning(f"DLE={mean_dle:.1f}mm exceeds 100mm — unusually large.")

    return mean_dle


def compute_sd(
    predicted_sources: NDArray[np.float32],
    region_centers: NDArray[np.float32],
    top_k: int = 5,
) -> float:
    """Spatial Dispersion (mm) — std of distances of top-k regions from
    their power-weighted centroid.

    Differs from Phase 2's compute_spatial_dispersion which uses all
    power-weighted regions. This one selects only the top_k most active
    regions to measure focality of the epileptogenic zone estimate.

    Args:
        predicted_sources: (batch, 76, 400) predicted source activity.
        region_centers: (76, 3) MNI coordinates in mm.
        top_k: Number of top regions to consider (default 5).

    Returns:
        float: Mean SD across batch in mm.
    """
    if predicted_sources.shape[1] != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} regions, got {predicted_sources.shape[1]}."
        )
    if region_centers.shape != (N_REGIONS, 3):
        raise ValueError(
            f"region_centers must be ({N_REGIONS}, 3), got {region_centers.shape}."
        )
    if not 1 <= top_k <= N_REGIONS:
        raise ValueError(f"top_k must be between 1 and {N_REGIONS}, got {top_k}.")

    batch_size = predicted_sources.shape[0]
    sd_values = []

    for sample_idx in range(batch_size):
        source_ts = predicted_sources[sample_idx]
        power = np.mean(source_ts ** 2, axis=1)

        top_indices = np.argsort(power)[-top_k:][::-1]
        top_power = power[top_indices]
        total_top_power = np.sum(top_power)

        if total_top_power < 1e-10:
            logger.warning(
                f"Sample {sample_idx}: all top-{top_k} regions have zero power. "
                "Using geometric center fallback."
            )
            centroid = np.mean(region_centers, axis=0)
        else:
            centroid = np.sum(
                top_power[:, np.newaxis] * region_centers[top_indices], axis=0
            ) / total_top_power

        distances = np.linalg.norm(
            region_centers[top_indices] - centroid[np.newaxis, :], axis=1
        )
        weighted_variance = np.sum(top_power * distances ** 2) / (
            total_top_power + 1e-10
        )
        sd = float(np.sqrt(weighted_variance))
        sd_values.append(sd)

    mean_sd = float(np.mean(sd_values))
    logger.info(
        f"compute_sd (top_k={top_k}) — mean={mean_sd:.2f}mm, "
        f"per_sample={np.round(sd_values, 2).tolist()}"
    )
    return mean_sd


def compute_auc(
    true_mask: NDArray[np.bool_],
    predicted_scores: NDArray[np.float32],
) -> float:
    """Area Under ROC Curve — discriminating epileptogenic from healthy regions.

    Delegates to Phase 2's compute_auc_epileptogenicity.
    Accepts 3-D (batch, 76, 400) or 2-D (batch, 76) scores.
    If 2-D, a dummy time dimension is added.

    Args:
        true_mask: (batch, 76) binary mask — True for epileptogenic regions.
        predicted_scores: (batch, 76) or (batch, 76, 400) score values.

    Returns:
        float: AUC in [0, 1].
    """
    if predicted_scores.ndim == 2:
        scores_3d = predicted_scores[:, :, np.newaxis]
    elif predicted_scores.ndim == 3:
        scores_3d = predicted_scores
    else:
        raise ValueError(
            f"predicted_scores must be 2-D or 3-D, got {predicted_scores.ndim}-D."
        )

    if scores_3d.shape[:2] != true_mask.shape:
        raise ValueError(
            f"Shape mismatch: scores {scores_3d.shape[:2]} vs "
            f"mask {true_mask.shape}."
        )

    if true_mask.shape[1] != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} regions, got {true_mask.shape[1]}."
        )

    unique_labels = np.unique(true_mask)
    if len(unique_labels) < 2:
        logger.warning(
            f"All labels are the same ({unique_labels}). AUC undefined; "
            "returning 0.5."
        )
        return 0.5

    auc = compute_auc_epileptogenicity(scores_3d, true_mask)
    logger.info(f"compute_auc — AUC={auc:.4f}")
    return auc


def compute_temporal_correlation(
    predicted: NDArray[np.float32],
    true: NDArray[np.float32],
) -> float:
    """Mean Pearson correlation across regions between predicted and true
    time courses. Delegates to Phase 2 implementation.

    Args:
        predicted: (batch, 76, 400) predicted source activity.
        true: (batch, 76, 400) ground truth source activity.

    Returns:
        float: Mean correlation in [-1, 1].
    """
    if predicted.shape != true.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs true {true.shape}."
        )
    if predicted.shape[1] != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} regions, got {predicted.shape[1]}."
        )

    corr = _phase2_temporal_corr(predicted, true)
    logger.info(f"compute_temporal_correlation — mean_corr={corr:.4f}")
    return corr


def _power_weighted_centroid(
    sources: NDArray[np.float32],
    region_centers: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute power-weighted centroid of source activity.

    Args:
        sources: (batch, 76, 400) source time series.
        region_centers: (76, 3) MNI coordinates.

    Returns:
        NDArray: (batch, 3) centroids in mm.
    """
    power = np.mean(sources ** 2, axis=2)
    total_power = np.sum(power, axis=1, keepdims=True)
    degenerate = total_power.squeeze(1) < 1e-10
    if np.any(degenerate):
        logger.warning(
            f"{np.sum(degenerate)}/{len(degenerate)} samples have zero total "
            "power. Using geometric center fallback."
        )
    total_power = np.maximum(total_power, 1e-10)
    centroids = np.matmul(power, region_centers) / total_power
    return centroids.astype(np.float32)


def compute_all_metrics(
    predicted: NDArray[np.float32],
    true: NDArray[np.float32],
    true_mask: NDArray[np.bool_],
    region_centers: NDArray[np.float32],
) -> Dict[str, float]:
    """Run all 4 validation metrics and return dict.

    Args:
        predicted: (batch, 76, 400) predicted source activity.
        true: (batch, 76, 400) ground truth source activity.
        true_mask: (batch, 76) binary epileptogenic mask.
        region_centers: (76, 3) MNI coordinates in mm.

    Returns:
        dict with keys: dle, sd, auc, temporal_correlation.
    """
    logger.info(
        f"compute_all_metrics — ENTER: pred={predicted.shape} "
        f"true={true.shape} mask={true_mask.shape} "
        f"centers={region_centers.shape}"
    )

    pred_centroids = _power_weighted_centroid(predicted, region_centers)
    true_centroids = _power_weighted_centroid(true, region_centers)

    dle = compute_dle(pred_centroids, true_centroids)
    sd = compute_sd(predicted, region_centers)
    auc = compute_auc(true_mask, predicted)
    temporal_corr = compute_temporal_correlation(predicted, true)

    results = {
        "dle": dle,
        "sd": sd,
        "auc": auc,
        "temporal_correlation": temporal_corr,
    }

    logger.info(
        f"compute_all_metrics — SUMMARY: DLE={dle:.2f}mm | "
        f"SD={sd:.2f}mm | AUC={auc:.4f} | Corr={temporal_corr:.4f}"
    )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Testing synthetic_metrics...")

    batch = 4
    pred = np.random.randn(batch, N_REGIONS, N_TIMES).astype(np.float32)
    true = np.random.randn(batch, N_REGIONS, N_TIMES).astype(np.float32)
    mask = np.zeros((batch, N_REGIONS), dtype=bool)
    mask[:, 10:15] = True
    centers = np.random.randn(N_REGIONS, 3).astype(np.float32) * 50

    compute_dle(
        np.random.randn(batch, 3).astype(np.float32),
        np.random.randn(batch, 3).astype(np.float32),
    )
    print("  compute_dle:      PASS")

    compute_sd(pred, centers)
    print("  compute_sd:       PASS")

    compute_auc(mask, pred)
    print("  compute_auc:      PASS")

    compute_temporal_correlation(pred, true)
    print("  compute_temporal_correlation: PASS")

    result = compute_all_metrics(pred, true, mask, centers)
    assert set(result.keys()) == {"dle", "sd", "auc", "temporal_correlation"}
    print("  compute_all_metrics: PASS (keys:", list(result.keys()), ")")
    print("  Values:", {k: round(v, 4) for k, v in result.items()})

    print("\nAll synthetic metrics tests passed!")
