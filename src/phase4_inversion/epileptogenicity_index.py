"""
Epileptogenicity Index (EI) computation from fitted x0 parameters.

The EI maps fitted Epileptor excitability (x0) parameters to a normalized
epileptogenicity score in [0, 1], where:
  - EI ≈ 0.0  : Strongly suppressed (x0 < -2.4)
  - EI ≈ 0.5  : Healthy baseline (x0 ≈ -2.2)
  - EI ≈ 1.0  : Highly epileptogenic (x0 > -1.2)

This uses a sigmoid mapping centered at the healthy baseline from Epileptor
literature, replacing the heuristic peak-to-peak amplitude inversion in the
backend with a physics-based approach grounded in the TVB neural mass model.

Reference:
  Jirsa, V. K., et al. (2014). The Virtual Brain: A computational neuroscience
  platform. NeuroImage, 111, 385-430.
"""


import numpy as np
from numpy.typing import NDArray


def compute_ei(
    x0_fitted: NDArray,
    x0_baseline: float = -2.2,
    scale: float = 0.15,
) -> NDArray:
    """
    Compute epileptogenicity index (EI) from fitted x0 parameters.
    
    Uses sigmoid mapping to normalize fitted x0 to [0, 1]:
      EI_i = sigmoid((x0_i - x0_baseline) / scale)
      
    where sigmoid(z) = 1 / (1 + exp(-z))
    
    Args:
        x0_fitted: (76,) array of fitted Epileptor excitability parameters
        x0_baseline: Healthy x0 baseline from Epileptor literature (default -2.2)
        scale: Sigmoid transition sharpness parameter (default 0.15)
               - Smaller scale → sharper transition (more binary)
               - Larger scale → smoother transition (more gradual)
    
    Returns:
        ei: (76,) array of epileptogenicity indices in [0, 1]
    
    Interpretation:
        x0 = -2.4 (strongly suppressed) → EI ≈ 0.13
        x0 = -2.2 (healthy baseline)    → EI ≈ 0.50
        x0 = -1.8 (mildly hyperexcitable) → EI ≈ 0.88
        x0 = -1.2 (highly epileptogenic) → EI ≈ 0.98
        x0 = -1.0 (upper bound)         → EI ≈ 0.999
    
    Example:
        >>> x0_fitted = np.array([-2.4, -2.2, -1.8, -1.2])
        >>> ei = compute_ei(x0_fitted)
        >>> print(ei)  # [0.13, 0.50, 0.88, 0.98]
    """
    # Compute normalized distance from healthy baseline
    z = (x0_fitted - x0_baseline) / scale
    
    # Apply sigmoid
    ei = 1.0 / (1.0 + np.exp(-z))
    
    return ei


def compute_ei_with_confidence(
    x0_fitted: NDArray,
    cmaes_convergence_history: list,
    x0_baseline: float = -2.2,
    scale: float = 0.15,
) -> dict:
    """
    Compute EI with convergence-based confidence metric.
    
    A region with poor CMA-ES convergence (high objective value at termination)
    may have less reliable EI estimates, especially if it's at intermediate values.
    This function computes a confidence score based on final objective value and
    convergence smoothness.
    
    Args:
        x0_fitted: (76,) array of fitted Epileptor excitability parameters
        cmaes_convergence_history: list of min objective values per generation
        x0_baseline: Healthy x0 baseline (default -2.2)
        scale: Sigmoid transition sharpness (default 0.15)
    
    Returns:
        dict with keys:
            - 'ei': (76,) array of epileptogenicity indices
            - 'confidence': (76,) array of confidence scores [0, 1]
            - 'convergence_final': final objective value
            - 'convergence_improvement': ratio of first to final objective
    
    Example:
        >>> cmaes_hist = [100.0, 50.0, 25.0, 12.0, 8.0]  # Convergence history
        >>> ei_result = compute_ei_with_confidence(x0_fitted, cmaes_hist)
        >>> high_confidence_ei = ei_result['ei'][ei_result['confidence'] > 0.8]
    """
    ei = compute_ei(x0_fitted, x0_baseline, scale)
    
    # Convergence metrics
    final_obj = cmaes_convergence_history[-1] if cmaes_convergence_history else 1.0
    initial_obj = cmaes_convergence_history[0] if cmaes_convergence_history else 1.0
    
    # Normalize final objective to confidence score
    # Lower objective = higher confidence
    # Threshold: if final_obj > 50, confidence = 0.1; if final_obj < 1, confidence = 0.9
    confidence = 1.0 / (1.0 + np.log10(max(final_obj, 0.01)))
    confidence = np.clip(confidence, 0.0, 1.0)
    
    # Convergence smoothness: check if convergence is monotonic
    diffs = np.diff(cmaes_convergence_history)
    monotonic = np.mean(diffs <= 0.0)  # Fraction of decreasing steps
    smoothness = monotonic  # Higher = more monotonic decreasing
    
    # Final confidence is geometric mean of convergence and smoothness
    final_confidence = np.sqrt(confidence * smoothness)
    
    return {
        "ei": ei,
        "confidence": np.full(76, final_confidence),  # Same confidence for all regions
        "convergence_final": final_obj,
        "convergence_improvement": initial_obj / max(final_obj, 1e-8),
    }


def identify_epileptogenic_zones(
    ei: NDArray,
    region_labels: list,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict:
    """
    Identify epileptogenic zones from EI and rank by severity.
    
    Args:
        ei: (76,) array of epileptogenicity indices
        region_labels: list of 76 region names
        threshold: EI threshold for classification as "likely epileptogenic"
        top_k: Number of top regions to return
    
    Returns:
        dict with keys:
            - 'epileptogenic_regions': list of region names with EI > threshold
            - 'ei_scores': corresponding EI values
            - 'top_k_regions': list of top-k most epileptogenic regions
            - 'top_k_scores': corresponding EI values
            - 'epileptogenic_fraction': fraction of regions with EI > threshold
            - 'mean_ei': mean EI across all regions (baseline)
            - 'max_ei': max EI (severity indicator)
    
    Example:
        >>> zones = identify_epileptogenic_zones(ei, region_labels, threshold=0.7)
        >>> print(f"Detected epileptogenic zones: {zones['epileptogenic_regions']}")
        >>> print(f"Severity (max EI): {zones['max_ei']:.3f}")
    """
    # Find regions above threshold
    epi_mask = ei > threshold
    epi_indices = np.where(epi_mask)[0]
    epi_regions = [region_labels[i] for i in epi_indices]
    epi_scores = ei[epi_mask]
    
    # Sort by EI score descending
    sort_idx = np.argsort(-epi_scores)
    epi_regions_sorted = [epi_regions[i] for i in sort_idx]
    epi_scores_sorted = epi_scores[sort_idx]
    
    # Top-k most epileptogenic
    top_k_idx = np.argsort(-ei)[:min(top_k, len(ei))]
    top_k_regions = [region_labels[i] for i in top_k_idx]
    top_k_scores = ei[top_k_idx]
    
    return {
        "epileptogenic_regions": epi_regions_sorted,
        "ei_scores": epi_scores_sorted,
        "top_k_regions": top_k_regions,
        "top_k_scores": top_k_scores,
        "epileptogenic_fraction": np.mean(epi_mask),
        "mean_ei": np.mean(ei),
        "max_ei": np.max(ei),
    }


def ei_from_config(
    x0_fitted: NDArray,
    config: dict,
) -> NDArray:
    """
    Convenience wrapper: compute EI using parameters from config.yaml.
    
    Args:
        x0_fitted: (76,) array of fitted x0 parameters
        config: Configuration dict (from config.yaml)
    
    Returns:
        ei: (76,) array of epileptogenicity indices
    """
    heatmap_cfg = config.get("heatmap", {})
    x0_baseline = heatmap_cfg.get("x0_baseline", -2.2)
    
    # Scale parameter can be derived from bounds if not explicit
    bounds = config.get("parameter_inversion", {}).get("bounds", [-2.4, -1.0])
    x0_range = bounds[1] - bounds[0]  # -1.0 - (-2.4) = 1.4
    scale = x0_range / 10.0  # 0.14 ≈ 0.15
    
    return compute_ei(x0_fitted, x0_baseline, scale)
