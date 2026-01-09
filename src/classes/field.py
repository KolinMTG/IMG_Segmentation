"""
Field class detection (Class 0).

Strategy:
- Fields have uniform texture with low spatial variance
- Medium vegetation intensity
- Regular patterns from cultivation
"""

import numpy as np
from general_processing import (
    apply_gaussian,
    apply_clahe,
    to_grayscale,
    compute_local_variance,
    normalize_to_01
)
from cste import ProcessingConfig


def process_field(img: np.ndarray) -> np.ndarray:
    """
    Generate field likelihood score map.
    
    Detection strategy:
    1. Apply light smoothing to reduce noise
    2. Enhance local contrast with CLAHE
    3. Compute smoothness score (inverse of local variance)
    4. Favor medium-intensity regions (typical for fields)
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing field likelihood
    """
    # Light smoothing to reduce high-frequency noise
    smoothed = apply_gaussian(img, sigma=ProcessingConfig.GAUSSIAN_SIGMA_LIGHT)
    
    # Local contrast normalization
    normalized = apply_clahe(
        smoothed,
        clip_limit=ProcessingConfig.CLAHE_CLIP_LIMIT_LOW,
        tile_size=ProcessingConfig.CLAHE_TILE_SIZE
    )
    
    # Convert to grayscale
    gray = to_grayscale(normalized)
    
    # Compute smoothness: fields have low local variance
    local_var = compute_local_variance(
        gray,
        window_size=ProcessingConfig.WINDOW_SIZE_MEDIUM
    )
    
    # Smoothness score (high for uniform areas)
    smoothness = 1.0 / (1.0 + local_var * ProcessingConfig.VARIANCE_SCALE_FIELD)
    
    # Intensity preference: fields are typically medium-bright
    # Create a Gaussian-like preference centered around 0.4-0.6
    intensity_center = 0.5
    intensity_width = 0.3
    intensity_score = np.exp(-((gray - intensity_center) ** 2) / (2 * intensity_width ** 2))
    
    # Combine smoothness and intensity preference
    combined = 0.7 * smoothness + 0.3 * intensity_score
    
    # Light smoothing of score map
    score_smoothed = apply_gaussian(combined, sigma=1.5)
    
    # Normalize to [0, 1]
    return normalize_to_01(score_smoothed)
