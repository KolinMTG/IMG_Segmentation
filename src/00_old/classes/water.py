"""
Water class detection (Class 3).

Strategy:
- High blue channel intensity
- Very smooth texture (low variance)
- Low saturation
"""

import numpy as np
from general_processing import (
    apply_gaussian,
    emphasize_channel,
    to_grayscale,
    compute_local_variance,
    rgb_to_hsv,
    normalize_to_01
)
from cste import ProcessingConfig


def process_water(img: np.ndarray) -> np.ndarray:
    """
    Generate water likelihood score map.
    
    Detection strategy:
    1. Apply smoothing to reduce noise
    2. Emphasize blue channel (water signature)
    3. Compute smoothness score (water is very uniform)
    4. Extract low saturation regions (water is often desaturated)
    5. Combine blue intensity, smoothness, and low saturation
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing water likelihood
    """
    # Smooth to reduce high-frequency noise
    smoothed = apply_gaussian(
        img,
        sigma=ProcessingConfig.GAUSSIAN_SIGMA_HEAVY
    )
    
    # Emphasize blue channel
    blue_emphasized = emphasize_channel(
        smoothed,
        channel_idx=2,
        factor=ProcessingConfig.BLUE_EMPHASIS_FACTOR
    )
    
    # Extract blue channel intensity
    blue_channel = blue_emphasized[:, :, 2]
    
    # Compute smoothness: water has very low variance
    gray = to_grayscale(blue_emphasized)
    local_var = compute_local_variance(
        gray,
        window_size=ProcessingConfig.WINDOW_SIZE_LARGE
    )
    
    # High smoothness score for uniform regions
    smoothness = 1.0 / (1.0 + local_var * ProcessingConfig.VARIANCE_SCALE_WATER)
    
    # Extract saturation from HSV
    hsv = rgb_to_hsv(blue_emphasized)
    saturation = hsv[:, :, 1]
    
    # Water typically has low saturation
    low_saturation_score = 1.0 - saturation
    
    # Additional feature: darkness preference
    # Water bodies can be dark, especially in shadows
    value = hsv[:, :, 2]
    
    # Prefer both dark water (shadows) and medium-bright water (sunlit)
    # Use a bimodal preference
    dark_preference = np.exp(-((value - 0.3) ** 2) / (2 * 0.2 ** 2))
    bright_preference = np.exp(-((value - 0.6) ** 2) / (2 * 0.25 ** 2))
    intensity_preference = np.maximum(dark_preference, bright_preference)
    
    # Combine features:
    # - Blue intensity: primary water indicator
    # - Smoothness: water is very uniform
    # - Low saturation: water is desaturated
    # - Intensity preference: handles different lighting conditions
    combined = (
        0.40 * blue_channel +
        0.30 * smoothness +
        0.20 * low_saturation_score +
        0.10 * intensity_preference
    )
    
    # Smooth final score map
    score_smoothed = apply_gaussian(
        combined,
        sigma=ProcessingConfig.GAUSSIAN_SIGMA_HEAVY
    )
    
    # Normalize to [0, 1]
    return normalize_to_01(score_smoothed)
