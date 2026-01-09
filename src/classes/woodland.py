"""
Woodland class detection (Class 2).

Strategy:
- High green channel intensity
- High texture (irregular, natural patterns)
- Dense vegetation signature
"""

import numpy as np
from skimage.feature import local_binary_pattern
from general_processing import (
    apply_gaussian,
    emphasize_channel,
    apply_unsharp_mask,
    to_grayscale,
    compute_local_variance,
    normalize_to_01
)
from cste import ProcessingConfig


def process_woodland(img: np.ndarray) -> np.ndarray:
    """
    Generate woodland likelihood score map.
    
    Detection strategy:
    1. Emphasize green channel (vegetation signature)
    2. Apply sharpening to enhance texture
    3. Compute texture measures (variance and LBP)
    4. Combine green intensity with texture features
    5. Apply smoothing to reduce noise
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing woodland likelihood
    """
    # Emphasize green channel for vegetation
    green_emphasized = emphasize_channel(
        img,
        channel_idx=1,
        factor=ProcessingConfig.GREEN_EMPHASIS_FACTOR
    )
    
    # Sharpen to enhance texture details
    sharpened = apply_unsharp_mask(
        green_emphasized,
        sigma=ProcessingConfig.GAUSSIAN_SIGMA_MEDIUM,
        strength=0.6
    )
    
    # Extract green channel intensity
    green_channel = sharpened[:, :, 1]
    
    # Compute texture variance
    texture_var = compute_local_variance(
        green_channel,
        window_size=ProcessingConfig.WINDOW_SIZE_MEDIUM
    )
    texture_score = normalize_to_01(texture_var)
    
    # Compute Local Binary Pattern for additional texture info
    gray = to_grayscale(sharpened)
    radius = ProcessingConfig.LBP_RADIUS
    n_points = ProcessingConfig.LBP_POINTS_MULTIPLIER * radius
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_norm = normalize_to_01(lbp)
    
    # Compute NDVI-like vegetation index
    # NDVI approximation: (G - R) / (G + R + epsilon)
    red = img[:, :, 0]
    green = img[:, :, 1]
    ndvi_approx = (green - red) / (green + red + 1e-8)
    ndvi_norm = normalize_to_01(ndvi_approx)
    
    # Combine features:
    # - Green intensity: woodlands are green
    # - Texture variance: woodlands have irregular texture
    # - LBP: captures fine-scale texture
    # - NDVI: strong vegetation indicator
    combined = (
        0.30 * green_channel +
        0.25 * texture_score +
        0.20 * lbp_norm +
        0.25 * ndvi_norm
    )
    
    # Smooth score map to reduce noise
    score_smoothed = apply_gaussian(
        combined,
        sigma=ProcessingConfig.GAUSSIAN_SIGMA_MEDIUM
    )
    
    # Normalize final score
    return normalize_to_01(score_smoothed)
