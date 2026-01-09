"""
Road class detection (Class 4).

Strategy:
- Strong linear edges
- Uniform texture within road surface
- High local contrast at boundaries
- High priority: favor recall over precision
"""

import numpy as np
from scipy import ndimage
from general_processing import (
    to_grayscale,
    apply_clahe,
    compute_canny_edges,
    compute_sobel_magnitude,
    apply_morphology,
    compute_edge_density,
    apply_gaussian,
    normalize_to_01
)
from cste import ProcessingConfig


def process_road(img: np.ndarray) -> np.ndarray:
    """
    Generate road likelihood score map.
    
    Detection strategy:
    1. Enhance contrast with strong CLAHE
    2. Detect edges using Canny (captures linear structures)
    3. Apply morphological operations to connect linear features
    4. Compute edge density and Sobel magnitude
    5. Detect oriented structures (roads are linear)
    6. Combine features with high recall bias
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing road likelihood
    """
    # Convert to grayscale
    gray = to_grayscale(img)
    
    # Strong contrast enhancement
    contrast_enhanced = apply_clahe(
        gray,
        clip_limit=ProcessingConfig.CLAHE_CLIP_LIMIT_HIGH,
        tile_size=ProcessingConfig.CLAHE_TILE_SIZE
    )
    
    # Detect edges using Canny (good for linear structures)
    canny_edges = compute_canny_edges(
        contrast_enhanced,
        low_thresh=ProcessingConfig.CANNY_LOW_THRESHOLD,
        high_thresh=ProcessingConfig.CANNY_HIGH_THRESHOLD
    )
    
    # Dilate edges to strengthen detection
    edges_dilated = apply_morphology(
        canny_edges,
        operation='dilate',
        kernel_size=ProcessingConfig.MORPH_KERNEL_SIZE_SMALL,
        iterations=2
    )
    
    # Close edges to connect linear structures
    edges_closed = apply_morphology(
        edges_dilated,
        operation='close',
        kernel_size=ProcessingConfig.MORPH_KERNEL_SIZE_MEDIUM,
        iterations=1
    )
    
    # Compute local edge density
    edge_density = compute_edge_density(
        edges_closed,
        kernel_size=ProcessingConfig.WINDOW_SIZE_XLARGE
    )
    
    # Compute Sobel magnitude for additional edge information
    sobel_mag = compute_sobel_magnitude(contrast_enhanced)
    sobel_norm = normalize_to_01(sobel_mag)
    
    # Detect oriented structures using directional filtering
    # Roads have strong orientation (horizontal or vertical in many cases)
    # Use Gabor-like filtering with Sobel directional components
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    
    # Compute orientation strength (roads have consistent orientation)
    orientation_strength = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation_norm = normalize_to_01(orientation_strength)
    
    # Compute local orientation consistency
    # Roads maintain consistent direction over local windows
    from scipy.ndimage import uniform_filter
    window = 7
    
    # Smooth orientation components
    sx_smooth = uniform_filter(sobel_x, size=window)
    sy_smooth = uniform_filter(sobel_y, size=window)
    
    # Orientation consistency score
    consistency = np.sqrt(sx_smooth**2 + sy_smooth**2)
    consistency_norm = normalize_to_01(consistency)
    
    # Combine features with weights favoring high recall:
    # - Edge density: roads have concentrated edges
    # - Sobel magnitude: roads have strong boundaries
    # - Orientation strength: roads are linear
    # - Consistency: roads maintain direction
    combined = (
        0.35 * edge_density +
        0.25 * sobel_norm +
        0.20 * orientation_norm +
        0.20 * consistency_norm
    )
    
    # Apply dilation to expand detected regions (increase recall)
    combined_expanded = apply_morphology(
        combined,
        operation='dilate',
        kernel_size=ProcessingConfig.MORPH_KERNEL_SIZE_SMALL,
        iterations=1
    )
    
    # Smooth final score
    score_smoothed = apply_gaussian(
        combined_expanded,
        sigma=ProcessingConfig.GAUSSIAN_SIGMA_MEDIUM
    )
    
    # Normalize to [0, 1]
    return normalize_to_01(score_smoothed)
