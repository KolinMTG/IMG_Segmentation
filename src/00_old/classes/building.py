"""
Building class detection (Class 1).

Strategy:
- Buildings have strong edges and high local contrast
- Rectangular or geometric shapes
- High priority: favor recall over precision
"""

import numpy as np
from general_processing import (
    to_grayscale,
    apply_clahe,
    compute_sobel_magnitude,
    apply_morphology,
    compute_edge_density,
    compute_local_std,
    normalize_to_01
)
from cste import ProcessingConfig
from scipy import ndimage


def process_building(img: np.ndarray) -> np.ndarray:
    """
    Generate building likelihood score map.
    
    Detection strategy:
    1. Enhance local contrast with CLAHE
    2. Detect strong edges using Sobel
    3. Apply morphological operations to connect nearby edges
    4. Compute edge density and local contrast
    5. Combine multiple features with high recall bias
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing building likelihood
    """
    # Convert to grayscale
    gray = to_grayscale(img)
    
    # Strong contrast enhancement for building detection
    contrast_enhanced = apply_clahe(
        gray,
        clip_limit=ProcessingConfig.CLAHE_CLIP_LIMIT_MEDIUM,
        tile_size=ProcessingConfig.CLAHE_TILE_SIZE
    )
    
    # Compute edge strength using Sobel
    edges = compute_sobel_magnitude(contrast_enhanced)
    edges_norm = normalize_to_01(edges)
    
    # Close gaps in edges to form continuous structures
    edges_closed = apply_morphology(
        edges_norm,
        operation='close',
        kernel_size=ProcessingConfig.MORPH_KERNEL_SIZE_SMALL,
        iterations=1
    )
    
    # Compute local edge density (buildings have high edge concentration)
    edge_density = compute_edge_density(
        edges_closed,
        kernel_size=ProcessingConfig.WINDOW_SIZE_LARGE
    )
    
    # Compute local standard deviation (buildings have high local contrast)
    local_contrast = compute_local_std(
        contrast_enhanced,
        window_size=ProcessingConfig.WINDOW_SIZE_MEDIUM
    )
    local_contrast_norm = normalize_to_01(local_contrast)
    
    # Additional feature: intensity discontinuity
    # Buildings often have sharp brightness transitions

    gradient_mag = np.sqrt(
        ndimage.sobel(gray, axis=0) ** 2 +
        ndimage.sobel(gray, axis=1) ** 2
    )
    gradient_norm = normalize_to_01(gradient_mag)
    
    # Combine features with weights favoring high recall
    # Higher weights on edge features to catch more buildings
    combined = (
        0.45 * edge_density +
        0.30 * local_contrast_norm +
        0.25 * gradient_norm
    )
    
    # Apply slight dilation to expand detected regions (increase recall)
    combined_expanded = apply_morphology(
        combined,
        operation='dilate',
        kernel_size=ProcessingConfig.MORPH_KERNEL_SIZE_SMALL,
        iterations=1
    )
    
    # Normalize final score
    return normalize_to_01(combined_expanded)
