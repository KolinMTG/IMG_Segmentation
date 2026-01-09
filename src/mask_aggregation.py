"""
Probabilistic mask aggregation with spatial coherence and noise suppression.

This module implements neighborhood-based voting for multi-class segmentation,
reducing pixel-level noise through spatial probability aggregation and
morphological post-processing.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.morphology import remove_small_objects, opening, closing, disk
from typing import Dict, Optional, Tuple, Union



# ============================================================================
# KERNEL GENERATION
# ============================================================================

def create_gaussian_kernel(size: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    Create a normalized 2D Gaussian kernel.
    
    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation (default: size/6 for ~3σ coverage)
        
    Returns:
        Normalized Gaussian kernel of shape (size, size)
        
    Example:
        >>> kernel = create_gaussian_kernel(5, sigma=1.0)
        >>> kernel.shape
        (5, 5)
        >>> np.isclose(kernel.sum(), 1.0)
        True
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}")
    
    if sigma is None:
        sigma = size / 6.0  # 3-sigma rule covers ~99.7% of distribution
    
    # Create coordinate grids
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute Gaussian
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize to sum to 1
    kernel = kernel / kernel.sum()
    
    return kernel


def create_uniform_kernel(size: int) -> np.ndarray:
    """
    Create a normalized uniform kernel.
    
    Args:
        size: Kernel size (must be odd)
        
    Returns:
        Normalized uniform kernel of shape (size, size)
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}")
    
    kernel = np.ones((size, size), dtype=np.float32)
    kernel = kernel / kernel.sum()
    
    return kernel


# ============================================================================
# SPATIAL PROBABILITY AGGREGATION
# ============================================================================

def aggregate_spatial_probabilities(
    class_masks: Dict[int, np.ndarray],
    kernel_size: int = 5,
    kernel_type: str = 'gaussian',
    sigma: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    Apply spatial probability aggregation using neighborhood voting.
    
    For each class, convolves the probability mask with a Gaussian kernel
    to compute contextual probabilities based on local neighborhoods.
    
    Intuition: If surrounding pixels have high probability for class C,
    the central pixel likely also belongs to class C.
    
    Args:
        class_masks: Dictionary {class_id: probability_map}
                    Each map is (H, W) with values in [0, 1]
        kernel_size: Size of convolution kernel (must be odd)
        kernel_type: 'gaussian' or 'uniform'
        sigma: Standard deviation for Gaussian (default: kernel_size/6)
        
    Returns:
        Dictionary {class_id: smoothed_probability_map}
        Each smoothed map represents contextual probability
        
    Example:
        >>> masks = {0: field_mask, 1: building_mask}
        >>> smoothed = aggregate_spatial_probabilities(masks, kernel_size=5)
        >>> # smoothed[1] = probability that pixel is building given neighborhood
    """
    if not class_masks:
        raise ValueError("class_masks dictionary is empty")
    
    # Create kernel
    if kernel_type == 'gaussian':
        kernel = create_gaussian_kernel(kernel_size, sigma)
    elif kernel_type == 'uniform':
        kernel = create_uniform_kernel(kernel_size)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
    # Smooth each class probability map
    smoothed_masks = {}
    
    for class_id, mask in class_masks.items():
        if mask.ndim != 2:
            raise ValueError(f"Mask for class {class_id} must be 2D")
        
        # Convolve with kernel (neighborhood voting)
        smoothed = ndimage.convolve(mask, kernel, mode='reflect')
        
        # Ensure values stay in [0, 1] (numerical precision)
        smoothed = np.clip(smoothed, 0.0, 1.0)
        
        smoothed_masks[class_id] = smoothed
    
    return smoothed_masks


def aggregate_spatial_probabilities_fast(
    class_masks: Dict[int, np.ndarray],
    kernel_size: int = 5,
    sigma: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    Fast spatial aggregation using separable Gaussian filtering.
    
    More efficient than full 2D convolution for large kernels.
    
    Args:
        class_masks: Dictionary {class_id: probability_map}
        kernel_size: Size of Gaussian kernel (approximate)
        sigma: Standard deviation (default: kernel_size/6)
        
    Returns:
        Dictionary {class_id: smoothed_probability_map}
    """
    if sigma is None:
        sigma = kernel_size / 6.0
    
    smoothed_masks = {}
    
    for class_id, mask in class_masks.items():
        # Gaussian filter is separable: much faster
        smoothed = gaussian_filter(mask, sigma=sigma, mode='reflect')
        smoothed = np.clip(smoothed, 0.0, 1.0)
        smoothed_masks[class_id] = smoothed
    
    return smoothed_masks


# ============================================================================
# MULTI-CLASS DECISION
# ============================================================================

def assign_max_probability_class(
    smoothed_masks: Dict[int, np.ndarray],
    confidence_threshold: Optional[float] = None,
    background_class: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each pixel to the class with maximum contextual probability.
    
    Args:
        smoothed_masks: Dictionary {class_id: smoothed_probability_map}
        confidence_threshold: Minimum probability to assign class
                             If max prob < threshold, assign background_class
        background_class: Class ID for low-confidence pixels (default: -1)
        
    Returns:
        Tuple of:
        - class_map: (H, W) array of class IDs
        - confidence_map: (H, W) array of maximum probabilities
        
    Example:
        >>> smoothed = {0: field_prob, 1: building_prob, 2: water_prob}
        >>> class_map, confidence = assign_max_probability_class(smoothed)
        >>> # class_map[i,j] = argmax_c(smoothed[c][i,j])
    """
    if not smoothed_masks:
        raise ValueError("smoothed_masks dictionary is empty")
    
    # Stack all probability maps: (H, W, num_classes)
    class_ids = sorted(smoothed_masks.keys())
    prob_stack = np.stack([smoothed_masks[cid] for cid in class_ids], axis=-1)
    
    # Find class with maximum probability
    max_prob_indices = np.argmax(prob_stack, axis=-1)  # (H, W)
    max_probabilities = np.max(prob_stack, axis=-1)    # (H, W)
    
    # Map indices back to class IDs
    class_map = np.zeros_like(max_prob_indices, dtype=np.int32)
    for idx, class_id in enumerate(class_ids):
        class_map[max_prob_indices == idx] = class_id
    
    # Apply confidence threshold if specified
    if confidence_threshold is not None:
        low_confidence = max_probabilities < confidence_threshold
        class_map[low_confidence] = background_class
    
    return class_map, max_probabilities


# ============================================================================
# MORPHOLOGICAL NOISE SUPPRESSION
# ============================================================================

def suppress_noise_per_class(
    class_map: np.ndarray,
    opening_radius: int = 2,
    closing_radius: int = 3,
    min_area: int = 50
) -> np.ndarray:
    """
    Apply morphological filtering per class to suppress noise.
    
    For each class:
    1. Opening: removes isolated noisy pixels (salt noise)
    2. Closing: fills small holes inside objects (pepper noise)
    3. Remove small connected components
    
    Args:
        class_map: (H, W) array of class IDs
        opening_radius: Radius of disk structuring element for opening
        closing_radius: Radius of disk structuring element for closing
        min_area: Minimum area (pixels) for connected components
        
    Returns:
        Cleaned class map with same shape as input
        
    Example:
        >>> noisy_map = assign_max_probability_class(masks)[0]
        >>> clean_map = suppress_noise_per_class(noisy_map, min_area=100)
    """
    cleaned_map = class_map.copy()
    unique_classes = np.unique(class_map)
    
    # Filter out background class if present
    unique_classes = unique_classes[unique_classes >= 0]
    
    for class_id in unique_classes:
        # Create binary mask for this class
        binary_mask = (class_map == class_id).astype(np.uint8)
        
        # Opening: remove small objects (isolated pixels)
        if opening_radius > 0:
            selem = disk(opening_radius)
            binary_mask = opening(binary_mask, selem)
        
        # Closing: fill small holes
        if closing_radius > 0:
            selem = disk(closing_radius)
            binary_mask = closing(binary_mask, selem)
        
        # Remove small connected components
        if min_area > 0:
            binary_mask = remove_small_objects(
                binary_mask.astype(bool),
                min_size=min_area,
                connectivity=2
            ).astype(np.uint8)
        
        # Update cleaned map
        # First, remove this class from cleaned map
        cleaned_map[class_map == class_id] = -1
        # Then, add back only the filtered regions
        cleaned_map[binary_mask > 0] = class_id
    
    return cleaned_map


def suppress_noise_global(
    class_map: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply global morphological filtering to entire segmentation.
    
    Simpler alternative to per-class filtering.
    
    Args:
        class_map: (H, W) array of class IDs
        kernel_size: Size of morphological kernel
        iterations: Number of opening/closing iterations
        
    Returns:
        Cleaned class map
    """
    # Convert to uint8 for OpenCV
    class_map_uint8 = class_map.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    
    # Opening then closing
    cleaned = cv2.morphologyEx(
        class_map_uint8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )
    
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )
    
    return cleaned.astype(np.int32)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_colored_segmentation(
    class_map: np.ndarray,
    color_map: Dict[int, Union[list, tuple]]
) -> np.ndarray:
    """
    Create RGB visualization of segmentation map.
    
    Args:
        class_map: (H, W) array of class IDs
        color_map: Dictionary {class_id: [R, G, B]} with values in [0, 255]
        
    Returns:
        RGB image (H, W, 3) with dtype uint8
        
    Example:
        >>> colors = {0: [0,0,0], 1: [255,0,0], 2: [0,255,0]}
        >>> rgb_img = create_colored_segmentation(class_map, colors)
    """
    h, w = class_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        mask = (class_map == class_id)
        colored[mask] = color
    
    return colored


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def aggregate_masks(
    class_masks: Dict[int, np.ndarray],
    kernel_size: int = 5,
    kernel_type: str = 'gaussian',
    sigma: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    background_class: int = -1,
    opening_radius: int = 2,
    closing_radius: int = 3,
    min_area: int = 50,
    fast_mode: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Complete pipeline for probabilistic mask aggregation.
    
    Steps:
    1. Spatial probability aggregation (neighborhood voting)
    2. Multi-class decision (argmax with optional threshold)
    3. Morphological noise suppression
    
    Args:
        class_masks: Dict {class_id: probability_map (H, W) in [0, 1]}
        kernel_size: Size of smoothing kernel (odd number)
        kernel_type: 'gaussian' or 'uniform'
        sigma: Gaussian standard deviation (default: kernel_size/6)
        confidence_threshold: Min probability to assign class (optional)
        background_class: Class ID for low-confidence pixels
        opening_radius: Morphological opening radius
        closing_radius: Morphological closing radius
        min_area: Minimum component area (pixels)
        fast_mode: Use separable Gaussian (faster for large kernels)
        
    Returns:
        Tuple of:
        - final_segmentation: (H, W) cleaned class map
        - confidence_map: (H, W) maximum probabilities
        - smoothed_masks: Dict {class_id: smoothed probability map}
        
    Example:
        >>> masks = {
        ...     0: field_probability_map,
        ...     1: building_probability_map,
        ...     2: water_probability_map
        ... }
        >>> segmentation, confidence, smoothed = aggregate_masks(
        ...     masks,
        ...     kernel_size=5,
        ...     min_area=100
        ... )
    """
    # Step 1: Spatial probability aggregation
    if fast_mode:
        smoothed_masks = aggregate_spatial_probabilities_fast(
            class_masks,
            kernel_size=kernel_size,
            sigma=sigma
        )
    else:
        smoothed_masks = aggregate_spatial_probabilities(
            class_masks,
            kernel_size=kernel_size,
            kernel_type=kernel_type,
            sigma=sigma
        )
    
    # Step 2: Multi-class decision
    class_map, confidence_map = assign_max_probability_class(
        smoothed_masks,
        confidence_threshold=confidence_threshold,
        background_class=background_class
    )
    
    # Step 3: Morphological noise suppression
    final_segmentation = suppress_noise_per_class(
        class_map,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        min_area=min_area
    )
    
    return final_segmentation, confidence_map, smoothed_masks


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create synthetic example
    print("Creating synthetic probability maps...")
    
    h, w = 512, 512
    
    # Simulate probability maps for 3 classes
    np.random.seed(42)
    
    # Class 0: field (top-left region)
    field_mask = np.zeros((h, w), dtype=np.float32)
    field_mask[:h//2, :w//2] = 0.8
    field_mask += np.random.normal(0, 0.1, (h, w))
    field_mask = np.clip(field_mask, 0, 1)
    
    # Class 1: building (center region)
    building_mask = np.zeros((h, w), dtype=np.float32)
    building_mask[h//4:3*h//4, w//4:3*w//4] = 0.9
    building_mask += np.random.normal(0, 0.15, (h, w))
    building_mask = np.clip(building_mask, 0, 1)
    
    # Class 2: water (bottom-right region)
    water_mask = np.zeros((h, w), dtype=np.float32)
    water_mask[h//2:, w//2:] = 0.85
    water_mask += np.random.normal(0, 0.12, (h, w))
    water_mask = np.clip(water_mask, 0, 1)
    
    class_masks = {
        0: field_mask,
        1: building_mask,
        2: water_mask
    }
    
    print("\nRunning aggregation pipeline...")
    
    # Run pipeline
    final_seg, confidence, smoothed = aggregate_masks(
        class_masks,
        kernel_size=7,
        kernel_type='gaussian',
        confidence_threshold=0.3,
        opening_radius=2,
        closing_radius=3,
        min_area=100
    )
    
    print(f"\nResults:")
    print(f"  Final segmentation shape: {final_seg.shape}")
    print(f"  Unique classes: {np.unique(final_seg)}")
    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    # Class statistics
    print(f"\nClass distribution:")
    for class_id in sorted(class_masks.keys()):
        pixels = np.sum(final_seg == class_id)
        percentage = 100 * pixels / final_seg.size
        print(f"  Class {class_id}: {pixels:6d} pixels ({percentage:5.2f}%)")
    
    # Create colored visualization
    color_map = {
        -1: [128, 128, 128],  # Background (gray)
        0: [0, 255, 0],       # Field (green)
        1: [255, 0, 0],       # Building (red)
        2: [0, 0, 255]        # Water (blue)
    }
    
    colored = create_colored_segmentation(final_seg, color_map)
    print(f"\nColored visualization shape: {colored.shape}")
    
    print("\nDone!")