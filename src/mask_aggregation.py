"""
Probabilistic mask aggregation with spatial coherence and noise suppression.

This module implements neighborhood-based voting for multi-class segmentation,
reducing pixel-level noise through spatial probability aggregation and
morphological post-processing.

Pipeline:
    1. Spatial probability aggregation (Gaussian/uniform smoothing)
    2. Multi-class decision (argmax with optional confidence threshold)
    3. Morphological noise suppression (per-class cleaning)
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, opening, closing, disk
from typing import Dict, Optional, Tuple, Union

from cste import ProcessingConfig


# ============================================================================
# KERNEL GENERATION
# ============================================================================

def create_gaussian_kernel(size: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    Create a normalized 2D Gaussian kernel for spatial filtering.
    
    Args:
        size: Kernel dimension (must be odd, e.g., 3, 5, 7)
        sigma: Standard deviation. If None, uses size/6 for ~3σ coverage
        
    Returns:
        Normalized Gaussian kernel of shape (size, size), sum equals 1.0
        
    Example:
        >>> kernel = create_gaussian_kernel(5, sigma=1.0)
        >>> kernel.shape
        (5, 5)
        >>> np.isclose(kernel.sum(), 1.0)
        True
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}")
    
    # ! Default sigma ensures 3-sigma rule covers 99.7% of distribution
    if sigma is None:
        sigma = size / 6.0
    
    # Create coordinate grids centered at kernel center
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute 2D Gaussian: exp(-(x² + y²) / (2σ²))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # ! Normalize to sum to 1 (preserves probability mass)
    kernel = kernel / kernel.sum()
    
    return kernel


def create_uniform_kernel(size: int) -> np.ndarray:
    """
    Create a normalized uniform (box) kernel for spatial averaging.
    
    Args:
        size: Kernel dimension (must be odd)
        
    Returns:
        Normalized uniform kernel of shape (size, size), all values equal
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
    
    For each class, convolves the probability mask with a kernel to compute
    contextual probabilities. High neighborhood probability → high pixel probability.
    
    Args:
        class_masks: Dictionary {class_id: probability_map (H, W) in [0, 1]}
        kernel_size: Convolution kernel size (must be odd)
        kernel_type: 'gaussian' for weighted voting, 'uniform' for simple averaging
        sigma: Gaussian standard deviation (default: kernel_size/6)
        
    Returns:
        Dictionary {class_id: smoothed_probability_map (H, W) in [0, 1]}
        
    Example:
        >>> masks = {0: field_prob, 1: building_prob, 2: water_prob}
        >>> smoothed = aggregate_spatial_probabilities(masks, kernel_size=5)
    """
    if not class_masks:
        raise ValueError("class_masks dictionary cannot be empty")
    
    # Create convolution kernel
    if kernel_type == 'gaussian':
        kernel = create_gaussian_kernel(kernel_size, sigma)
    elif kernel_type == 'uniform':
        kernel = create_uniform_kernel(kernel_size)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Use 'gaussian' or 'uniform'")
    
    smoothed_masks = {}
    
    for class_id, mask in class_masks.items():
        if mask.ndim != 2:
            raise ValueError(f"Mask for class {class_id} must be 2D, got shape {mask.shape}")
        
        # ! Convolve with kernel for neighborhood-weighted probability
        smoothed = ndimage.convolve(mask, kernel, mode='reflect')
        
        # ! Ensure numerical stability: clip to valid probability range
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
    
    More efficient than 2D convolution for large kernels. Uses scipy's optimized
    Gaussian filter which separates into 1D horizontal and vertical passes.
    
    Args:
        class_masks: Dictionary {class_id: probability_map (H, W)}
        kernel_size: Approximate kernel size (used to compute sigma if not provided)
        sigma: Gaussian standard deviation (default: kernel_size/6)
        
    Returns:
        Dictionary {class_id: smoothed_probability_map}
    """
    if sigma is None:
        sigma = kernel_size / 6.0
    
    smoothed_masks = {}
    
    for class_id, mask in class_masks.items():
        # ! Separable Gaussian filter: O(N·M·σ) vs O(N·M·k²) for 2D convolution
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
    background_class: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each pixel to the class with maximum contextual probability.
    
    Implements winner-takes-all classification with optional confidence thresholding.
    Low-confidence pixels are assigned to background class.
    
    Args:
        smoothed_masks: Dictionary {class_id: smoothed_probability_map}
        confidence_threshold: Minimum probability required for class assignment.
                            If max_prob < threshold, assigns background_class
        background_class: Class ID for low-confidence or filtered pixels (default: 0)
        
    Returns:
        Tuple of:
        - class_map: (H, W) integer array with class IDs (0-5)
        - confidence_map: (H, W) float array with maximum probabilities
        
    Example:
        >>> smoothed = {0: field_prob, 1: building_prob, 2: water_prob}
        >>> class_map, confidence = assign_max_probability_class(smoothed, confidence_threshold=0.5)
    """
    if not smoothed_masks:
        raise ValueError("smoothed_masks dictionary cannot be empty")
    
    # ! Stack probability maps: shape (H, W, num_classes)
    class_ids = sorted(smoothed_masks.keys())
    prob_stack = np.stack([smoothed_masks[cid] for cid in class_ids], axis=-1)
    
    # ! Find class with maximum probability at each pixel
    max_prob_indices = np.argmax(prob_stack, axis=-1)  # (H, W)
    max_probabilities = np.max(prob_stack, axis=-1)    # (H, W)
    
    # ! Map array indices back to actual class IDs
    class_map = np.zeros_like(max_prob_indices, dtype=np.int32)
    for idx, class_id in enumerate(class_ids):
        class_map[max_prob_indices == idx] = class_id
    
    # ! Apply confidence threshold: low-confidence pixels → background
    if confidence_threshold is not None:
        low_confidence_mask = max_probabilities < confidence_threshold
        class_map[low_confidence_mask] = background_class
    
    return class_map, max_probabilities


# ============================================================================
# MORPHOLOGICAL NOISE SUPPRESSION
# ============================================================================

def suppress_noise_per_class(
    class_map: np.ndarray,
    opening_radius: int = 2,
    closing_radius: int = 3,
    min_area: int = 50,
    background_class: int = 0
) -> np.ndarray:
    """
    Apply morphological filtering per class to remove noise and small artifacts.
    
    For each class (excluding background):
    1. Opening: removes isolated noisy pixels (salt noise)
    2. Closing: fills small holes inside objects (pepper noise)
    3. Remove small connected components
    
    ! Filtered pixels are assigned to background_class, NOT negative values.
    
    Args:
        class_map: (H, W) integer array with class IDs
        opening_radius: Disk radius for morphological opening
        closing_radius: Disk radius for morphological closing
        min_area: Minimum area (pixels) for connected components
        background_class: Class ID to assign to filtered pixels (default: 0)
        
    Returns:
        Cleaned class map (H, W) with same integer class IDs
        
    Example:
        >>> noisy_map = assign_max_probability_class(masks)[0]
        >>> clean_map = suppress_noise_per_class(noisy_map, min_area=100, background_class=0)
    """
    cleaned_map = class_map.copy()
    unique_classes = np.unique(class_map)
    
    # ! Filter out background class and any negative values
    unique_classes = unique_classes[(unique_classes >= 0) & (unique_classes != background_class)]
    
    for class_id in unique_classes:
        # Create binary mask for this class
        binary_mask = (class_map == class_id).astype(np.uint8)
        
        # ! Opening: remove small isolated objects
        if opening_radius > 0:
            selem = disk(opening_radius)
            binary_mask = opening(binary_mask, selem)
        
        # ! Closing: fill small holes within objects
        if closing_radius > 0:
            selem = disk(closing_radius)
            binary_mask = closing(binary_mask, selem)
        
        # ! Remove small connected components below area threshold
        if min_area > 0:
            binary_mask = remove_small_objects(
                binary_mask.astype(bool),
                min_size=min_area,
                connectivity=2
            ).astype(np.uint8)
        
        # ! Update cleaned map: filtered pixels → background_class
        # First, clear all pixels of this class
        cleaned_map[class_map == class_id] = background_class
        # Then, restore only pixels that survived filtering
        cleaned_map[binary_mask > 0] = class_id
    
    return cleaned_map


def suppress_noise_global(
    class_map: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply global morphological filtering to entire segmentation map.
    
    Simpler alternative to per-class filtering. Uses elliptical structuring element
    for isotropic noise removal.
    
    Args:
        class_map: (H, W) integer array with class IDs
        kernel_size: Size of morphological structuring element
        iterations: Number of opening/closing iterations
        
    Returns:
        Cleaned class map (H, W)
    """
    # ! Convert to uint8 for OpenCV compatibility
    class_map_uint8 = class_map.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    
    # ! Opening removes small bright regions (isolated class pixels)
    cleaned = cv2.morphologyEx(
        class_map_uint8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )
    
    # ! Closing fills small dark regions (holes in class regions)
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
    Create RGB visualization of segmentation map with class-specific colors.
    
    Args:
        class_map: (H, W) integer array with class IDs
        color_map: Dictionary {class_id: [R, G, B]} with values in [0, 255]
        
    Returns:
        RGB image (H, W, 3) with dtype uint8
        
    Example:
        >>> from cste import ClassInfo
        >>> colors = ClassInfo.CLASS_COLORS
        >>> rgb_img = create_colored_segmentation(class_map, colors)
    """
    h, w = class_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        mask = (class_map == class_id)
        colored[mask] = color
    
    return colored


# ============================================================================
# MAIN AGGREGATION PIPELINE
# ============================================================================

def aggregate_masks(
    class_masks: Dict[int, np.ndarray],
    kernel_size: int = 5,
    kernel_type: str = 'gaussian',
    sigma: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
    background_class: int = ProcessingConfig.BACKGROUND_CLASS,
    opening_radius: int = ProcessingConfig.OPENING_RADIUS,
    closing_radius: int = ProcessingConfig.CLOSING_RADIUS,
    min_area: int = ProcessingConfig.MIN_AREA,
    fast_mode: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Complete pipeline for probabilistic multi-class mask aggregation.
    
    Pipeline steps:
    1. Spatial probability aggregation (neighborhood voting with Gaussian/uniform kernel)
    2. Multi-class decision (argmax classification with optional confidence threshold)
    3. Morphological noise suppression (per-class cleaning, removes artifacts)
    
    ! The final segmentation contains only integer class IDs (0-5), no negative values.
    ! Filtered or low-confidence pixels are assigned to background_class.
    
    Args:
        class_masks: Dict {class_id: probability_map (H, W) in [0, 1]}
                    Keys should be integers 0-5 representing class IDs
        kernel_size: Size of smoothing kernel (odd number, e.g., 5, 7, 9)
        kernel_type: 'gaussian' for weighted voting, 'uniform' for simple averaging
        sigma: Gaussian standard deviation (default: kernel_size/6)
        confidence_threshold: Minimum probability for class assignment (optional)
        background_class: Class ID for low-confidence/filtered pixels (default: 0)
        opening_radius: Morphological opening disk radius
        closing_radius: Morphological closing disk radius
        min_area: Minimum connected component area in pixels
        fast_mode: Use separable Gaussian filter (faster for large kernels)
        
    Returns:
        Tuple of:
        - final_segmentation: (H, W) integer array with class IDs 0-5
        - confidence_map: (H, W) float array with maximum probabilities [0, 1]
        - smoothed_masks: Dict {class_id: smoothed_probability_map (H, W)}
        
    Example:
        >>> masks = {
        ...     0: field_probability_map,
        ...     1: building_probability_map,
        ...     2: water_probability_map,
        ...     3: woodland_probability_map,
        ...     4: road_probability_map
        ... }
        >>> segmentation, confidence, smoothed = aggregate_masks(
        ...     masks,
        ...     kernel_size=5,
        ...     confidence_threshold=0.5,
        ...     min_area=100
        ... )
        >>> assert segmentation.min() >= 0  # No negative values
        >>> assert segmentation.max() <= 4  # Valid class range
    """
    # ! Step 1: Spatial probability aggregation (neighborhood voting)
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
    
    # ! Step 2: Multi-class decision (winner-takes-all with optional threshold)
    class_map, confidence_map = assign_max_probability_class(
        smoothed_masks,
        confidence_threshold=confidence_threshold,
        background_class=background_class
    )
    
    # ! Step 3: Morphological noise suppression (removes artifacts, assigns background)
    final_segmentation = suppress_noise_per_class(
        class_map,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        min_area=min_area,
        background_class=background_class
    )
    
    # ! Ensure no negative values in final output
    final_segmentation = np.maximum(final_segmentation, 0)
    return final_segmentation, confidence_map, smoothed_masks