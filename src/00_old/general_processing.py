"""
Shared preprocessing and feature extraction utilities.
"""

import numpy as np
import cv2
from skimage.filters import gaussian
from scipy import ndimage
from typing import Optional


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_to_01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize array to [0, 1] range using min-max scaling.
    
    Args:
        img: Input array
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Normalized array in [0, 1]
    """
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < eps:
        return np.zeros_like(img, dtype=np.float32)
    
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminosity method.
    
    Args:
        img: RGB image (H, W, 3)
        
    Returns:
        Grayscale image (H, W)
    """
    return np.dot(img, [0.299, 0.587, 0.114])


# ============================================================================
# SMOOTHING
# ============================================================================

def apply_gaussian(
    img: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian smoothing.
    
    Args:
        img: Input image (grayscale or RGB)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed image
    """
    if len(img.shape) == 3:
        return gaussian(img, sigma=sigma, channel_axis=2, preserve_range=True)
    else:
        return gaussian(img, sigma=sigma, preserve_range=True)


# ============================================================================
# CONTRAST ENHANCEMENT
# ============================================================================

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        img: Input image (grayscale or RGB), values in [0, 1]
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image in [0, 1]
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for i in range(img.shape[2]):
            channel = (img[:, :, i] * 255).astype(np.uint8)
            result[:, :, i] = clahe.apply(channel) / 255.0
        return result
    else:
        channel = (img * 255).astype(np.uint8)
        return clahe.apply(channel) / 255.0


def apply_unsharp_mask(
    img: np.ndarray,
    sigma: float = 1.5,
    strength: float = 0.6
) -> np.ndarray:
    """
    Apply unsharp masking for image sharpening.
    
    Args:
        img: Input image
        sigma: Standard deviation for Gaussian blur
        strength: Sharpening strength factor
        
    Returns:
        Sharpened image
    """
    blurred = apply_gaussian(img, sigma=sigma)
    sharpened = img + strength * (img - blurred)
    return np.clip(sharpened, 0, 1)


# ============================================================================
# EDGE DETECTION
# ============================================================================

def compute_sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edge magnitude for grayscale image.
    
    Args:
        img: Grayscale image in [0, 1]
        
    Returns:
        Edge magnitude map
    """
    img_uint8 = (img * 255).astype(np.uint8)
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude


def compute_canny_edges(
    img: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150
) -> np.ndarray:
    """
    Compute Canny edges for grayscale image.
    
    Args:
        img: Grayscale image in [0, 1]
        low_thresh: Lower threshold for edge detection
        high_thresh: Upper threshold for edge detection
        
    Returns:
        Binary edge map in [0, 1]
    """
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, low_thresh, high_thresh)
    return edges.astype(np.float32) / 255.0


# ============================================================================
# MORPHOLOGICAL OPERATIONS
# ============================================================================

def apply_morphology(
    img: np.ndarray,
    operation: str,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological operation.
    
    Args:
        img: Input binary or grayscale image in [0, 1]
        operation: 'dilate', 'erode', 'close', or 'open'
        kernel_size: Size of morphological kernel
        iterations: Number of times to apply operation
        
    Returns:
        Processed image in [0, 1]
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)
    
    op_map = {
        'dilate': cv2.MORPH_DILATE,
        'erode': cv2.MORPH_ERODE,
        'close': cv2.MORPH_CLOSE,
        'open': cv2.MORPH_OPEN
    }
    
    if operation in ['dilate', 'erode']:
        if operation == 'dilate':
            result = cv2.dilate(img_uint8, kernel, iterations=iterations)
        else:
            result = cv2.erode(img_uint8, kernel, iterations=iterations)
    else:
        result = cv2.morphologyEx(
            img_uint8,
            op_map[operation],
            kernel,
            iterations=iterations
        )
    
    return result.astype(np.float32) / 255.0


# ============================================================================
# TEXTURE ANALYSIS
# ============================================================================

def compute_local_variance(
    img: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Compute local variance for texture analysis.
    
    Args:
        img: Grayscale image
        window_size: Size of local window
        
    Returns:
        Local variance map
    """
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(img, kernel, mode='reflect')
    local_mean_sq = ndimage.convolve(img ** 2, kernel, mode='reflect')
    local_variance = local_mean_sq - local_mean ** 2
    
    return np.maximum(local_variance, 0)


def compute_local_std(
    img: np.ndarray,
    window_size: int = 7
) -> np.ndarray:
    """
    Compute local standard deviation.
    
    Args:
        img: Grayscale image
        window_size: Size of local window
        
    Returns:
        Local standard deviation map
    """
    return ndimage.generic_filter(img, np.std, size=window_size)


def compute_edge_density(
    edge_map: np.ndarray,
    kernel_size: int = 9
) -> np.ndarray:
    """
    Compute local edge density.
    
    Args:
        edge_map: Binary edge map
        kernel_size: Size of integration window
        
    Returns:
        Edge density map
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return ndimage.convolve(edge_map, kernel, mode='reflect')


# ============================================================================
# COLOR SPACE OPERATIONS
# ============================================================================

def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB to HSV color space.
    
    Args:
        img: RGB image in [0, 1]
        
    Returns:
        HSV image with H in [0, 180], S in [0, 1], V in [0, 1]
    """
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    # Normalize S and V to [0, 1]
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 1:] /= 255.0
    
    return hsv_float


def emphasize_channel(
    img: np.ndarray,
    channel_idx: int,
    factor: float
) -> np.ndarray:
    """
    Emphasize a specific color channel.
    
    Args:
        img: RGB image
        channel_idx: Channel index (0=R, 1=G, 2=B)
        factor: Multiplication factor
        
    Returns:
        Image with emphasized channel
    """
    result = img.copy()
    result[:, :, channel_idx] = np.clip(
        result[:, :, channel_idx] * factor,
        0,
        1
    )
    return result
