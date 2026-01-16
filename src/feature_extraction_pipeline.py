"""
Optimized feature extraction and batch processing pipeline for large image datasets.

This module provides a highly optimized implementation with:
- Vectorized operations (no Python loops over pixels)
- Early downsampling for memory efficiency
- Cached intermediate computations
- Numba JIT compilation for unavoidable loops
- Multiprocessing for batch operations
- Comprehensive 19-dimensional feature extraction
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, corner_harris
from scipy import ndimage
from typing import Optional, Tuple, Dict, List
import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from numba import jit, prange
from functools import partial

from cste import ClassInfo, DataPath, GeneralConfig, ProcessingConfig
from data_augmentation import augment_segmentation_data
from logger import get_logger


log = get_logger("optimized_feature_extraction")

# ============================================================================
# CORE UTILITIES
# ============================================================================

def normalize_to_01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize array to [0, 1] range using min-max scaling."""

    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < eps:
        return np.zeros_like(img, dtype=np.float32)
    
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def downsample_image(img: np.ndarray, fraction: float) -> np.ndarray:
    """Downsample image by given fraction using area interpolation.
    Return new image and new shape."""
    if fraction >= 1.0:
        log.warning("Trying to downsample with fraction >= 1.0, returning original image")
        return img
    
    h, w = img.shape[:2]
    new_h, new_w = int(h * fraction), int(w * fraction)
    new_shape = (new_w, new_h)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), new_shape


# ============================================================================
# OPTIMIZED COLOR SPACE CONVERSIONS
# ============================================================================

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminosity method
    Args: img: RGB image (H, W, 3)
    Returns: Grayscale image (H, W)
    """
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Optimized RGB to HSV conversion.
    
    Args:
        img: RGB image in [0, 1]
        
    Returns:
        HSV image with all channels in [0, 1]
    """
    # Convert to uint8 once for OpenCV
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    # Vectorized normalization
    hsv_float = hsv.astype(np.float32)
    hsv_float[..., 0] /= 179.0  # H: [0, 179] -> [0, 1]
    hsv_float[..., 1:] /= 255.0  # S, V: [0, 255] -> [0, 1]
    
    return hsv_float


# ============================================================================
# OPTIMIZED TEXTURE FEATURES (VECTORIZED)
# ============================================================================

def compute_local_variance_fast(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Vectorized local variance using box filters (no Python loops).
    
    Uses: Var(X) = E[X²] - E[X]²
    
    Args:
        img: Grayscale image
        window_size: Size of local window
        
    Returns:
        Local variance map
    """
    # Use OpenCV boxFilter for speed
    kernel_area = window_size * window_size
    
    # E[X]
    mean = cv2.boxFilter(img, -1, (window_size, window_size), normalize=True)
    
    # E[X²]
    mean_sq = cv2.boxFilter(img ** 2, -1, (window_size, window_size), normalize=True)
    
    # Var(X) = E[X²] - E[X]²
    variance = mean_sq - mean ** 2
    
    return np.maximum(variance, 0).astype(np.float32)


@jit(nopython=True, parallel=True, fastmath=True)
def _compute_entropy_numba(img_quantized: np.ndarray, window_size: int) -> np.ndarray:
    """
    Numba-accelerated local entropy computation.
    
    Args:
        img_quantized: Quantized image (H, W) with values [0, 15]
        window_size: Size of local window
        
    Returns:
        Local entropy map
    """
    h, w = img_quantized.shape
    entropy = np.zeros((h, w), dtype=np.float32)
    half_win = window_size // 2
    
    for i in prange(h):
        for j in range(w):
            # Extract window
            i_min = max(0, i - half_win)
            i_max = min(h, i + half_win + 1)
            j_min = max(0, j - half_win)
            j_max = min(w, j + half_win + 1)
            
            window = img_quantized[i_min:i_max, j_min:j_max]
            
            # Compute histogram
            hist = np.zeros(16, dtype=np.float32)
            for ii in range(window.shape[0]):
                for jj in range(window.shape[1]):
                    hist[window[ii, jj]] += 1
            
            # Compute entropy
            total = window.size
            ent = 0.0
            for k in range(16):
                if hist[k] > 0:
                    p = hist[k] / total
                    ent -= p * np.log2(p)
            
            entropy[i, j] = ent
    
    return entropy


def compute_local_entropy_fast(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Fast local entropy computation with Numba acceleration.
    
    Args:
        img: Grayscale image in [0, 1]
        window_size: Size of local window
        
    Returns:
        Local entropy map
    """
    # Quantize to 16 bins for speed
    img_quantized = (img * 15).astype(np.uint8)
    
    # Use Numba for the sliding window operation
    return _compute_entropy_numba(img_quantized, window_size)


def compute_edge_density_fast(edge_map: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Vectorized edge density using box filter.
    
    Args:
        edge_map: Binary edge map
        kernel_size: Size of integration window
        
    Returns:
        Edge density map
    """
    return cv2.boxFilter(edge_map, -1, (kernel_size, kernel_size), normalize=True)


# ============================================================================
# OPTIMIZED GRADIENT FEATURES
# ============================================================================

def compute_gradients_fast(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation efficiently.
    
    Args:
        gray: Grayscale image
        
    Returns:
        (sobel_x, sobel_y, gradient_magnitude)
    """
    # Use OpenCV Sobel for speed
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Vectorized magnitude
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    return sobel_x, sobel_y, gradient_mag


def compute_gradient_anisotropy_fast(gray: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Optimized gradient anisotropy computation using vectorized operations.
    
    Anisotropy = (λ1 - λ2) / (λ1 + λ2 + ε)
    
    Args:
        gray: Grayscale image
        window_size: Size of local window for structure tensor
        
    Returns:
        Anisotropy map in [0, 1]
    """
    # Compute gradients
    sobel_x, sobel_y, _ = compute_gradients_fast(gray)
    
    # Structure tensor components using box filters
    Ixx = cv2.boxFilter(sobel_x * sobel_x, -1, (window_size, window_size), normalize=True)
    Iyy = cv2.boxFilter(sobel_y * sobel_y, -1, (window_size, window_size), normalize=True)
    Ixy = cv2.boxFilter(sobel_x * sobel_y, -1, (window_size, window_size), normalize=True)
    
    # Eigenvalues (vectorized)
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = np.sqrt(np.maximum(trace ** 2 / 4 - det, 0))
    
    lambda1 = trace / 2 + discriminant
    lambda2 = trace / 2 - discriminant
    
    # Anisotropy measure
    anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
    
    return np.clip(anisotropy, 0, 1).astype(np.float32)


# ============================================================================
# SPECTRAL INDICES (VECTORIZED)
# ============================================================================

def compute_ndvi_fast(img: np.ndarray) -> np.ndarray:
    """
    Vectorized NDVI approximation: (G - R) / (G + R).
    
    Args:
        img: RGB image in [0, 1]
        
    Returns:
        NDVI-like index in [-1, 1]
    """
    red = img[..., 0]
    green = img[..., 1]
    
    denominator = green + red + 1e-8
    ndvi = (green - red) / denominator
    
    return ndvi.astype(np.float32)


def compute_water_index_fast(img: np.ndarray) -> np.ndarray:
    """
    Vectorized water index: blue / (red + blue).
    
    Args:
        img: RGB image in [0, 1]
        
    Returns:
        Water index in [0, 1]
    """
    red = img[..., 0]
    blue = img[..., 2]
    
    water_idx = blue / (red + blue + 1e-8)
    
    return np.clip(water_idx, 0, 1).astype(np.float32)


# ============================================================================
# GEOMETRIC FEATURES
# ============================================================================

def compute_corner_density_fast(gray: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    Fast corner density using Harris detector and box filter.
    
    Args:
        gray: Grayscale image in [0, 1]
        window_size: Size of local window for counting corners
        
    Returns:
        Corner density map in [0, 1]
    """
    # Harris corner response
    harris_response = corner_harris(gray, sigma=1.5, k=0.04)
    
    # Threshold to get binary corner map
    threshold = harris_response.mean() + 2 * harris_response.std()
    corners = (harris_response > threshold).astype(np.float32)
    
    # Vectorized density using box filter
    corner_density = cv2.boxFilter(
        corners, -1, (window_size, window_size), normalize=True
    )
    
    return normalize_to_01(corner_density)


# ============================================================================
# CACHED PREPROCESSING
# ============================================================================

class ImageCache:
    """Cache for intermediate computations to avoid recomputation."""
    
    def __init__(self, img: np.ndarray, downsample_fraction: float):
        """
        Initialize cache with downsampled image.
        
        Args:
            img: Original RGB image in [0, 1]
            downsample_fraction: Downsampling fraction
        """
        # Downsample ONCE at the beginning
        self.img_down, _ = downsample_image(img, downsample_fraction)
        
        # Cached conversions
        self._gray = None
        self._hsv = None
        self._gradients = None
        
    @property
    def gray(self) -> np.ndarray:
        """Lazy grayscale conversion."""
        if self._gray is None:
            self._gray = to_grayscale(self.img_down)
        return self._gray
    
    @property
    def hsv(self) -> np.ndarray:
        """Lazy HSV conversion."""
        if self._hsv is None:
            self._hsv = rgb_to_hsv(self.img_down)
        return self._hsv
    
    @property
    def gradients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lazy gradient computation."""
        if self._gradients is None:
            self._gradients = compute_gradients_fast(self.gray)
        return self._gradients


# ============================================================================
# MAIN FEATURE EXTRACTION
# ============================================================================

def extract_features(
    img: np.ndarray,
    normalize: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    downsample_fraction: float = ProcessingConfig.DOWNSAMPLE_FRACTION
) -> np.ndarray:
    """
    Extract comprehensive multi-scale feature vector for each pixel (OPTIMIZED).
    
    Feature vector (19 dimensions):
    1. Color features (RGB, HSV) - 6 features
    2. Grayscale and multi-scale intensity - 4 features
    3. Gradient features (magnitude, orientation) - 2 features
    4. Texture features (variance, entropy, LBP) - 3 features
    5. Spectral indices (NDVI, water index) - 2 features
    6. Geometric features (anisotropy, corner density) - 2 features
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        normalize: If True, normalize each feature to [0, 1]
        save: If True, save features to disk
        save_path: Path to save features (.npy)
        downsample_fraction: Downsampling fraction (default from config)
        
    Returns:
        Feature tensor (H', W', 19) where H', W' are downsampled dimensions
    """
    # log.info(f"Extracting features (downsample={downsample_fraction}, normalize={normalize})")
    
    # Initialize cache with downsampled image
    cache = ImageCache(img, downsample_fraction)
    img_down = cache.img_down
    gray = cache.gray
    hsv = cache.hsv
    sobel_x, sobel_y, gradient_mag = cache.gradients
    
    H, W = img_down.shape[:2]
    features = np.zeros((H, W, 19), dtype=np.float32)
    
    # ========================================================================
    # COLOR FEATURES (6) - Direct assignment, no conversion needed
    # ========================================================================
    
    features[..., 0] = img_down[..., 0]  # Red
    features[..., 1] = img_down[..., 1]  # Green
    features[..., 2] = img_down[..., 2]  # Blue
    
    features[..., 3] = hsv[..., 0]  # Hue
    features[..., 4] = hsv[..., 1]  # Saturation
    features[..., 5] = hsv[..., 2]  # Value
    
    # ========================================================================
    # INTENSITY FEATURES (4)
    # ========================================================================
    
    features[..., 6] = gray
    
    # Multi-scale blurred intensities using Gaussian
    features[..., 7] = cv2.GaussianBlur(gray, (0, 0), 1.0)
    features[..., 8] = cv2.GaussianBlur(gray, (0, 0), 2.5)
    features[..., 9] = cv2.GaussianBlur(gray, (0, 0), 5.0)
    
    # ========================================================================
    # GRADIENT FEATURES (2)
    # ========================================================================
    
    features[..., 10] = gradient_mag
    
    # Gradient orientation (normalized to [0, 1])
    gradient_orient = np.arctan2(sobel_y, sobel_x)
    gradient_orient = (gradient_orient + np.pi) / (2 * np.pi)
    features[..., 11] = gradient_orient
    
    # ========================================================================
    # TEXTURE FEATURES (3)
    # ========================================================================
    
    # Local variance (vectorized)
    features[..., 12] = compute_local_variance_fast(gray, window_size=7)
    
    # Local entropy (Numba-accelerated)
    features[..., 13] = compute_local_entropy_fast(gray, window_size=7)
    
    # Local Binary Pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    features[..., 14] = lbp.astype(np.float32)
    
    # ========================================================================
    # SPECTRAL INDICES (2)
    # ========================================================================
    
    # NDVI approximation [-1, 1] -> [0, 1]
    ndvi = compute_ndvi_fast(img_down)
    features[..., 15] = (ndvi + 1.0) / 2.0
    
    # Water index [0, 1]
    features[..., 16] = compute_water_index_fast(img_down)
    
    # ========================================================================
    # GEOMETRIC FEATURES (2)
    # ========================================================================
    
    # Gradient anisotropy (optimized)
    features[..., 17] = compute_gradient_anisotropy_fast(gray, window_size=7)
    
    # Corner density (optimized)
    features[..., 18] = compute_corner_density_fast(gray, window_size=9)
    
    # ========================================================================
    # NORMALIZATION (optional)
    # ========================================================================
    
    if normalize:
        log.debug("Normalizing features to [0, 1]")
        for i in range(19):
            features[..., i] = normalize_to_01(features[..., i])
    
    # ========================================================================
    # SAVE (optional)
    # ========================================================================
    
    if save and save_path is not None:
        # log.info(f"Saving features to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
    
    return features


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def _process_single_image(row: dict, 
                          normalize: bool= True, 
                          downsample_fraction: float= ProcessingConfig.DOWNSAMPLE_FRACTION) -> tuple:
    """
    Worker function for processing a single image.
    Must be at module level for multiprocessing pickling.
    
    Args:
        row: Dictionary with 'img_path' and 'feature_path'
        normalize: Whether to normalize features
        downsample_fraction: Downsampling fraction for feature extraction
        
    Returns:
        (success: bool, error_message: str or None)
    """
    img_path = row['img_path']
    feature_path = row['feature_path']
    
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return False, f"Failed to read image: {img_path}"
        
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Extract features with downsampling
        features = extract_features(
            img,
            normalize=normalize,
            save=True,
            save_path=feature_path,
            downsample_fraction=downsample_fraction
        )
        
        return True, None
        
    except Exception as e:
        return False, f"Error processing {img_path}: {str(e)}"


def extract_features_batch(
    mapping_csv_path: str,
    feature_dir: str = DataPath.FEATURE_TRAIN,
    mask_dir: str = DataPath.MASK_TRAIN,
    num_workers: int = GeneralConfig.NB_JOBS,
    normalize: bool = True,
    downsample_fraction: float = ProcessingConfig.DOWNSAMPLE_FRACTION
) -> None:
    """
    Extract features for multiple images using multiprocessing.

    Reads a CSV file with columns:
    img_id, img_path, label_path

    Each image is processed independently and features are saved as .npy files.
    Images causing errors are skipped.
    """
    # Read CSV mapping
    log.info(f"Reading CSV mapping from {mapping_csv_path}")
    df = pd.read_csv(mapping_csv_path)
    total_images = len(df)

    log.info(f"Found {total_images} images to process")

    if total_images == 0:
        log.warning("No images found in CSV, exiting.")
        return

    # Convert DataFrame to list of dicts (picklable)
    rows = df.to_dict(orient="records")
    # add feature_path to each row
    for row in rows:
        img_id = row["img_id"]
        row["feature_path"] = os.path.join(feature_dir, f"{img_id}.npy")


    # Prepare worker function (picklable)
    worker_fn = partial(
        _process_single_image,
        normalize=normalize,
        downsample_fraction=downsample_fraction
    )

    # log.info(f"Starting feature extraction with {num_workers} workers")

    success_count = 0
    error_count = 0

    # Multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker_fn, rows),
                total=total_images,
                desc="Extracting features"
            )
        )

    # Collect results
    for success, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            if error_msg:
                log.warning(error_msg)

    # Final summary
    log.info("=" * 60)
    log.info("Feature extraction completed")
    log.info(f"Total images: {total_images}")
    log.info(f"Successfully processed: {success_count}")
    log.info(f"Failed / Skipped: {error_count}")
    log.info(f"Success rate: {100.0 * success_count / total_images:.2f}%")
    log.info("=" * 60)




if __name__ == "__main__":
    # Example usage
    extract_features_batch(
        mapping_csv_path=DataPath.CSV_MAPPING_TRAIN,
        num_workers=GeneralConfig.NB_JOBS,
        normalize=True,
        downsample_fraction=ProcessingConfig.DOWNSAMPLE_FRACTION
    )