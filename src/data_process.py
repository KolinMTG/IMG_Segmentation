import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
from skimage import filters, exposure, feature
from skimage.filters import gaussian, sobel
from skimage.feature import local_binary_pattern

from src.cste import *
from src.logger import get_logger
from src.data_info import show_img_labels

log = get_logger("data_process.log")

def preprocess_image_by_class(img_path: str, class_id: int, display: bool = False) -> np.ndarray:
    """
    Apply class-specific image preprocessing for satellite images.
    
    Args:
        img_path: Path to the RGB image
        class_id: Integer representing the class (0=Field, 1=Building, 2=Woodland, 3=Water, 4=Road)
        display: If True, display the image at all preprocessing steps
        
    Returns:
        Preprocessed image as numpy array (float32, normalized [0,1])
    """
    # Load image and convert to RGB
    log.info(f"Loading image from: {img_path}")
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    log.info(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
    log.info(f"Applying preprocessing for class {class_id}")
    
    # Store intermediate results for display
    steps = []
    step_names = []
    
    # Add original image
    steps.append(img_array.copy())
    step_names.append("Original Image")
    
    # Apply class-specific preprocessing
    if class_id == 0:  # Field
        log.info("Class 0 (Field): Applying light smoothing and local histogram normalization")
        
        # Light smoothing using Gaussian filter
        log.info("  - Applying Gaussian smoothing (sigma=1.0)")
        smoothed = gaussian(img_array, sigma=1.0, channel_axis=2, preserve_range=True)
        steps.append(smoothed.copy())
        step_names.append("Gaussian Smoothing")
        
        # Local histogram normalization using CLAHE on each channel
        log.info("  - Applying CLAHE for local histogram normalization")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = np.zeros_like(smoothed)
        for i in range(3):
            # Convert to uint8 for CLAHE
            channel = (smoothed[:, :, i] * 255).astype(np.uint8)
            processed[:, :, i] = clahe.apply(channel) / 255.0
        
        steps.append(processed.copy())
        step_names.append("CLAHE Normalization")
        result = processed
        
    elif class_id == 1:  # Building
        log.info("Class 1 (Building): Applying edge enhancement and local contrast enhancement")
        
        # Edge enhancement using Sobel filter
        log.info("  - Applying Sobel edge detection")
        edges = np.zeros_like(img_array)
        for i in range(3):
            edges[:, :, i] = sobel(img_array[:, :, i])
        
        # Normalize edges to [0, 1]
        edges = np.clip(edges, 0, 1)
        steps.append(edges.copy())
        step_names.append("Sobel Edges")
        
        # Blend original with edges for enhancement
        log.info("  - Blending original image with edges (alpha=0.3)")
        edge_enhanced = np.clip(img_array + 0.3 * edges, 0, 1)
        steps.append(edge_enhanced.copy())
        step_names.append("Edge Enhanced")
        
        # Local contrast enhancement using CLAHE
        log.info("  - Applying CLAHE for local contrast enhancement")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = np.zeros_like(edge_enhanced)
        for i in range(3):
            channel = (edge_enhanced[:, :, i] * 255).astype(np.uint8)
            processed[:, :, i] = clahe.apply(channel) / 255.0
        
        steps.append(processed.copy())
        step_names.append("CLAHE Enhancement")
        result = processed
        
    elif class_id == 2:  # Woodland
        log.info("Class 2 (Woodland): Emphasizing green channel, sharpening, and texture filtering")
        
        # Emphasize green channel
        log.info("  - Emphasizing green channel (multiplier=1.3)")
        emphasized = img_array.copy()
        emphasized[:, :, 1] = np.clip(emphasized[:, :, 1] * 1.3, 0, 1)
        steps.append(emphasized.copy())
        step_names.append("Green Channel Emphasized")
        
        # Sharpening using unsharp mask
        log.info("  - Applying sharpening with unsharp mask")
        blurred = gaussian(emphasized, sigma=1.5, channel_axis=2, preserve_range=True)
        sharpened = np.clip(emphasized + 0.5 * (emphasized - blurred), 0, 1)
        steps.append(sharpened.copy())
        step_names.append("Sharpened")
        
        # Texture filtering using Local Binary Pattern on grayscale
        log.info("  - Computing Local Binary Pattern for texture")
        gray = np.dot(sharpened, [0.299, 0.587, 0.114])
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Normalize LBP to [0, 1]
        lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
        
        # Create texture-enhanced image by blending
        texture_enhanced = sharpened.copy()
        for i in range(3):
            texture_enhanced[:, :, i] = np.clip(
                sharpened[:, :, i] + 0.1 * lbp_normalized, 0, 1
            )
        
        steps.append(np.stack([lbp_normalized] * 3, axis=2))
        step_names.append("LBP Texture")
        steps.append(texture_enhanced.copy())
        step_names.append("Texture Enhanced")
        result = texture_enhanced
        
    elif class_id == 3:  # Water
        log.info("Class 3 (Water): Applying smoothing and emphasizing blue channel")
        
        # Smoothing using Gaussian filter
        log.info("  - Applying Gaussian smoothing (sigma=1.5)")
        smoothed = gaussian(img_array, sigma=1.5, channel_axis=2, preserve_range=True)
        steps.append(smoothed.copy())
        step_names.append("Gaussian Smoothing")
        
        # Emphasize blue channel
        log.info("  - Emphasizing blue channel (multiplier=1.4)")
        emphasized = smoothed.copy()
        emphasized[:, :, 2] = np.clip(emphasized[:, :, 2] * 1.4, 0, 1)
        steps.append(emphasized.copy())
        step_names.append("Blue Channel Emphasized")
        
        result = emphasized
        
    elif class_id == 4:  # Road
        log.info("Class 4 (Road): Applying edge detection, morphological operations, and contrast enhancement")
        
        # Convert to grayscale for edge detection
        log.info("  - Converting to grayscale and applying Canny edge detection")
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        gray_uint8 = (gray * 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, 50, 150) / 255.0
        
        steps.append(np.stack([edges] * 3, axis=2))
        step_names.append("Canny Edges")
        
        # Morphological operations (dilation to connect edges)
        log.info("  - Applying morphological dilation to connect edges")
        kernel = np.ones((3, 3), np.uint8)
        edges_uint8 = (edges * 255).astype(np.uint8)
        dilated = cv2.dilate(edges_uint8, kernel, iterations=1) / 255.0
        
        steps.append(np.stack([dilated] * 3, axis=2))
        step_names.append("Dilated Edges")
        
        # Blend edges with original image
        log.info("  - Blending edges with original image")
        edge_enhanced = img_array.copy()
        for i in range(3):
            edge_enhanced[:, :, i] = np.clip(
                img_array[:, :, i] + 0.4 * dilated, 0, 1
            )
        
        steps.append(edge_enhanced.copy())
        step_names.append("Edge Enhanced")
        
        # Contrast enhancement using CLAHE
        log.info("  - Applying CLAHE for contrast enhancement")
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        processed = np.zeros_like(edge_enhanced)
        for i in range(3):
            channel = (edge_enhanced[:, :, i] * 255).astype(np.uint8)
            processed[:, :, i] = clahe.apply(channel) / 255.0
        
        steps.append(processed.copy())
        step_names.append("CLAHE Enhancement")
        result = processed
        
    else:
        log.info(f"Unknown class_id {class_id}, returning original image")
        result = img_array
    
    # Display intermediate results if requested
    if display:
        log.info("Displaying preprocessing steps")
        n_steps = len(steps)
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        # Flatten axes array for easier indexing
        if n_steps == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        # Display each step
        for idx, (step_img, step_name) in enumerate(zip(steps, step_names)):
            axes[idx].imshow(np.clip(step_img, 0, 1))
            axes[idx].set_title(step_name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_steps, len(axes)):
            axes[idx].axis('off')
        
        class_names = {
            0: "Field", 
            1: "Building", 
            2: "Woodland", 
            3: "Water", 
            4: "Road"
        }
        
        plt.suptitle(
            f"Preprocessing Steps for Class {class_id} ({class_names.get(class_id, 'Unknown')})",
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.show()
    
    log.info("Preprocessing completed successfully")
    
    return result.astype(np.float32)


import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import filters, exposure, feature
from skimage.filters import gaussian, sobel
from skimage.feature import local_binary_pattern
from scipy import ndimage


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_to_01(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: Input image array
        
    Returns:
        Normalized image in [0, 1] range
    """
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def apply_gaussian_smoothing(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to image.
    
    Args:
        img: Input image (can be grayscale or RGB)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed image
    """
    if len(img.shape) == 3:
        return gaussian(img, sigma=sigma, channel_axis=2, preserve_range=True)
    else:
        return gaussian(img, sigma=sigma, preserve_range=True)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        img: Input image (grayscale or RGB)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    
    if len(img.shape) == 3:
        result = np.zeros_like(img)
        for i in range(img.shape[2]):
            channel = (img[:, :, i] * 255).astype(np.uint8)
            result[:, :, i] = clahe.apply(channel) / 255.0
        return result
    else:
        channel = (img * 255).astype(np.uint8)
        return clahe.apply(channel) / 255.0


def compute_sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edge magnitude for grayscale image.
    
    Args:
        img: Input grayscale image
        
    Returns:
        Edge magnitude map
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude


def compute_canny_edges(img: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Compute Canny edges for grayscale image.
    
    Args:
        img: Input grayscale image (0-1 range)
        low_thresh: Lower threshold for Canny
        high_thresh: Upper threshold for Canny
        
    Returns:
        Binary edge map (0-1 range)
    """
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, low_thresh, high_thresh)
    return edges.astype(np.float32) / 255.0


def apply_morphological_closing(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological closing operation.
    
    Args:
        img: Input image
        kernel_size: Size of morphological kernel
        iterations: Number of times to apply operation
        
    Returns:
        Processed image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)
    closed = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed.astype(np.float32) / 255.0


def apply_morphological_dilation(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological dilation operation.
    
    Args:
        img: Input image
        kernel_size: Size of morphological kernel
        iterations: Number of times to apply operation
        
    Returns:
        Dilated image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)
    dilated = cv2.dilate(img_uint8, kernel, iterations=iterations)
    return dilated.astype(np.float32) / 255.0


def compute_local_variance(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local variance for texture analysis.
    
    Args:
        img: Input grayscale image
        window_size: Size of local window
        
    Returns:
        Local variance map
    """
    # Compute local mean
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = ndimage.convolve(img, kernel, mode='reflect')
    
    # Compute local variance
    local_mean_sq = ndimage.convolve(img ** 2, kernel, mode='reflect')
    local_variance = local_mean_sq - local_mean ** 2
    
    return np.maximum(local_variance, 0)  # Ensure non-negative


def display_processing_steps(steps: list, step_names: list, title: str):
    """
    Display all processing steps in a single figure.
    
    Args:
        steps: List of images to display
        step_names: List of names for each step
        title: Overall title for the figure
    """
    n_steps = len(steps)
    cols = min(4, n_steps)
    rows = (n_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Flatten axes for easier indexing
    if n_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Display each step
    for idx, (step_img, step_name) in enumerate(zip(steps, step_names)):
        if len(step_img.shape) == 2:
            axes[idx].imshow(step_img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[idx].imshow(np.clip(step_img, 0, 1))
        axes[idx].set_title(step_name, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


# ============================================================================
# CLASS-SPECIFIC PROCESSING FUNCTIONS
# ============================================================================

def process_field(img: np.ndarray, display: bool = False) -> np.ndarray:
    """
    Process RGB image to generate field likelihood score map.
    
    Strategy:
    - Fields typically have uniform texture with low spatial frequency
    - Apply smoothing to reduce noise
    - Use local histogram normalization to enhance uniform areas
    - Compute smoothness score based on low variance
    
    Args:
        img: RGB image (HxWx3) with values in [0, 1]
        display: If True, display all intermediate steps
        
    Returns:
        Grayscale score map (HxW) in [0, 1] representing field likelihood
    """
    log.info("Processing for Field class (Class 0)")
    
    steps = []
    step_names = []
    
    # Store original image
    steps.append(img.copy())
    step_names.append("Original Image")
    
    # Step 1: Apply light Gaussian smoothing
    log.info("  - Applying Gaussian smoothing (sigma=1.0)")
    smoothed = apply_gaussian_smoothing(img, sigma=1.0)
    steps.append(smoothed.copy())
    step_names.append("Smoothed Image")
    
    # Step 2: Apply CLAHE for local histogram normalization
    log.info("  - Applying CLAHE for local normalization")
    normalized = apply_clahe(smoothed, clip_limit=2.0, tile_size=8)
    steps.append(normalized.copy())
    step_names.append("CLAHE Normalized")
    
    # Step 3: Convert to grayscale
    log.info("  - Converting to grayscale")
    gray = np.dot(normalized, [0.299, 0.587, 0.114])
    steps.append(gray.copy())
    step_names.append("Grayscale")
    
    # Step 4: Compute smoothness score (inverse of local variance)
    log.info("  - Computing smoothness score based on local variance")
    local_var = compute_local_variance(gray, window_size=7)
    
    # Inverse variance: smooth areas have high score
    smoothness_score = 1.0 / (1.0 + local_var * 100)
    steps.append(smoothness_score.copy())
    step_names.append("Smoothness Score")
    
    # Step 5: Apply slight smoothing to score map
    log.info("  - Smoothing score map")
    score_smoothed = apply_gaussian_smoothing(smoothness_score, sigma=2.0)
    steps.append(score_smoothed.copy())
    step_names.append("Smoothed Score")
    
    # Step 6: Normalize final score to [0, 1]
    log.info("  - Normalizing final score map to [0, 1]")
    final_score = normalize_to_01(score_smoothed)
    steps.append(final_score.copy())
    step_names.append("Final Normalized Score")
    
    # Display if requested
    if display:
        log.info("  - Displaying processing steps")
        display_processing_steps(steps, step_names, "Field Processing Pipeline")
    
    log.info("Field processing completed")
    return final_score


def process_building(img: np.ndarray, display: bool = False) -> np.ndarray:
    """
    Process RGB image to generate building likelihood score map.
    
    Strategy:
    - Buildings have strong edges and high local contrast
    - Apply edge enhancement using Sobel
    - Use CLAHE to enhance local contrast
    - Score based on edge strength
    
    Args:
        img: RGB image (HxWx3) with values in [0, 1]
        display: If True, display all intermediate steps
        
    Returns:
        Grayscale score map (HxW) in [0, 1] representing building likelihood
    """
    log.info("Processing for Building class (Class 1)")
    
    steps = []
    step_names = []
    
    # Store original image
    steps.append(img.copy())
    step_names.append("Original Image")
    
    # Step 1: Convert to grayscale
    log.info("  - Converting to grayscale")
    gray = np.dot(img, [0.299, 0.587, 0.114])
    steps.append(gray.copy())
    step_names.append("Grayscale")
    
    # Step 2: Apply CLAHE for contrast enhancement
    log.info("  - Applying CLAHE for contrast enhancement")
    contrast_enhanced = apply_clahe(gray, clip_limit=3.0, tile_size=8)
    steps.append(contrast_enhanced.copy())
    step_names.append("CLAHE Enhanced")
    
    # Step 3: Compute Sobel edge magnitude
    log.info("  - Computing Sobel edge magnitude")
    edges = compute_sobel_magnitude(contrast_enhanced)
    edges_normalized = normalize_to_01(edges)
    steps.append(edges_normalized.copy())
    step_names.append("Sobel Edges")
    
    # Step 4: Apply morphological closing to connect nearby edges
    log.info("  - Applying morphological closing to connect edges")
    edges_closed = apply_morphological_closing(edges_normalized, kernel_size=3, iterations=1)
    steps.append(edges_closed.copy())
    step_names.append("Closed Edges")
    
    # Step 5: Compute local edge density
    log.info("  - Computing local edge density")
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    edge_density = ndimage.convolve(edges_closed, kernel, mode='reflect')
    steps.append(edge_density.copy())
    step_names.append("Edge Density")
    
    # Step 6: Combine edge strength with local contrast
    log.info("  - Computing local standard deviation for contrast")
    local_std = ndimage.generic_filter(contrast_enhanced, np.std, size=7)
    local_std_normalized = normalize_to_01(local_std)
    steps.append(local_std_normalized.copy())
    step_names.append("Local Std Dev")
    
    # Step 7: Combine scores (edge density + local contrast)
    log.info("  - Combining edge density and contrast scores")
    combined_score = 0.6 * edge_density + 0.4 * local_std_normalized
    steps.append(combined_score.copy())
    step_names.append("Combined Score")
    
    # Step 8: Normalize final score to [0, 1]
    log.info("  - Normalizing final score map to [0, 1]")
    final_score = normalize_to_01(combined_score)
    steps.append(final_score.copy())
    step_names.append("Final Normalized Score")
    
    # Display if requested
    if display:
        log.info("  - Displaying processing steps")
        display_processing_steps(steps, step_names, "Building Processing Pipeline")
    
    log.info("Building processing completed")
    return final_score


def process_woodland(img: np.ndarray, display: bool = False) -> np.ndarray:
    """
    Process RGB image to generate woodland likelihood score map.
    
    Strategy:
    - Woodlands have high green channel intensity and high texture
    - Emphasize green channel
    - Apply sharpening to enhance texture
    - Use texture measures (local variance or LBP) for scoring
    
    Args:
        img: RGB image (HxWx3) with values in [0, 1]
        display: If True, display all intermediate steps
        
    Returns:
        Grayscale score map (HxW) in [0, 1] representing woodland likelihood
    """
    log.info("Processing for Woodland class (Class 2)")
    
    steps = []
    step_names = []
    
    # Store original image
    steps.append(img.copy())
    step_names.append("Original Image")
    
    # Step 1: Emphasize green channel
    log.info("  - Emphasizing green channel (multiplier=1.4)")
    green_emphasized = img.copy()
    green_emphasized[:, :, 1] = np.clip(green_emphasized[:, :, 1] * 1.4, 0, 1)
    steps.append(green_emphasized.copy())
    step_names.append("Green Emphasized")
    
    # Step 2: Apply sharpening using unsharp mask
    log.info("  - Applying unsharp mask for sharpening")
    blurred = apply_gaussian_smoothing(green_emphasized, sigma=1.5)
    sharpened = np.clip(green_emphasized + 0.6 * (green_emphasized - blurred), 0, 1)
    steps.append(sharpened.copy())
    step_names.append("Sharpened Image")
    
    # Step 3: Extract green channel intensity
    log.info("  - Extracting green channel intensity")
    green_channel = sharpened[:, :, 1]
    steps.append(green_channel.copy())
    step_names.append("Green Channel")
    
    # Step 4: Compute local variance for texture measure
    log.info("  - Computing local variance for texture analysis")
    texture_variance = compute_local_variance(green_channel, window_size=7)
    texture_score = normalize_to_01(texture_variance)
    steps.append(texture_score.copy())
    step_names.append("Texture Variance")
    
    # Step 5: Compute Local Binary Pattern for additional texture
    log.info("  - Computing Local Binary Pattern (LBP)")
    gray = np.dot(sharpened, [0.299, 0.587, 0.114])
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_normalized = normalize_to_01(lbp)
    steps.append(lbp_normalized.copy())
    step_names.append("LBP Texture")
    
    # Step 6: Combine green intensity with texture measures
    log.info("  - Combining green intensity, variance, and LBP scores")
    # Higher green + higher texture = woodland
    combined_score = 0.4 * green_channel + 0.3 * texture_score + 0.3 * lbp_normalized
    steps.append(combined_score.copy())
    step_names.append("Combined Score")
    
    # Step 7: Apply smoothing to score map
    log.info("  - Smoothing combined score map")
    score_smoothed = apply_gaussian_smoothing(combined_score, sigma=1.5)
    steps.append(score_smoothed.copy())
    step_names.append("Smoothed Score")
    
    # Step 8: Normalize final score to [0, 1]
    log.info("  - Normalizing final score map to [0, 1]")
    final_score = normalize_to_01(score_smoothed)
    steps.append(final_score.copy())
    step_names.append("Final Normalized Score")
    
    # Display if requested
    if display:
        log.info("  - Displaying processing steps")
        display_processing_steps(steps, step_names, "Woodland Processing Pipeline")
    
    log.info("Woodland processing completed")
    return final_score


def process_water(img: np.ndarray, display: bool = False) -> np.ndarray:
    """
    Process RGB image to generate water likelihood score map.
    
    Strategy:
    - Water bodies have high blue channel intensity and are smooth
    - Apply smoothing to reduce noise
    - Emphasize blue channel
    - Score based on blue intensity and smoothness
    
    Args:
        img: RGB image (HxWx3) with values in [0, 1]
        display: If True, display all intermediate steps
        
    Returns:
        Grayscale score map (HxW) in [0, 1] representing water likelihood
    """
    log.info("Processing for Water class (Class 3)")
    
    steps = []
    step_names = []
    
    # Store original image
    steps.append(img.copy())
    step_names.append("Original Image")
    
    # Step 1: Apply Gaussian smoothing
    log.info("  - Applying Gaussian smoothing (sigma=2.0)")
    smoothed = apply_gaussian_smoothing(img, sigma=2.0)
    steps.append(smoothed.copy())
    step_names.append("Smoothed Image")
    
    # Step 2: Emphasize blue channel
    log.info("  - Emphasizing blue channel (multiplier=1.5)")
    blue_emphasized = smoothed.copy()
    blue_emphasized[:, :, 2] = np.clip(blue_emphasized[:, :, 2] * 1.5, 0, 1)
    steps.append(blue_emphasized.copy())
    step_names.append("Blue Emphasized")
    
    # Step 3: Extract blue channel intensity
    log.info("  - Extracting blue channel intensity")
    blue_channel = blue_emphasized[:, :, 2]
    steps.append(blue_channel.copy())
    step_names.append("Blue Channel")
    
    # Step 4: Compute smoothness score (low variance = smooth = water)
    log.info("  - Computing smoothness score")
    gray = np.dot(blue_emphasized, [0.299, 0.587, 0.114])
    local_var = compute_local_variance(gray, window_size=9)
    smoothness_score = 1.0 / (1.0 + local_var * 150)
    steps.append(smoothness_score.copy())
    step_names.append("Smoothness Score")
    
    # Step 5: Compute low saturation regions (water is often desaturated)
    log.info("  - Computing saturation (water has low saturation)")
    # Convert RGB to HSV to get saturation
    img_uint8 = (blue_emphasized * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    low_saturation_score = 1.0 - saturation  # Inverse: low saturation = high score
    steps.append(low_saturation_score.copy())
    step_names.append("Low Saturation Score")
    
    # Step 6: Combine blue intensity, smoothness, and low saturation
    log.info("  - Combining blue intensity, smoothness, and saturation scores")
    combined_score = 0.5 * blue_channel + 0.3 * smoothness_score + 0.2 * low_saturation_score
    steps.append(combined_score.copy())
    step_names.append("Combined Score")
    
    # Step 7: Apply smoothing to final score
    log.info("  - Smoothing final score map")
    score_smoothed = apply_gaussian_smoothing(combined_score, sigma=2.0)
    steps.append(score_smoothed.copy())
    step_names.append("Smoothed Score")
    
    # Step 8: Normalize final score to [0, 1]
    log.info("  - Normalizing final score map to [0, 1]")
    final_score = normalize_to_01(score_smoothed)
    steps.append(final_score.copy())
    step_names.append("Final Normalized Score")
    
    # Display if requested
    if display:
        log.info("  - Displaying processing steps")
        display_processing_steps(steps, step_names, "Water Processing Pipeline")
    
    log.info("Water processing completed")
    return final_score


def process_road(img: np.ndarray, display: bool = False) -> np.ndarray:
    """
    Process RGB image to generate road likelihood score map.
    
    Strategy:
    - Roads have strong linear edges and uniform texture within the road
    - Apply edge detection (Canny)
    - Use morphological operations to enhance linear structures
    - Score based on edge density and contrast
    
    Args:
        img: RGB image (HxWx3) with values in [0, 1]
        display: If True, display all intermediate steps
        
    Returns:
        Grayscale score map (HxW) in [0, 1] representing road likelihood
    """
    log.info("Processing for Road class (Class 4)")
    
    steps = []
    step_names = []
    
    # Store original image
    steps.append(img.copy())
    step_names.append("Original Image")
    
    # Step 1: Convert to grayscale
    log.info("  - Converting to grayscale")
    gray = np.dot(img, [0.299, 0.587, 0.114])
    steps.append(gray.copy())
    step_names.append("Grayscale")
    
    # Step 2: Apply CLAHE for contrast enhancement
    log.info("  - Applying CLAHE for contrast enhancement")
    contrast_enhanced = apply_clahe(gray, clip_limit=3.5, tile_size=8)
    steps.append(contrast_enhanced.copy())
    step_names.append("CLAHE Enhanced")
    
    # Step 3: Compute Canny edges
    log.info("  - Computing Canny edges")
    edges = compute_canny_edges(contrast_enhanced, low_thresh=30, high_thresh=100)
    steps.append(edges.copy())
    step_names.append("Canny Edges")
    
    # Step 4: Apply morphological dilation to strengthen edges
    log.info("  - Applying morphological dilation")
    edges_dilated = apply_morphological_dilation(edges, kernel_size=3, iterations=2)
    steps.append(edges_dilated.copy())
    step_names.append("Dilated Edges")
    
    # Step 5: Apply morphological closing to connect linear structures
    log.info("  - Applying morphological closing to connect structures")
    edges_closed = apply_morphological_closing(edges_dilated, kernel_size=5, iterations=1)
    steps.append(edges_closed.copy())
    step_names.append("Closed Edges")
    
    # Step 6: Compute local edge density
    log.info("  - Computing local edge density")
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    edge_density = ndimage.convolve(edges_closed, kernel, mode='reflect')
    steps.append(edge_density.copy())
    step_names.append("Edge Density")
    
    # Step 7: Compute Sobel magnitude for additional edge information
    log.info("  - Computing Sobel magnitude")
    sobel_mag = compute_sobel_magnitude(contrast_enhanced)
    sobel_normalized = normalize_to_01(sobel_mag)
    steps.append(sobel_normalized.copy())
    step_names.append("Sobel Magnitude")
    
    # Step 8: Combine edge density with Sobel magnitude
    log.info("  - Combining edge density and Sobel magnitude")
    combined_score = 0.6 * edge_density + 0.4 * sobel_normalized
    steps.append(combined_score.copy())
    step_names.append("Combined Score")
    
    # Step 9: Apply smoothing to final score
    log.info("  - Smoothing final score map")
    score_smoothed = apply_gaussian_smoothing(combined_score, sigma=1.5)
    steps.append(score_smoothed.copy())
    step_names.append("Smoothed Score")
    
    # Step 10: Normalize final score to [0, 1]
    log.info("  - Normalizing final score map to [0, 1]")
    final_score = normalize_to_01(score_smoothed)
    steps.append(final_score.copy())
    step_names.append("Final Normalized Score")
    
    # Display if requested
    if display:
        log.info("  - Displaying processing steps")
        display_processing_steps(steps, step_names, "Road Processing Pipeline")
    
    log.info("Road processing completed")
    return final_score


import numpy as np
from PIL import Image

def merge_class_masks(class_masks: dict[int, np.ndarray], save_path: str = None) -> np.ndarray:
    """
    Merge multiple class probability masks into a single mask using majority vote.
    Optionally save the resulting mask as PNG.

    Args:
        class_masks (dict): Dictionary {class_id: np.ndarray}, each array is HxW float [0,1]
        save_path (str, optional): Path to save the resulting mask as PNG. Default None.

    Returns:
        np.ndarray: single-channel integer mask HxW with values 0..N_classes
    """
    if not class_masks:
        raise ValueError("class_masks dictionary is empty")

    # Stack all masks into a 3D array (H, W, num_classes)
    mask_list = []
    class_ids = []
    for class_id, mask in class_masks.items():
        if mask.ndim != 2:
            raise ValueError(f"Mask for class {class_id} is not 2D")
        mask_list.append(mask)
        class_ids.append(class_id)

    stacked_masks = np.stack(mask_list, axis=-1)  # Shape: H x W x num_classes

    # Find the index of the max value along the last axis (vote majoritaire)
    max_indices = np.argmax(stacked_masks, axis=-1)  # H x W, index in mask_list

    # Map indices back to class_ids
    final_mask = np.zeros_like(max_indices, dtype=np.uint8)
    for idx, class_id in enumerate(class_ids):
        final_mask[max_indices == idx] = class_id

    # Save the mask if save_path is provided
    if save_path is not None:
        print("test")
        img_to_save = Image.fromarray(final_mask)
        img_to_save.save(save_path)
        log.info(f"Mask saved to {save_path}")

    return final_mask



# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from cste import DataPath
    import os
    img_name = 'M-34-51-C-d-4-1_191.jpg'
    # Load a sample image
    img_path = os.path.join(DataPath.IMG_TRAIN, img_name)
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    log.info("="*70)
    log.info("Testing all class-specific processing functions")
    log.info("="*70)
    
    # Test each processing function
    log.info("\n" + "="*70)
    field_score = process_field(img_array)
    log.info(f"Field score - min: {field_score.min():.4f}, max: {field_score.max():.4f}, mean: {field_score.mean():.4f}")
    
    log.info("\n" + "="*70)
    building_score = process_building(img_array)
    log.info(f"Building score - min: {building_score.min():.4f}, max: {building_score.max():.4f}, mean: {building_score.mean():.4f}")
    
    log.info("\n" + "="*70)
    woodland_score = process_woodland(img_array)
    log.info(f"Woodland score - min: {woodland_score.min():.4f}, max: {woodland_score.max():.4f}, mean: {woodland_score.mean():.4f}")
    
    log.info("\n" + "="*70)
    water_score = process_water(img_array)
    log.info(f"Water score - min: {water_score.min():.4f}, max: {water_score.max():.4f}, mean: {water_score.mean():.4f}")
    
    log.info("\n" + "="*70)
    road_score = process_road(img_array)
    log.info(f"Road score - min: {road_score.min():.4f}, max: {road_score.max():.4f}, mean: {road_score.mean():.4f}")
    
    # Display all final scores together for comparison
    log.info("\n" + "="*70)
    log.info("Displaying all final score maps for comparison")
    log.info("="*70)
    
    all_scores: dict[int, np.ndarray] = {
        0: field_score,
        1: building_score,
        2: woodland_score,
        3: water_score,
        4: road_score
    }
    
    score_names = [
        "Field Score",
        "Building Score",
        "Woodland Score",
        "Water Score",
        "Road Score"
    ]
    
    # Add original image at the beginning
    # all_scores.insert(0, img_array)
    # score_names.insert(0, "Original Image")
    
    # display_processing_steps(all_scores, score_names, "Comparison of All Class Score Maps")
    
    log.info("\n" + "="*70)
    log.info("All processing completed successfully")
    log.info("="*70)
    predicted_img_path = os.path.join(ResultPath.PREDICTION_PATH, f"{img_name}")
    merge_class_masks(all_scores, save_path=predicted_img_path)
    show_img_labels(img_path, predicted_img_path)