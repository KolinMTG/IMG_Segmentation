"""
Constants and configuration for satellite image segmentation pipeline.
"""

from typing import Dict, List

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
class GeneralConfig:
    """General project configuration."""
    RANDOM_SEED: int = 42
    NB_JOBS: int = 4  # Number of parallel jobs for processing


class GeneralPath:
    """General project paths."""
    LOG_PATH: str = r".logs/"


class DataPath:
    """Data directory paths."""
    IMG_TRAIN: str = r"data/images/train/"
    IMG_VAL: str = r"data/images/val/"
    IMG_TEST: str = r"data/images/test/"
    LABEL_TRAIN: str = r"data/labels/train/"
    LABEL_VAL: str = r"data/labels/val/"
    LABEL_TEST: str = r"data/labels/test/"
    REPORT_PATH:str = r"data/reports/"
    RESULT_PATH:str = r"data/results/"

class ResultPath:
    """Result output paths."""
    PREDICTION_PATH: str = r"data/results/predictions/"
    TEST_PATH: str = r"data/results/tests/"
    EVALUATION_CSV_PATH:str = r"data/results/predictions_metrics.csv"

class TestPath:
    """Paths for test images and labels."""
    IMG_TEST: str = r"data/images/test/M-33-7-A-d-2-3_19.jpg"
    LABEL_TEST: str = r"data/labels/test/M-33-7-A-d-2-3_19.png"
    IMG_TEST_LIST: List[str] = [r"data/images/test/M-33-7-A-d-2-3_19.jpg",
                                r"data/images/test/M-33-7-A-d-2-3_27.jpg",
                                r"data/images/test/M-33-7-A-d-2-3_29.jpg",
                                r"data/images/test/M-33-7-A-d-2-3_30.jpg",
                                r"data/images/test/M-33-7-A-d-2-3_40.jpg"]
    LABEL_TEST_LIST: List[str] = [r"data/labels/test/M-33-7-A-d-2-3_19.png",
                                  r"data/labels/test/M-33-7-A-d-2-3_27.png",
                                  r"data/labels/test/M-33-7-A-d-2-3_29.png",
                                  r"data/labels/test/M-33-7-A-d-2-3_30.png",
                                  r"data/labels/test/M-33-7-A-d-2-3_40.png"]



# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class ClassInfo:
    """Semantic class definitions and metadata."""
    
    # Mapping from class ID to class name /!\
    CLASS_NAMES: Dict[int, str] = {
        0: "Field",
        1: "Building",
        2: "Woodland",
        3: "Water",
        4: "Road",
    }
    
    # Mapping from class ID to RGB color for visualization
    CLASS_COLORS: Dict[int, list] = {
        0: [0, 0, 0],        # Black
        1: [255, 0, 0],      # Red
        2: [0, 255, 0],      # Green
        3: [0, 0, 255],      # Blue
        4: [128, 128, 128],  # Gray
    }
    
    # Priority weights for detection (higher = more important to detect)
    CLASS_PRIORITY: Dict[int, str] = {
        0: "medium",   # Field
        1: "high",     # Building - critical to detect
        2: "medium",   # Woodland
        3: "medium",   # Water
        4: "high",     # Road - critical to detect
    }


# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

class ProcessingConfig:
    """Default parameters for image processing operations."""

    #Background class (most common class in images)
    BACKGROUND_CLASS: int = 0  # Field
    
    # Gaussian smoothing
    GAUSSIAN_SIGMA_LIGHT: float = 1.0
    GAUSSIAN_SIGMA_MEDIUM: float = 1.5
    GAUSSIAN_SIGMA_HEAVY: float = 2.0
    
    # CLAHE parameters
    CLAHE_CLIP_LIMIT_LOW: float = 2.0
    CLAHE_CLIP_LIMIT_MEDIUM: float = 3.0
    CLAHE_CLIP_LIMIT_HIGH: float = 3.5
    CLAHE_TILE_SIZE: int = 8
    
    # Edge detection
    CANNY_LOW_THRESHOLD: int = 30
    CANNY_HIGH_THRESHOLD: int = 100
    SOBEL_KERNEL_SIZE: int = 3
    
    # Morphological operations
    MORPH_KERNEL_SIZE_SMALL: int = 3
    MORPH_KERNEL_SIZE_MEDIUM: int = 5
    MORPH_KERNEL_SIZE_LARGE: int = 7
    
    # Local window sizes
    WINDOW_SIZE_SMALL: int = 5
    WINDOW_SIZE_MEDIUM: int = 7
    WINDOW_SIZE_LARGE: int = 9
    WINDOW_SIZE_XLARGE: int = 11
    
    # Texture analysis
    LBP_RADIUS: int = 2
    LBP_POINTS_MULTIPLIER: int = 8
    
    # Channel emphasis multipliers
    GREEN_EMPHASIS_FACTOR: float = 1.4
    BLUE_EMPHASIS_FACTOR: float = 1.5
    
    # Variance scaling factors
    VARIANCE_SCALE_FIELD: float = 100.0
    VARIANCE_SCALE_WATER: float = 150.0

    # Post-processing parameters
    OPENING_RADIUS: int = 2
    CLOSING_RADIUS: int = 3
    MIN_AREA: int = 50  # Minimum area to keep a detected region
    CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence to accept detection
