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
    MODEL_DIR: str = r"data/models/"

    FEATURE_DIR: str = r"data/features/"
    FEATURE_TEST: str = r"data/features/test/"
    FEATURE_TRAIN: str = r"data/features/train/"
    FEATURE_VAL: str = r"data/features/val/"
    MASK_TRAIN: str = r"data/masks/train/"
    MASK_VAL: str = r"data/masks/val/"
    MASK_TEST: str = r"data/masks/test/"

    CSV_MAPPING_TRAIN: str = r"data/metadata/train_mapping.csv"
    CSV_MAPPING_VAL: str = r"data/metadata/val_mapping.csv"
    CSV_MAPPING_TEST: str = r"data/metadata/test_mapping.csv"

    CSV_CLASS_STATISTICS_TRAIN: str = r"data/metadata/train_class_statistics.csv"
    CSV_CLASS_STATISTICS_VAL: str = r"data/metadata/val_class_statistics.csv"
    CSV_CLASS_STATISTICS_TEST: str = r"data/metadata/test_class_statistics.csv"

    CSV_SELECTED_IMAGES_TRAIN: str = r"data/metadata/selected_train_images.csv"
    CSV_SELECTED_IMAGES_VAL: str = r"data/metadata/selected_val_images.csv"
    CSV_SELECTED_IMAGES_TEST: str = r"data/metadata/selected_test_images.csv"

    

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

    PLOT_PATH : str = r"documents/plot/"

class CSVKeys:
    """Keys for CSV mapping files."""
    IMAGE_PATH: str = "img_path"
    LABEL_PATH: str = "label_path"
    FEATURE_PATH: str = "feature_path"


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class ClassInfo:
    """Semantic class defi  nitions and metadata."""
    
    # Mapping from class ID to class name /!\
    CLASS_NAMES: Dict[int, str] = {
        0: "Field",
        1: "Building",
        2: "Woodland",
        3: "Water",
        4: "Road",
    }
    NUM_CLASSES: int = len(CLASS_NAMES)
    
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



class FeatureInfo:
    """
    Centralized definition of feature indices for the optimized feature pipeline.

    Each attribute maps a semantic feature name to its corresponding index
    in the feature tensor of shape (H, W, 19).
    For more information, check documentation
    """

    # Color (RGB)
    RED: int = 0          # Direct from downsampled RGB
    GREEN: int = 1        # Direct from downsampled RGB
    BLUE: int = 2         # Direct from downsampled RGB

    # Color (HSV)
    HUE: int = 3          # From cached HSV
    SATURATION: int = 4  # From cached HSV
    VALUE: int = 5       # From cached HSV

    # Intensity
    GRAYSCALE: int = 6   # From cached grayscale

    # Multi-scale Gaussian blur
    BLUR_SIGMA_1: int = 7    # Gaussian blur σ=1.0
    BLUR_SIGMA_2_5: int = 8  # Gaussian blur σ=2.5
    BLUR_SIGMA_5: int = 9    # Gaussian blur σ=5.0

    # Gradient
    GRADIENT_MAG: int = 10       # From cached gradients
    GRADIENT_ORIENT: int = 11    # Orientation from cached gradients

    # Texture
    LOCAL_VARIANCE: int = 12     # Vectorized with box filters
    LOCAL_ENTROPY: int = 13      # Numba-accelerated
    LBP: int = 14                # Local Binary Pattern (skimage)

    # Spectral indices
    NDVI: int = 15               # Vectorized
    WATER_INDEX: int = 16        # Vectorized

    # Geometric features
    ANISOTROPY: int = 17         # Vectorized with box filters
    CORNER_DENSITY: int = 18     # Vectorized with box filters

    # Total number of features
    NUM_FEATURES: int = 19

    # Extracted feature indices

    FEATURE_NAMES: Dict[int, str] = {
        0: "Red",
        1: "Green",
        2: "Blue",
        3: "Hue",
        4: "Saturation",
        5: "Value",
        6: "Grayscale",
        7: "Blur_Sigma_1",
        8: "Blur_Sigma_2.5",
        9: "Blur_Sigma_5",
        10: "Gradient_Magnitude",
        11: "Gradient_Orientation",
        12: "Local_Variance",
        13: "Local_Entropy",
        14: "LBP",
        15: "NDVI",
        16: "Water_Index",
        17: "Anisotropy",
        18: "Corner_Density",
        19: "Num_Features",
    }

    FEATURE_UNET_SELECTION: List[int] = [
        RED, GREEN, BLUE,     # Color
        NDVI,                 # Vegetation indicator
        CORNER_DENSITY,       # Geometric structures (Building/Road)
        GRADIENT_MAG,         # Edges / contours
        LOCAL_ENTROPY,        # Texture
        ANISOTROPY            # Linear structures, orientation
    ]

    # Subset of features for K-Means (larger, multi-dimensional)
    FEATURE_KMEANS_SELECTION: List[int] = [
        RED, GREEN, BLUE,     # Color
        HUE, SATURATION, VALUE,  # HSV channels for better color separation
        GRAYSCALE,            # Intensity
        BLUR_SIGMA_1, BLUR_SIGMA_2_5, BLUR_SIGMA_5,  # Multi-scale smoothing
        GRADIENT_MAG, GRADIENT_ORIENT,               # Edges
        LOCAL_VARIANCE, LOCAL_ENTROPY, LBP,          # Texture features
        NDVI, WATER_INDEX,                            # Spectral indices
        ANISOTROPY, CORNER_DENSITY                   # Geometric features
    ]
        
    

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

    #PCA Subsample size
    PCA_SUBSAMPLE_SIZE: int = 200000

    # Downsampling faction for feature extraction
    DOWNSAMPLE_FRACTION: float = 0.5


class ImgSelectionRule:
    """Rules for selecting images based on class proportions."""

    BUILDING_AND_ROAD = {
        "__logic__": "AND",
        1: ">=1",  # At least 1 building
        4: ">=1"   # At least 1 road
}
    BUILDING_OR_ROAD_OR_WATER = {
        "__logic__": "OR",
        1: ">=1",  # At least 1 building
        4: ">=1",  # At least 1 road
        3: ">=1"   # At least 1 water
}
    NO_FIELD = {
        "__logic__": "AND",
        0: "0"     # No field pixels
}
    
