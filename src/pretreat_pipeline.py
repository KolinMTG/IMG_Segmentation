"""
Main segmentation pipeline for satellite image processing.
"""

import numpy as np
from typing import Dict, List
import sys
import os

# Add classes directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classes'))

from classes.field import process_field
from classes.building import process_building
from classes.woodland import process_woodland
from classes.water import process_water
from classes.road import process_road
from cste import ClassInfo


# Map class IDs to processing functions
CLASS_PROCESSORS = {
    0: process_field,
    1: process_building,
    2: process_woodland,
    3: process_water,
    4: process_road,
}


def segmentation_pipeline(
    img: np.ndarray,
    class_ids: List[int]
) -> Dict[int, np.ndarray]:
    """
    Execute segmentation pipeline for specified classes.
    
    This function processes an RGB satellite image and generates independent
    confidence score maps for each requested semantic class. Each class is
    processed by its specialized detection algorithm.
    
    Args:
        img: RGB image as numpy array
             - Shape: (H, W, 3)
             - Dtype: float32
             - Value range: [0, 1]
        class_ids: List of class IDs to process (0-4)
        
    Returns:
        Dictionary mapping class_id -> score_map
        - Each score_map is (H, W) float32 array in [0, 1]
        - Higher values indicate higher confidence
        
    Raises:
        ValueError: If invalid class ID provided
        TypeError: If img is not numpy array or has wrong shape
        
    Example:
        >>> img = load_image("satellite.jpg", normalize=True)
        >>> masks = segmentation_pipeline(img, class_ids=[0, 1, 4])
        >>> field_mask = masks[0]
        >>> building_mask = masks[1]
        >>> road_mask = masks[4]
    """
    # Input validation
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be numpy array")
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"img must have shape (H, W, 3), got {img.shape}")
    
    if img.dtype != np.float32:
        raise TypeError(f"img must be float32, got {img.dtype}")
    
    # Validate class IDs
    valid_classes = set(CLASS_PROCESSORS.keys())
    invalid_classes = set(class_ids) - valid_classes
    
    if invalid_classes:
        raise ValueError(
            f"Invalid class IDs: {invalid_classes}. "
            f"Valid IDs are: {sorted(valid_classes)}"
        )
    
    # Process each requested class
    result = {}
    
    for class_id in class_ids:
        processor = CLASS_PROCESSORS[class_id]
        score_map = processor(img)
        result[class_id] = score_map
    
    return result


def process_all_classes(img: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Process image for all available classes.
    
    Convenience wrapper that processes all 5 classes.
    
    Args:
        img: RGB image in [0, 1]
        
    Returns:
        Dictionary with all class score maps
    """
    return segmentation_pipeline(img, class_ids=list(CLASS_PROCESSORS.keys()))



def get_class_name(class_id: int) -> str:
    """get human-readable class name."""
    return ClassInfo.CLASS_NAMES.get(class_id, f"Unknown({class_id})")


def get_class_color(class_id: int) -> list:
    """get RGB color for visualization."""
    return ClassInfo.CLASS_COLORS.get(class_id, [128, 128, 128])
