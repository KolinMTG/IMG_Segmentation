"""
Input/Output utilities for image loading and saving.
"""

import numpy as np
from PIL import Image
from typing import Optional
import os


def load_image(img_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load RGB image from file path.
    
    Args:
        img_path: Path to image file
        normalize: If True, normalize to [0, 1], else keep [0, 255]
        
    Returns:
        RGB image as numpy array
        - If normalize=True: shape (H, W, 3), dtype float32, range [0, 1]
        - If normalize=False: shape (H, W, 3), dtype uint8, range [0, 255]
        
    Raises:
        FileNotFoundError: If image path does not exist
        ValueError: If image cannot be loaded or converted to RGB
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {img_path}: {e}")
    
    img_array = np.array(img)
    
    if normalize:
        return img_array.astype(np.float32) / 255.0
    else:
        return img_array


def save_mask(
    mask: np.ndarray,
    save_path: str,
    as_uint8: bool = True
) -> None:
    """
    Save mask to disk as PNG.
    
    Args:
        mask: 2D array with values in [0, 1] or [0, 255]
        save_path: Output file path
        as_uint8: If True, convert [0, 1] to [0, 255] uint8
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if as_uint8 and mask.dtype != np.uint8:
        mask_to_save = (mask * 255).astype(np.uint8)
    else:
        mask_to_save = mask
    
    img = Image.fromarray(mask_to_save)
    img.save(save_path)


def save_colored_mask(
    mask: np.ndarray,
    save_path: str,
    color_map: dict
) -> None:
    """
    Save class mask as colored RGB image.
    
    Args:
        mask: 2D integer array with class IDs
        save_path: Output file path
        color_map: Dictionary mapping class_id -> [R, G, B]
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        colored[mask == class_id] = color
    
    img = Image.fromarray(colored)
    img.save(save_path)
