"""
Input/Output utilities for image loading and saving.
"""

import numpy as np
from PIL import Image
from typing import Optional
import os
from typing import List, Dict


def load_image(img_path: str, normalize: bool = True, one_channel: bool = False) -> np.ndarray:
    """
    Load RGB image from file path.
    
    Args:
        img_path: Path to image file
        normalize: If True, normalize to [0, 1], else keep [0, 255]
        one_channel: If True, load as grayscale mono channel
        
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
        if one_channel: # convert to grayscale if requested
            img = img.convert("L")
    except Exception as e:
        raise ValueError(f"Failed to load image {img_path}: {e}")
    
    img_array = np.array(img)

    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.uint8)
    return img_array


def save_mask(
    mask: np.ndarray,
    save_path: str,
    as_uint8: bool = True
) -> None:
    """
    Save a single-channel integer mask to disk as PNG.

    Args:
        mask: 2D array with integer values [0, NUM_CLASSES-1], one pixel per class.
        save_path: Output file path.
        as_uint8: Whether to convert the mask to uint8 before saving.
                  Recommended if NUM_CLASSES <= 255.

    Notes:
        ! The mask values are assumed to be integers representing class IDs.
        ! No normalization or scaling is applied; pixels retain their class IDs.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to uint8 if requested
    if as_uint8 and mask.dtype != np.uint8:
        mask_to_save = mask.astype(np.uint8)
    else:
        mask_to_save = mask

    # Save mask using PIL
    img = Image.fromarray(mask_to_save)
    img.save(save_path)



def list_dir_endwith(
    dir_path: str,
    suffixes: Optional[tuple] = ('.png', '.jpg', '.jpeg')
) -> List[str]:
    """
    List files in directory with specific suffixes.
    Args:
        dir_path: Directory path
        suffixes: Tuple of file extensions to filter by
        
    Returns:
        List of file paths matching the suffixes
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    
    list_files_names = os.listdir(dir_path)
    list_selected_files = [f for f in list_files_names if "."+f.split('.')[-1].lower() in suffixes]
    return [os.path.join(dir_path, f) for f in list_selected_files]
