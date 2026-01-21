"""
Input/Output utilities for image loading and saving.
"""

import numpy as np
from PIL import Image
from typing import Optional
import os
from typing import List, Dict
from src.logger import get_logger
import csv
from pathlib import Path
from src.cste import DataPath

log = get_logger("io_utils")


def load_image(
    img_path: str, normalize: bool = True, one_channel: bool = False
) -> np.ndarray:
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
        if one_channel:  # convert to grayscale if requested
            img = img.convert("L")
    except Exception as e:
        raise ValueError(f"Failed to load image {img_path}: {e}")

    img_array = np.array(img)

    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.uint8)
    return img_array


def save_mask(mask: np.ndarray, save_path: str, as_uint8: bool = True) -> None:
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
    dir_path: str, suffixes: Optional[tuple] = (".png", ".jpg", ".jpeg")
) -> List[str]:
    """
    List files in directory with specific suffixes.
    Args:
        dir_path: Directory path
        suffixes: Tuple of file extensions to filter by

    Returns:
        List of file paths matching the suffixes (initial given directory + filename)
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    list_files_names = os.listdir(dir_path)
    list_selected_files = [
        f for f in list_files_names if "." + f.split(".")[-1].lower() in suffixes
    ]
    return [os.path.join(dir_path, f) for f in list_selected_files]


def get_filename_noext(path: str) -> str:
    """Return the file name without its extension from a given path.
    Example : '/path/to/file/image.jpg' -> 'image'
    """
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return name


def build_mapping_csv(
    img_dir: str, label_dir: str, output_csv_path: str = DataPath.CSV_MAPPING_TRAIN
) -> None:
    """
    Build a coherent CSV mapping for training data.
    CSV columns: img_id, img_path, label_path, feature_path
    """
    img_paths = list_dir_endwith(img_dir, ".jpg")
    processed = 0
    skipped = 0
    rows: List[List[str]] = []

    for img_path in img_paths:
        img_id = get_filename_noext(img_path)
        label_path = os.path.join(label_dir, img_id + "_m.png")
        rows.append([img_id, img_path, label_path])
        processed += 1

    # Write CSV
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "img_path", "label_path"])
        writer.writerows(rows)

    log.info(
        f"Dataset mapping completed: {processed}/{len(img_paths)} files processed successfully "
        f"({skipped} skipped)"
    )


def clear_folder_if_exists(folder_path: str) -> None:
    """Clear all files in the specified folder if it exists."""
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil

                    shutil.rmtree(file_path)
            except Exception as e:
                log.error(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    load_image("data/images/train/M-33-7-A-d-2-3_0.jpg", normalize=True)
    print("Shape:", load_image("data/images/train/M-33-7-A-d-2-3_0.jpg").shape)
