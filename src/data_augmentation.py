import numpy as np
import cv2
from typing import List, Tuple, Optional

from src.cste import ProcessingConfig

from src.logger import get_logger

log = get_logger("data_augmentation")


def augment_segmentation_data(
    img: np.ndarray,
    mask: np.ndarray,
    augmentation_ratio: int = ProcessingConfig.AUGMENTATION_RATIO,
    critical_class_ids: Optional[List[int]] = ProcessingConfig.CRITICAL_CLASS_IDS,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate augmented image-mask pairs for semantic segmentation.

    Args:
        img: Image array of shape (H, W, 3), dtype float32 in [0, 1]
        mask: Mask array of shape (H, W), dtype int32 with class IDs
        augmentation_ratio: Number of augmented samples to generate
        critical_class_ids: List of class IDs to prioritize (optional)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of (augmented_image, augmented_mask) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]
    augmented_pairs = []

    # Determine if we should oversample critical classes
    use_critical_sampling = critical_class_ids and len(critical_class_ids) > 0
    critical_mask = (
        _create_critical_mask(mask, critical_class_ids)
        if use_critical_sampling
        else None
    )

    for i in range(augmentation_ratio):
        # Decide whether to apply critical class focus
        apply_critical_focus = use_critical_sampling and np.random.rand() > 0.3

        if apply_critical_focus and np.any(critical_mask):
            aug_img, aug_mask = _augment_with_critical_focus(
                img, mask, critical_mask, h, w
            )
        else:
            aug_img, aug_mask = _augment_standard(img, mask, h, w)

        augmented_pairs.append((aug_img, aug_mask))

    return augmented_pairs


def _create_critical_mask(
    mask: np.ndarray, critical_class_ids: List[int]
) -> np.ndarray:
    """Create binary mask indicating presence of critical classes."""
    critical_mask = np.zeros_like(mask, dtype=bool)
    for class_id in critical_class_ids:
        critical_mask |= mask == class_id
    return critical_mask


def _augment_standard(
    img: np.ndarray, mask: np.ndarray, target_h: int, target_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply standard augmentation pipeline."""
    # Random rotation (discrete angles)
    angle = np.random.choice(ProcessingConfig.ROTATION_ANGLES)
    aug_img, aug_mask = _rotate_pair(img, mask, angle)

    # Random zoom
    zoom_factor = np.random.uniform(
        ProcessingConfig.ZOOM_RANGE[0], ProcessingConfig.ZOOM_RANGE[1]
    )
    aug_img, aug_mask = _zoom_pair(aug_img, aug_mask, zoom_factor, target_h, target_w)

    # Random horizontal/vertical flip
    if np.random.rand() > 0.5:
        aug_img = np.fliplr(aug_img)
        aug_mask = np.fliplr(aug_mask)
    if np.random.rand() > 0.5:
        aug_img = np.flipud(aug_img)
        aug_mask = np.flipud(aug_mask)

    # Gaussian blur on image only
    if np.random.rand() > 0.5:
        kernel_size = np.random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)

    # Brightness and contrast adjustment (image only)
    if np.random.rand() > 0.5:
        aug_img = _adjust_brightness_contrast(aug_img)

    return aug_img, aug_mask


def _augment_with_critical_focus(
    img: np.ndarray,
    mask: np.ndarray,
    critical_mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentation with focus on critical class regions."""
    # Find bounding box of critical regions
    coords = np.argwhere(critical_mask)
    if len(coords) == 0:
        return _augment_standard(img, mask, target_h, target_w)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Expand bounding box with margin
    margin = 20
    y_min = max(0, y_min - margin)
    y_max = min(img.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(img.shape[1], x_max + margin)

    # Random shift within the critical region
    shift_range = 10
    dy = np.random.randint(-shift_range, shift_range + 1)
    dx = np.random.randint(-shift_range, shift_range + 1)

    y_min = np.clip(y_min + dy, 0, img.shape[0] - target_h)
    x_min = np.clip(x_min + dx, 0, img.shape[1] - target_w)

    # Ensure we don't exceed boundaries
    if y_min + target_h > img.shape[0]:
        y_min = img.shape[0] - target_h
    if x_min + target_w > img.shape[1]:
        x_min = img.shape[1] - target_w

    # Crop region
    crop_img = img[y_min : y_min + target_h, x_min : x_min + target_w].copy()
    crop_mask = mask[y_min : y_min + target_h, x_min : x_min + target_w].copy()

    # Apply standard augmentations to cropped region
    return _augment_standard(crop_img, crop_mask, target_h, target_w)


def _rotate_pair(
    img: np.ndarray, mask: np.ndarray, angle: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by discrete angle."""
    if angle == 0:
        return img.copy(), mask.copy()

    k = angle // 90
    rot_img = np.rot90(img, k)
    rot_mask = np.rot90(mask, k)

    return rot_img, rot_mask


def _zoom_pair(
    img: np.ndarray, mask: np.ndarray, zoom_factor: float, target_h: int, target_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply zoom to image and mask, maintaining original dimensions."""
    h, w = img.shape[:2]

    if zoom_factor > 1.0:
        # Zoom in: resize larger then crop center
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Center crop
        start_y = (new_h - target_h) // 2
        start_x = (new_w - target_w) // 2
        cropped_img = resized_img[
            start_y : start_y + target_h, start_x : start_x + target_w
        ]
        cropped_mask = resized_mask[
            start_y : start_y + target_h, start_x : start_x + target_w
        ]

        return cropped_img, cropped_mask
    else:
        # Zoom out: resize smaller then pad
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to original size
        pad_y = (target_h - new_h) // 2
        pad_x = (target_w - new_w) // 2

        padded_img = np.zeros((target_h, target_w, 3), dtype=img.dtype)
        padded_mask = np.zeros((target_h, target_w), dtype=mask.dtype)

        padded_img[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_img
        padded_mask[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_mask

        return padded_img, padded_mask


def _adjust_brightness_contrast(img: np.ndarray) -> np.ndarray:
    """Randomly adjust brightness and contrast of image."""
    # Brightness adjustment
    brightness_delta = np.random.uniform(
        ProcessingConfig.BRIGHTNESS_RANGE[0], ProcessingConfig.BRIGHTNESS_RANGE[1]
    )
    img_adjusted = img + brightness_delta

    # Contrast adjustment
    contrast_factor = np.random.uniform(
        ProcessingConfig.CONTRAST_RANGE[0], ProcessingConfig.CONTRAST_RANGE[1]
    )
    mean = img_adjusted.mean()
    img_adjusted = (img_adjusted - mean) * contrast_factor + mean

    # Clip to valid range
    return np.clip(img_adjusted, 0.0, 1.0)
