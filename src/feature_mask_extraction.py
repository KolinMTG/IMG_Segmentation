"""
Augmented feature extraction pipeline with synchronized image-mask processing.

Extends the original feature extraction to include data augmentation while
maintaining feature consistency and deterministic reproducibility.
"""

import numpy as np
import cv2
import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from typing import Optional, Tuple, List, Dict

from cste import DataPath, GeneralConfig, ProcessingConfig
from data_augmentation import augment_segmentation_data
from feature_extraction_pipeline import extract_features
from io_utils import clear_folder_if_exists
from logger import get_logger

log = get_logger("augmented_feature_pipeline")


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================


def process_image_with_augmentation(
    img_id: str,
    img_path: str,
    mask_path: str,
    feature_dir: str,
    mask_dir: str,
    augmentation_ratio: int = 0,
    normalize: bool = True,
    downsample_fraction: float = ProcessingConfig.DOWNSAMPLE_FRACTION,
    critical_class_ids: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    """
    Process single image with optional augmentation.

    Args:
        img_id: Unique identifier for the image
        img_path: Path to input image
        mask_path: Path to input mask
        feature_dir: Directory to save features
        mask_dir: Directory to save masks
        augmentation_ratio: Number of augmentations (0 = no augmentation)
        normalize: Whether to normalize features
        downsample_fraction: Downsampling fraction for features
        critical_class_ids: List of class IDs to prioritize in augmentation

    Returns:
        List of metadata dictionaries for CSV mapping
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Convert to RGB and normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        # Ensure mask is int32
        mask = mask.astype(np.int32)

        # Store original dimensions for validation
        original_h, original_w = img.shape[:2]

        # Generate metadata records
        metadata_records = []

        if augmentation_ratio == 0:
            # No augmentation: process original only
            record = _process_single_pair(
                img,
                mask,
                img_id,
                "original",
                feature_dir,
                mask_dir,
                normalize,
                downsample_fraction,
                seed=None,
            )
            metadata_records.append(record)

        else:
            # Process original first
            record = _process_single_pair(
                img,
                mask,
                img_id,
                "original",
                feature_dir,
                mask_dir,
                normalize,
                downsample_fraction,
                seed=None,
            )
            metadata_records.append(record)

            # Generate augmented pairs
            # Use a base seed derived from img_id for reproducibility
            base_seed = _generate_seed_from_id(img_id)

            for aug_idx in range(augmentation_ratio):
                # Deterministic seed for this augmentation
                aug_seed = base_seed + aug_idx + 1

                # Generate augmented pair
                augmented_pairs = augment_segmentation_data(
                    img=img.copy(),
                    mask=mask.copy(),
                    augmentation_ratio=1,
                    critical_class_ids=critical_class_ids,
                    seed=aug_seed,
                )

                # Skip the first pair (original) and take the augmented one
                if len(augmented_pairs) > 1:
                    aug_img, aug_mask = augmented_pairs[1]
                else:
                    aug_img, aug_mask = augmented_pairs[0]

                # Validate dimensions
                if aug_img.shape[:2] != (original_h, original_w):
                    raise ValueError(
                        f"Augmented image dimension mismatch: "
                        f"expected {(original_h, original_w)}, "
                        f"got {aug_img.shape[:2]}"
                    )

                # Process augmented pair
                aug_id = f"aug_{aug_idx:03d}"
                record = _process_single_pair(
                    aug_img,
                    aug_mask,
                    img_id,
                    aug_id,
                    feature_dir,
                    mask_dir,
                    normalize,
                    downsample_fraction,
                    seed=aug_seed,
                )
                metadata_records.append(record)

        return metadata_records

    except Exception as e:
        log.error(f"Error processing {img_id}: {str(e)}")
        return []


def _process_single_pair(
    img: np.ndarray,
    mask: np.ndarray,
    img_id: str,
    variant_id: str,
    feature_dir: str,
    mask_dir: str,
    normalize: bool,
    downsample_fraction: float,
    seed: Optional[int],
) -> Dict[str, str]:
    """
    Process and save a single image-mask pair.

    Args:
        img: RGB image in [0, 1]
        mask: Integer mask
        img_id: Original image ID
        variant_id: Variant identifier ("original" or "aug_XXX")
        feature_dir: Directory for features
        mask_dir: Directory for masks
        normalize: Whether to normalize features
        downsample_fraction: Downsampling fraction
        seed: Augmentation seed (None for original)

    Returns:
        Metadata dictionary with paths and seed
    """
    # Generate unique ID for this variant
    if variant_id == "original":
        unique_id = img_id
    else:
        unique_id = f"{img_id}_{variant_id}"

    # Define paths
    feature_path = os.path.join(feature_dir, f"{unique_id}.npy")
    mask_save_path = os.path.join(mask_dir, f"{unique_id}.npy")

    # Extract features (this applies downsampling)
    features = extract_features(
        img=img,
        normalize=normalize,
        save=True,
        save_path=feature_path,
        downsample_fraction=downsample_fraction,
    )

    # Downsample mask to match feature dimensions
    feature_h, feature_w = features.shape[:2]
    mask_downsampled = cv2.resize(
        mask, (feature_w, feature_h), interpolation=cv2.INTER_NEAREST
    )

    # Save downsampled mask
    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    np.save(mask_save_path, mask_downsampled.astype(np.int32))

    # Create metadata record
    return {
        "img_id": unique_id,
        "original_id": img_id,
        "variant": variant_id,
        "feature_path": feature_path,
        "mask_path": mask_save_path,
        "seed": seed if seed is not None else -1,
    }


def _generate_seed_from_id(img_id: str) -> int:
    """Generate deterministic seed from image ID."""
    # Use hash of img_id for reproducible seed
    return abs(hash(img_id)) % (2**31)


# ============================================================================
# BATCH PROCESSING WITH MULTIPROCESSING
# ============================================================================


def _worker_process_image(
    row: Dict,
    feature_dir: str,
    mask_dir: str,
    augmentation_ratio: int,
    normalize: bool,
    downsample_fraction: float,
    critical_class_ids: Optional[List[int]],
) -> Tuple[bool, List[Dict], str]:
    """
    Worker function for multiprocessing.

    Args:
        row: Dictionary with img_id, img_path, label_path
        feature_dir: Directory for features
        mask_dir: Directory for masks
        augmentation_ratio: Number of augmentations
        normalize: Whether to normalize features
        downsample_fraction: Downsampling fraction
        critical_class_ids: List of critical class IDs

    Returns:
        (success, metadata_records, error_message)
    """
    try:
        metadata_records = process_image_with_augmentation(
            img_id=row["img_id"],
            img_path=row["img_path"],
            mask_path=row["label_path"],
            feature_dir=feature_dir,
            mask_dir=mask_dir,
            augmentation_ratio=augmentation_ratio,
            normalize=normalize,
            downsample_fraction=downsample_fraction,
            critical_class_ids=critical_class_ids,
        )

        if len(metadata_records) > 0:
            return True, metadata_records, None
        else:
            return False, [], f"No records generated for {row['img_id']}"

    except Exception as e:
        return False, [], f"Error processing {row['img_id']}: {str(e)}"


def extract_features_with_augmentation(
    input_csv_path: str,
    output_csv_path: str,
    feature_dir: str = DataPath.FEATURE_TRAIN,
    mask_dir: str = DataPath.MASK_TRAIN,
    augmentation_ratio: int = ProcessingConfig.AUGMENTATION_RATIO,
    critical_class_ids: Optional[List[int]] = ProcessingConfig.CRITICAL_CLASS_IDS,
    num_workers: int = GeneralConfig.NB_JOBS,
    normalize: bool = True,
    downsample_fraction: float = ProcessingConfig.DOWNSAMPLE_FRACTION,
    clear_folders: bool = False,
) -> pd.DataFrame:
    """
    Extract features with optional augmentation for entire dataset.

    This function:
    1. Reads input CSV with columns: img_id, img_path, label_path
    2. For each image, generates augmented variants if augmentation_ratio > 0
    3. Extracts features for each image/augmented variant
    4. Saves features and masks to specified directories
    5. Creates output CSV mapping: img_id, feature_path, mask_path, seed

    Args:
        input_csv_path: Path to input CSV mapping
        output_csv_path: Path to save output CSV mapping
        feature_dir: Directory to save features (.npy)
        mask_dir: Directory to save masks (.npy)
        augmentation_ratio: Number of augmentations per image (0 = no augmentation)
        critical_class_ids: List of class IDs to prioritize in augmentation
        num_workers: Number of parallel workers
        normalize: Whether to normalize features to [0, 1]
        downsample_fraction: Downsampling fraction for feature extraction

    Returns:
        DataFrame with mapping: img_id, original_id, variant, feature_path, mask_path, seed
    """
    log.info("=" * 60)
    log.info("AUGMENTED FEATURE EXTRACTION PIPELINE")
    log.info("=" * 60)
    log.info(f"Input CSV: {input_csv_path}")
    log.info(f"Output CSV: {output_csv_path}")
    log.info(f"Feature directory: {feature_dir}")
    log.info(f"Mask directory: {mask_dir}")
    log.info(f"Augmentation ratio: {augmentation_ratio}")
    log.info(f"Critical classes: {critical_class_ids}")
    log.info(f"Downsample fraction: {downsample_fraction}")
    log.info(f"Workers: {num_workers}")
    log.info("=" * 60)

    # Create output directories
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if clear_folders:
        log.info("Clearing existing feature and mask directories...")
        clear_folder_if_exists(feature_dir)
        clear_folder_if_exists(mask_dir)
        log.info("Cleared existing data.")

    # Read input CSV
    log.info(f"Reading input CSV: {input_csv_path}")
    df_input = pd.read_csv(input_csv_path)
    total_images = len(df_input)

    if total_images == 0:
        log.warning("No images found in input CSV")
        return pd.DataFrame()

    log.info(f"Found {total_images} images to process")

    if augmentation_ratio > 0:
        expected_outputs = total_images * (1 + augmentation_ratio)
        log.info(
            f"Expected output samples: {expected_outputs} "
            f"({total_images} original + {total_images * augmentation_ratio} augmented)"
        )

    # Convert to list of dicts for multiprocessing
    rows = df_input.to_dict(orient="records")

    # Prepare worker function
    worker_fn = partial(
        _worker_process_image,
        feature_dir=feature_dir,
        mask_dir=mask_dir,
        augmentation_ratio=augmentation_ratio,
        normalize=normalize,
        downsample_fraction=downsample_fraction,
        critical_class_ids=critical_class_ids,
    )

    # Process with multiprocessing
    log.info(f"Starting processing with {num_workers} workers...")

    all_metadata_records = []
    success_count = 0
    error_count = 0

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker_fn, rows), total=total_images, desc="Processing images"
            )
        )

    # Collect results
    for success, metadata_records, error_msg in results:
        if success:
            success_count += 1
            all_metadata_records.extend(metadata_records)
        else:
            error_count += 1
            if error_msg:
                log.warning(error_msg)

    # Create output DataFrame
    if len(all_metadata_records) > 0:
        df_output = pd.DataFrame(all_metadata_records)

        # Save output CSV
        df_output.to_csv(output_csv_path, index=False)
        log.info(f"Saved output CSV: {output_csv_path}")
    else:
        df_output = pd.DataFrame()
        log.warning("No records to save")

    # Final summary
    log.info("=" * 60)
    log.info("PROCESSING COMPLETE")
    log.info("=" * 60)
    log.info(f"Total input images: {total_images}")
    log.info(f"Successfully processed: {success_count}")
    log.info(f"Failed: {error_count}")
    log.info(f"Success rate: {100.0 * success_count / total_images:.2f}%")
    log.info(f"Total output samples: {len(all_metadata_records)}")

    if augmentation_ratio > 0:
        original_count = sum(
            1 for r in all_metadata_records if r["variant"] == "original"
        )
        augmented_count = len(all_metadata_records) - original_count
        log.info(f"  - Original samples: {original_count}")
        log.info(f"  - Augmented samples: {augmented_count}")

    log.info("=" * 60)

    return df_output


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example: Process training set with augmentation
    log.info("Processing training set with augmentation...")
    extract_features_with_augmentation(
        input_csv_path=DataPath.CSV_SELECTED_IMAGES_TRAIN,
        output_csv_path=DataPath.CSV_FEATURE_MASK_MAPPING_TRAIN,
        feature_dir=DataPath.FEATURE_TRAIN,
        mask_dir=DataPath.MASK_DIR_TRAIN,
        augmentation_ratio=GeneralConfig.AUGMENTATION_RATIO,
        critical_class_ids=GeneralConfig.CRITICAL_CLASS_IDS,
        num_workers=GeneralConfig.NB_JOBS,
        normalize=True,
        downsample_fraction=ProcessingConfig.DOWNSAMPLE_FRACTION,
    )

    log.info("All processing complete!")
