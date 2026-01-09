"""
Example usage of the segmentation pipeline.
"""

import os
import numpy as np
from io_utils import load_image, save_mask, save_colored_mask
from pretreat_pipeline import (
    segmentation_pipeline,
    process_all_classes,
    merge_masks_majority_vote,
    get_class_name
)
from cste import DataPath, ResultPath, ClassInfo
from logger import get_logger

log = get_logger(__name__, console=True)


def example_single_class():
    """Example: Process single class."""
    # Load imag
    img_path = os.path.join(DataPath.IMG_TRAIN, "example.jpg")
    img = load_image(img_path, normalize=True)
    
    # Process only building class
    masks = segmentation_pipeline(img, class_ids=[1])
    building_mask = masks[1]
    
    # Save result
    save_path = os.path.join(ResultPath.PREDICTION_PATH, "building_mask.png")
    save_mask(building_mask, save_path)
    
    log.info(f"Building mask saved to {save_path}")


def example_multiple_classes():
    """Example: Process multiple classes."""
    # Load image
    img_path = os.path.join(DataPath.IMG_TRAIN, "example.jpg")
    img = load_image(img_path, normalize=True)
    
    # Process specific classes
    class_ids = [0, 1, 4]  # Field, Building, Road
    masks = segmentation_pipeline(img, class_ids=class_ids)
    
    # Save each mask
    for class_id, mask in masks.items():
        class_name = get_class_name(class_id)
        save_path = os.path.join(
            ResultPath.PREDICTION_PATH,
            f"{class_name.lower()}_mask.png"
        )
        save_mask(mask, save_path)
        log.info(f"{class_name} mask saved to {save_path}")


def example_full_segmentation():
    """Example: Full segmentation with all classes."""
    # Load image
    img_path = os.path.join(DataPath.IMG_TRAIN, "example.jpg")
    img = load_image(img_path, normalize=True)
    
    # Process all classes
    masks = process_all_classes(img)
    
    # Merge into final segmentation
    final_mask = merge_masks_majority_vote(masks)
    
    # Save final mask as colored image
    save_path = os.path.join(
        ResultPath.PREDICTION_PATH,
        "final_segmentation.png"
    )
    save_colored_mask(final_mask, save_path, ClassInfo.CLASS_COLORS)
    
    log.info(f"Final segmentation saved to {save_path}")
    
    # log.info class statistics
    log.info("\nClass distribution:")
    for class_id in sorted(ClassInfo.CLASS_NAMES.keys()):
        pixel_count = np.sum(final_mask == class_id)
        percentage = 100 * pixel_count / final_mask.size
        log.info(f"  {get_class_name(class_id)}: {percentage:.2f}%")


def example_batch_processing():
    """Example: Batch process multiple images."""
    import glob
    
    # Get all images in train directory
    img_pattern = os.path.join(DataPath.IMG_TRAIN, "*.jpg")
    img_paths = glob.glob(img_pattern)
    
    log.info(f"Processing {len(img_paths)} images...")
    
    for i, img_path in enumerate(img_paths):
        # Load image
        img = load_image(img_path, normalize=True)
        
        # Process all classes
        masks = process_all_classes(img)
        
        # Merge masks
        final_mask = merge_masks_majority_vote(masks)
        
        # Save result
        img_name = os.path.basename(img_path)
        save_path = os.path.join(
            ResultPath.PREDICTION_PATH,
            img_name.replace('.jpg', '_seg.png')
        )
        save_colored_mask(final_mask, save_path, ClassInfo.CLASS_COLORS)
        
        log.info(f"  [{i+1}/{len(img_paths)}] Processed {img_name}")
    
    log.info("Batch processing complete!")


if __name__ == "__main__":

    log.info("\n" + "=" * 70)
    log.info("Example 4: Batch processing")
    log.info("=" * 70)
    example_batch_processing()
