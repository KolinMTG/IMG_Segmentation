"""
Example usage of the segmentation pipeline.
"""

import os
import numpy as np
from io_utils import load_image, save_mask, list_dir_endwith
from pretreat_pipeline import process_all_classes, segmentation_pipeline
from mask_aggregation import aggregate_masks
from cste import DataPath, ResultPath, ClassInfo
from logger import get_logger

log = get_logger(__name__, console=True)

def example_batch_processing():
    """Example: Batch process multiple images."""
    import glob
    
    # Get all images in test directory
    img_paths = list_dir_endwith(DataPath.IMG_TEST, suffixes=['.jpg', '.png'])
    
    log.info(f"Processing {len(img_paths)} images...")
    
    for i, img_path in enumerate(img_paths):
        # Load image
        img = load_image(img_path, normalize=True)
        
        # Process all classes
        masks = process_all_classes(img)
        
        # Merge masks
        final_mask, _,_ = aggregate_masks(masks)
        
        # Save result
        img_name = os.path.basename(img_path)
        save_path = os.path.join(
            ResultPath.PREDICTION_PATH,
            img_name.replace('.jpg', '_pred.png')
        )
        save_mask(final_mask, save_path, as_uint8=True)
        
        log.info(f"  [{i+1}/{len(img_paths)}] Processed {img_path}, saved to {save_path}")
    
    log.info("Batch processing complete!")


if __name__ == "__main__":

    log.info("\n" + "=" * 70)
    log.info("Example 4: Batch processing")
    log.info("=" * 70)
    example_batch_processing()