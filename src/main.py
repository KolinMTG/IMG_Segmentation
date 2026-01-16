"""
Complete usage examples for Bayesian probabilistic segmentation system.

This script demonstrates:
1. Training GMM models from labeled data
2. Running inference on single images
3. Batch processing
4. Integration with existing aggregation pipeline
5. Visualization and evaluation
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt


# from classes.field import process_field
# from classes.building import process_building
# from classes.woodland import process_woodland
# from classes.water import process_water
# from classes.road import process_road
from cste import ClassInfo, DataPath, ImgSelectionRule, GeneralConfig, ProcessingConfig

from src.logger import get_logger
from src.io_utils import build_mapping_csv
from src.data_utils import save_class_statistics, select_img
from feature_extraction_pipeline import extract_features_batch
from feature_mask_extraction import extract_features_with_augmentation



log = get_logger("main_examples")

# ============================================================================
# MAIN EXAMPLES
# ============================================================================

if __name__ == "__main__":

    #! 1 Generate mapping CSV for each data group (train, test, validation) training data
    # configs = [
    #     (DataPath.IMG_TRAIN, DataPath.LABEL_TRAIN, DataPath.FEATURE_TRAIN, DataPath.MASK_TRAIN, DataPath.CSV_MAPPING_TRAIN), # Training data
    #     (DataPath.IMG_TEST, DataPath.LABEL_TEST, DataPath.FEATURE_TEST, DataPath.MASK_TEST, DataPath.CSV_MAPPING_TEST), # Test data
    #     (DataPath.IMG_VAL, DataPath.LABEL_VAL, DataPath.FEATURE_VAL, DataPath.MASK_VAL, DataPath.CSV_MAPPING_VAL), # Validation data
    # ]

    # for img_dir, label_dir, feature_dir, mask_dir, output_csv_path in configs:
    #     build_mapping_csv(
    #         img_dir=img_dir,
    #         label_dir=label_dir,
    #         output_csv_path=output_csv_path
    #     )
    #     log.info(f"Mapping CSV saved to {output_csv_path}")
    # return 3 CSV files with columns : img_id,img_path,label_path

    #! 2 Select image based on class statistics for training data (e.g., at least 1% building or road or water)
    # save_class_statistics(
    #     mapping_csv=DataPath.CSV_MAPPING_TRAIN,
    #     output_csv=DataPath.CSV_CLASS_STATISTICS_TRAIN,
    #     num_classes=ClassInfo.NUM_CLASSES
    # )
    #return a CSV with class proportions and counts for each image (header : img_id,img_path,label_path,prop_class_0,...,prop_class_N,count_class_0,...,count_class_N)

    # select_img(
    #     mapping_csv=DataPath.CSV_MAPPING_TRAIN,
    #     class_statistics_csv=DataPath.CSV_CLASS_STATISTICS_TRAIN,
    #     rule=ImgSelectionRule.BUILDING_OR_ROAD_OR_WATER,
    #     output_csv=DataPath.CSV_SELECTED_IMAGES_TRAIN
    # )
    # return a CSV with only the selected images (header : img_id,img_path,label_path)

    #! 3 Select feature to extract (only selected classes for training, and all features for testing/validation)

    extraction_feature_configs = [
        # (DataPath.CSV_SELECTED_IMAGES_TRAIN, DataPath.CSV_FEATURE_MASK_MAPPING_TRAIN, DataPath.FEATURE_TRAIN, DataPath.MASK_TRAIN, ProcessingConfig.AUGMENTATION_RATIO),
        # (DataPath.CSV_MAPPING_TEST, DataPath.CSV_FEATURE_MASK_MAPPING_TEST, DataPath.FEATURE_TEST, DataPath.MASK_TEST, 0),  # No augmentation for test data
        (DataPath.CSV_MAPPING_VAL, DataPath.CSV_FEATURE_MASK_MAPPING_VAL, DataPath.FEATURE_VAL, DataPath.MASK_VAL, 0),  # No augmentation for validation data
    ]

    for input_csv, output_csv, feature_dir, mask_dir, augmentation_ratio in extraction_feature_configs:
        extract_features_with_augmentation(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            feature_dir=feature_dir,
            mask_dir=mask_dir,
            augmentation_ratio=augmentation_ratio,
            num_workers=GeneralConfig.NB_JOBS,
            normalize=True,
            downsample_fraction=ProcessingConfig.DOWNSAMPLE_FRACTION,
            clear_folders=True
        )





    # return a CSV and extracted features/masks for training data with augmentation (header : img_id,img_path,label_path,feature_path,mask_path)


    # 2 Extract features for the training dataset (if not already done)
    # extract_features_batch(
    #     mapping_csv_path=DataPath.CSV_MAPPING_TRAIN,
    #     num_workers=GeneralConfig.NB_JOBS,
    #     normalize=True,
    #     downsample_fraction=ProcessingConfig.DOWNSAMPLE_FRACTION
    # )
