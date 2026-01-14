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

from train_classifier import train_models_from_directory
from classes.field import process_field
from classes.building import process_building
from classes.woodland import process_woodland
from classes.water import process_water
from classes.road import process_road
from classifier_inference import compute_normalized_probabilities
from cste import ClassInfo, DataPath

from src.logger import get_logger
from src.io_utils import build_mapping_csv
from src.optimized_feature_pipeline import extract_features_batch
from src.cste import GeneralConfig, ProcessingConfig




# ============================================================================
# SINGLE IMAGE INFERENCE
# ============================================================================

def single_image_inference_example() -> None:
    """
    Example: Run inference on a single image.
    
    ! This demonstrates the main API: process_<class>(img) -> prob_map
    """
    # Load test image
    img_path = "data/images/test/M-33-7-A-d-2-3_19.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # ! Get probability map for each class (API unchanged from filter-based version)
    field_prob = process_field(img)
    building_prob = process_building(img)
    woodland_prob = process_woodland(img)
    water_prob = process_water(img)
    road_prob = process_road(img)
    
    # Verify sum-to-one constraint
    total_prob = (
        field_prob + building_prob + woodland_prob + 
        water_prob + road_prob
    )
    
    # Should be ~1.0 everywhere (within numerical precision)
    assert np.allclose(total_prob, 1.0, atol=1e-5)




# ============================================================================
# MAIN EXAMPLES
# ============================================================================

if __name__ == "__main__":

    # 1 Generate mapping CSV for training data
    build_mapping_csv(
        img_dir=DataPath.IMG_TRAIN,
        label_dir=DataPath.LABEL_TRAIN,
        feature_dir=DataPath.FEATURE_TRAIN,
        output_csv_path=DataPath.CSV_MAPPING_TRAIN
    )

    # 2 Extract features for the training dataset (if not already done)
    # extract_features_batch(
    #     mapping_csv_path=DataPath.CSV_MAPPING_TRAIN,
    #     num_workers=GeneralConfig.NB_JOBS,
    #     normalize=True,
    #     downsample_fraction=ProcessingConfig.DOWNSAMPLE_FRACTION
    # )

    # 3 Train GMM models from training data
    train_models_from_directory()