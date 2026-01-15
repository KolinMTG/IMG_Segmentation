"""
Example usage of the satellite image segmentation pipeline.

This script demonstrates how to train and evaluate all models:
- U-Net (deep learning)
- Random Forest (classical supervised)
- K-Means (classical unsupervised)
"""

import logging
from pathlib import Path

from data.dataset import SegmentationDataset
from models.unet import UNet
from models.random_forest import RandomForestSegmentation
from models.kmeans import KMeansSegmentation
from training.trainer import train_and_evaluate_model

#! Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    """Main training and evaluation pipeline."""
    
    #! ========================================
    #! Configuration
    #! ========================================
    
    #! Data paths
    TRAIN_CSV = 'data/train.csv'
    TEST_CSV = 'data/test.csv'
    
    #! Model configuration
    NUM_CLASSES = 5  # e.g., background, building, road, vegetation, water
    IMAGE_HEIGHT = 254
    IMAGE_WIDTH = 254
    
    #! Feature selection (None = use all features, or specify indices)
    FEATURE_IDS = None  # Example: [0, 3, 4, 7, 10] to select specific features
    NUM_FEATURES = 18 if FEATURE_IDS is None else len(FEATURE_IDS)
    
    #! Training configuration
    VAL_SPLIT = 0.2
    OUTPUT_BASE_DIR = './outputs'
    
    #! ========================================
    #! 1. Train U-Net (Deep Learning)
    #! ========================================
    
    log.info("\n" + "="*70)
    log.info("TRAINING U-NET")
    log.info("="*70)
    
    unet = UNet(
        num_classes=NUM_CLASSES,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FEATURES),
        filters=32,
        use_one_hot=False  # Use integer masks with categorical crossentropy
    )
    
    unet_results = train_and_evaluate_model(
        model=unet,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        num_classes=NUM_CLASSES,
        feature_ids=FEATURE_IDS,
        val_split=VAL_SPLIT,
        output_dir=f'{OUTPUT_BASE_DIR}/unet',
        #! U-Net specific training parameters
        epochs=50,
        batch_size=8,
        learning_rate=1e-3
    )
    
    #! ========================================
    #! 2. Train Random Forest (Classical Supervised)
    #! ========================================
    
    log.info("\n" + "="*70)
    log.info("TRAINING RANDOM FOREST")
    log.info("="*70)
    
    rf = RandomForestSegmentation(
        num_classes=NUM_CLASSES,
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    rf_results = train_and_evaluate_model(
        model=rf,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        num_classes=NUM_CLASSES,
        feature_ids=FEATURE_IDS,
        val_split=VAL_SPLIT,
        output_dir=f'{OUTPUT_BASE_DIR}/random_forest',
        #! Random Forest specific training parameters
        sample_fraction=0.1  # Sample 10% of pixels for memory efficiency
    )
    
    #! ========================================
    #! 3. Train K-Means (Classical Unsupervised)
    #! ========================================
    
    log.info("\n" + "="*70)
    log.info("TRAINING K-MEANS")
    log.info("="*70)
    
    kmeans = KMeansSegmentation(
        num_classes=NUM_CLASSES,
        batch_size=10000,
        n_init=10,
        random_state=42
    )
    
    kmeans_results = train_and_evaluate_model(
        model=kmeans,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        num_classes=NUM_CLASSES,
        feature_ids=FEATURE_IDS,
        val_split=VAL_SPLIT,
        output_dir=f'{OUTPUT_BASE_DIR}/kmeans',
        #! K-Means specific training parameters
        sample_fraction=0.2  # Sample 20% of pixels for clustering
    )
    
    #! ========================================
    #! Summary Comparison
    #! ========================================
    
    log.info("\n" + "="*70)
    log.info("FINAL COMPARISON")
    log.info("="*70)
    
    models = {
        'U-Net': unet_results['evaluation'],
        'Random Forest': rf_results['evaluation'],
        'K-Means': kmeans_results['evaluation']
    }
    
    log.info("\nModel Performance Comparison:")
    log.info(f"{'Model':<20} {'Accuracy':<12} {'Mean IoU':<12} {'Mean Dice':<12}")
    log.info("-" * 70)
    
    for model_name, results in models.items():
        log.info(
            f"{model_name:<20} "
            f"{results['accuracy']:<12.4f} "
            f"{results['mean_iou']:<12.4f} "
            f"{results['mean_dice']:<12.4f}"
        )
    
    log.info("\nAll models trained and evaluated successfully!")
    log.info(f"Results saved to: {OUTPUT_BASE_DIR}/")


def example_feature_selection():
    """
    Example of training models with specific feature subsets.
    
    Demonstrates feature selection for dimensionality reduction.
    """
    
    #! Configuration
    TRAIN_CSV = 'data/train.csv'
    TEST_CSV = 'data/test.csv'
    NUM_CLASSES = 5
    
    #! Select a subset of features (e.g., the 5 most informative)
    SELECTED_FEATURES = [0, 3, 7, 10, 15]  # Example indices
    
    log.info(f"Training models with selected features: {SELECTED_FEATURES}")
    
    #! Train U-Net with selected features
    unet = UNet(
        num_classes=NUM_CLASSES,
        input_shape=(254, 254, len(SELECTED_FEATURES)),
        filters=32,
        use_one_hot=False
    )
    
    unet_results = train_and_evaluate_model(
        model=unet,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        num_classes=NUM_CLASSES,
        feature_ids=SELECTED_FEATURES,  # Only use selected features
        output_dir='./outputs/unet_selected_features',
        epochs=30,
        batch_size=8
    )
    
    log.info(f"U-Net with {len(SELECTED_FEATURES)} features achieved:")
    log.info(f"  Accuracy: {unet_results['evaluation']['accuracy']:.4f}")
    log.info(f"  Mean IoU: {unet_results['evaluation']['mean_iou']:.4f}")


def example_model_loading():
    """Example of loading a trained model and making predictions."""
    
    from models.unet import UNet
    import numpy as np
    
    #! Load trained model
    unet = UNet(
        num_classes=5,
        input_shape=(254, 254, 18),
        filters=32
    )
    unet.load('./outputs/unet')
    
    log.info("Model loaded successfully")
    
    #! Make predictions on new data
    # X_new = np.load('path/to/new/features.npy')  # Shape: (N, 254, 254, 18)
    # predictions = unet.predict(X_new)
    # log.info(f"Predictions shape: {predictions.shape}")


if __name__ == '__main__':
    """
    Run the complete training pipeline for all models.
    
    Usage:
        python main.py
    
    Expected outputs:
        ./outputs/
        ├── unet/
        │   ├── unet_model.h5
        │   ├── config.json
        │   ├── UNet_training_history.json
        │   └── UNet_evaluation.json
        ├── random_forest/
        │   ├── random_forest.pkl
        │   ├── config.json
        │   ├── RandomForest_training_history.json
        │   └── RandomForest_evaluation.json
        └── kmeans/
            ├── kmeans.pkl
            ├── cluster_mapping.pkl
            ├── config.json
            ├── KMeans_training_history.json
            └── KMeans_evaluation.json
    """
    main()
    
    #! Optionally run additional examples
    # example_feature_selection()
    # example_model_loading()
