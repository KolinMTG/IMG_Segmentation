# Project Source Structure Documentation

## 1. Overview

This project implements a complete pipeline for semantic segmentation of satellite imagery, featuring multiple segmentation approaches (U-Net, histogram-based KDE, K-Means, Random Forest) and comprehensive data processing utilities.

### All Python Files

- **[cste.py](../src/cste.py)** – Central configuration file with paths, class definitions, and processing parameters
- **[logger.py](../src/logger.py)** – Logging configuration for the application
- **[io_utils.py](../src/io_utils.py)** – File I/O utilities for loading/saving images and masks
- **[data_utils.py](../src/data_utils.py)** – Data manipulation utilities for masks and class statistics
- **[data_info.py](../src/data_info.py)** – Visualization and analysis of dataset statistics
- **[data_feature_analysis.py](../src/data_feature_analysis.py)** – Global statistical analysis of pixel-level features
- **[data_augmentation.py](../src/data_augmentation.py)** – Data augmentation for segmentation pairs
- **[feature_extraction_pipeline.py](../src/feature_extraction_pipeline.py)** – Optimized feature extraction with 19-dimensional vectors
- **[feature_mask_extraction.py](../src/feature_mask_extraction.py)** – Augmented feature extraction with synchronized image-mask processing
- **[histogram_extraction.py](../src/histogram_extraction.py)** – Histogram-based KDE extraction for segmentation
- **[histogram_evaluation.py](../src/histogram_evaluation.py)** – Inference and evaluation using histogram-based KDE
- **[training_examples.py](../src/training_examples.py)** – Training scripts for all segmentation models
- **[inference.py](../src/inference.py)** – U-Net inference pipeline
- **[post_treatement.py](../src/post_treatement.py)** – Post-processing pipeline for segmentation masks
- **[evaluation.py](../src/evaluation.py)** – Comprehensive evaluation metrics and visualizations
- **[main.py](../src/main.py)** – Main execution script orchestrating the complete pipeline

---

## 2. Grouping by Major Themes

### 2.1 Configuration & Utilities

#### [cste.py](../src/cste.py)
Central configuration hub containing:
- **Path definitions** (`DataPath`, `GeneralPath`, `ResultPath`, `TestPath`) for all data directories
- **Class information** (`ClassInfo`) with class names, colors, and priorities
- **Feature definitions** (`FeatureInfo`) mapping feature indices to semantic names
- **Processing parameters** (`ProcessingConfig`) for image processing operations
- **Selection rules** (`ImgSelectionRule`) for filtering images based on class content

**Key usage**: Import configuration constants throughout the project to ensure consistency.

#### [logger.py](../src/logger.py)
Provides the `get_logger()` function to create configured loggers with:
- File output to `.logs/` directory
- Optional console output
- Timestamped log messages

**Key usage**: `log = get_logger("module_name")` at the start of each module.

---

### 2.2 Data Loading & I/O

#### [io_utils.py](../src/io_utils.py)
Core I/O operations:
- **`load_image()`** – Load RGB/grayscale images with optional normalization
- **`save_mask()`** – Save segmentation masks as PNG files
- **`build_mapping_csv()`** – Generate CSV mappings linking images to labels
- **`list_dir_endwith()`** – List files with specific extensions
- **`clear_folder_if_exists()`** – Clean directories before processing

**Key usage**: Essential for all data loading operations; generates initial CSV mappings for train/val/test splits.

#### [data_utils.py](../src/data_utils.py)
Data manipulation utilities:
- **`mask_0n_to_onehot()`** – Convert integer masks to one-hot encoding
- **`compute_class_proportions()`** – Calculate class distribution in masks
- **`save_class_statistics()`** – Generate CSV with per-image class statistics
- **`select_img()`** – Filter images based on class presence rules (AND/OR logic)

**Key usage**: Used to analyze dataset composition and select training images with specific class distributions.

---

### 2.3 Data Visualization & Analysis

#### [data_info.py](../src/data_info.py)
Comprehensive visualization tools:
- **`show_img_labels()`** – Visualize image with colored segmentation overlay
- **`class_proportion()`** – Analyze global class distribution across dataset
- **`class_proportion_by_image()`** – Analyze class presence per image
- **`show_feature()`** – Visualize individual feature channels
- **`show_img_gt_vs_pred()`** – Compare ground truth vs. predictions
- **`generate_prediction_visualizations()`** – Batch generate comparison visualizations
- **`save_precision_recall_grouped_plot()`** – Create metrics bar charts from JSON

**Key usage**: Essential for understanding dataset characteristics and visually inspecting model predictions.

#### [data_feature_analysis.py](../src/data_feature_analysis.py)
Global statistical feature analysis:
- **`sample_balanced_pixels()`** – Sample balanced pixels across classes for analysis
- **`analyze_global_distributions()`** – Visualize feature distributions with histograms/KDE
- **`analyze_class_conditional()`** – Create boxplots and violin plots per class
- **`analyze_discriminativeness()`** – Rank features by ANOVA F-score
- **`analyze_correlations()`** – Visualize feature correlation matrix
- **`analyze_pca()`** – PCA analysis with 2D/3D projections
- **`run_global_analysis()`** – Execute complete analysis pipeline

**Key usage**: Run before model training to understand feature separability and identify important features.

---

### 2.4 Feature Extraction & Augmentation

#### [feature_extraction_pipeline.py](../src/feature_extraction_pipeline.py)
Optimized 19-dimensional feature extraction:
- **Feature types**: RGB, HSV, grayscale, multi-scale blur, gradients, texture (variance, entropy, LBP), spectral indices (NDVI, water index), geometric features (anisotropy, corner density)
- **`extract_features()`** – Core function extracting all features with optional downsampling and normalization
- **`extract_features_batch()`** – Multiprocessing batch extraction for entire datasets
- **Optimization techniques**: Vectorized operations, cached computations, Numba JIT compilation

**Key usage**: First step in pipeline; generates `.npy` feature files for each image.

#### [feature_mask_extraction.py](../src/feature_mask_extraction.py)
Augmented feature extraction pipeline:
- **`process_image_with_augmentation()`** – Process single image with optional augmentation
- **`extract_features_with_augmentation()`** – Batch extraction with synchronized augmentation
- **Key features**: Deterministic reproducibility with seeds, critical class oversampling, metadata tracking

**Key usage**: Used for training set to generate augmented samples while maintaining feature-mask consistency.

#### [data_augmentation.py](../src/data_augmentation.py)
Segmentation-specific augmentation:
- **`augment_segmentation_data()`** – Generate augmented image-mask pairs
- **Transformations**: Rotation (90°/180°/270°), zoom, flip, blur, brightness/contrast adjustment
- **Critical class focus**: Oversample regions containing buildings/roads
- **Deterministic**: Uses seeds for reproducibility

**Key usage**: Called by feature extraction to increase training data diversity.

---

### 2.5 Histogram-Based Model

#### [histogram_extraction.py](../src/histogram_extraction.py)
Build class-conditional KDE distributions:
- **`extract_histograms_streaming()`** – Stream through dataset accumulating histograms per class-feature pair
- **Key approach**: Discretize features into bins, apply Gaussian smoothing for KDE approximation
- **Memory efficient**: Processes images one-by-one to avoid loading entire dataset
- **`visualize_kde()`** – Plot KDE distributions for inspection

**Key usage**: Training phase for histogram-based model; generates `.npz` file with KDE data.

#### [histogram_evaluation.py](../src/histogram_evaluation.py)
Inference using histogram-based KDE:
- **`score_image_kde()`** – Compute class probability scores using pre-computed KDE
- **`scores_to_mask()`** – Convert probability scores to segmentation mask
- **`run_histogram_inference()`** – Batch inference on test set
- **`evaluate_predictions()`** – Compute IoU metrics

**Key usage**: Inference phase; applies trained KDE to predict segmentation masks.

---

### 2.6 U-Net Model

#### [training_examples.py](../src/training_examples.py)
Training scripts for all models:
- **`train_unet()`** – Train U-Net with specified features
- **`train_kmeans()`, `train_random_forest()`** – Train alternative models
- **`evaluate_unet()`, `evaluate_kmeans()`, etc.** – Evaluate trained models
- **`feature_ablation_study()`** – Test different feature combinations
- **`compare_all_models()`** – Train and compare all four models
- **`compute_reduced_csv()`** – Generate subsampled datasets for faster experimentation

**Key usage**: Main training entry point; configure model parameters and run training.

#### [inference.py](../src/inference.py)
U-Net inference pipeline:
- **`run_unet_inference()`** – Load model, process test images, save predicted masks
- **Features**: Batch processing, memory-mapped loading, optional post-treatment, resizing to original dimensions

**Key usage**: Apply trained U-Net model to generate predictions on test set.

---

### 2.7 Post-Processing

#### [post_treatement.py](../src/post_treatement.py)
Aggressive post-processing pipeline:
- **`posttreat_pipeline()`** – Main pipeline applying class-specific processing
- **Road processing**: Aggressive topology enforcement, bridging aligned components, extension to borders
- **Building processing**: Geometry regularization, expansion
- **Natural classes**: Noise removal, morphological smoothing
- **Conflict resolution**: Enforce class hierarchy (Building > Road > Water > Woodland > Field)

**Key usage**: Applied after model prediction to improve segmentation quality, especially for roads.

---

### 2.8 Evaluation

#### [evaluation.py](../src/evaluation.py)
Comprehensive evaluation suite:
- **`compute_confusion_matrix()`** – Build confusion matrix from predictions
- **`compute_iou()`, `compute_dice()`** – Calculate IoU and Dice scores
- **`compute_precision_recall()`** – Per-class precision and recall
- **`plot_confusion_matrix()`** – Visualize confusion matrix heatmap
- **`plot_metric_bars()`** – Bar charts for per-class metrics
- **`generate_report()`** – Create text and JSON evaluation reports
- **`evaluate_predictions()`** – Complete evaluation pipeline

**Key usage**: Run after inference to quantify model performance and generate comprehensive reports.

---

### 2.9 Main Pipeline

#### [main.py](../src/main.py)
Orchestrates the complete pipeline:
1. Generate mapping CSVs for train/test/val splits
2. Compute class statistics and select images
3. Extract features with augmentation
4. Train models (U-Net, histogram-based, K-Means, Random Forest)
5. Run inference
6. Apply post-processing
7. Evaluate results

**Key usage**: Single entry point to execute the entire pipeline from raw data to final predictions.