"""
Histogram-based KDE extraction for image segmentation.

This module implements streaming histogram extraction from labeled datasets
to build class-conditional probability distributions for each feature.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
import os

from src.cste import ClassInfo, HistogramConfig, FeatureInfo, DataPath
from src.logger import get_logger

log = get_logger("histogram_extraction")


# ============================================================================
# HISTOGRAM CONFIGURATION
# ============================================================================


class HistogramConfig:
    """Configuration for histogram-based KDE."""

    NUM_BINS: int = 256  # Number of bins for histogram discretization
    SMOOTHING_SIGMA: float = 2.0  # Gaussian smoothing sigma for KDE approximation


# ============================================================================
# HISTOGRAM EXTRACTION
# ============================================================================


def extract_histograms_streaming(
    csv_path: str,
    feature_ids: Optional[List[int]] = FeatureInfo.FEATURE_ALL,
    num_bins: int = HistogramConfig.NUM_BINS,
    sigma: float = HistogramConfig.SMOOTHING_SIGMA,
    num_classes: int = ClassInfo.NUM_CLASSES,
    output_path: Optional[str] = DataPath.MODEL_DIR
    + r"/histogram/histogram_kde_streaming.npz",
    batch_log_interval: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Extract histograms for each class and feature using streaming approach.

    This function processes images one at a time to minimize memory usage,
    accumulating histogram counts across the entire dataset.

    Args:
        csv_path: Path to CSV with columns 'img_id', 'feature_path', 'mask_path'
        feature_ids: List of feature indices to extract. If None, use all features.
        num_bins: Number of bins for histogram discretization (fixed)
        sigma: Standard deviation for Gaussian smoothing (approximate KDE)
        num_classes: Total number of segmentation classes
        output_path: Optional path to save histograms. If None, returns without saving.
        batch_log_interval: Log progress every N images

    Returns:
        Dictionary containing:
            - 'histograms': Raw histogram counts, shape (num_classes, num_features, num_bins)
            - 'kde': Smoothed KDE approximation, shape (num_classes, num_features, num_bins)
            - 'bin_edges': Bin edges for each feature, shape (num_features, num_bins + 1)
            - 'feature_ids': List of feature indices used
            - 'num_samples': Number of pixels per class, shape (num_classes,)
    """
    log.info("=" * 70)
    log.info("HISTOGRAM-BASED KDE EXTRACTION (STREAMING)")
    log.info("=" * 70)
    log.info(f"CSV path: {csv_path}")
    log.info(f"Feature selection: {feature_ids if feature_ids else 'All features'}")
    log.info(f"Number of bins: {num_bins}")
    log.info(f"Smoothing sigma: {sigma}")
    log.info(f"Number of classes: {num_classes}")
    log.info("=" * 70)

    #! Read CSV mapping
    df = pd.read_csv(csv_path)
    total_images = len(df)

    if total_images == 0:
        raise ValueError(f"No images found in CSV: {csv_path}")

    log.info(f"Found {total_images} images to process")

    #! Determine number of features from first image
    first_feature_path = df.iloc[0]["feature_path"]
    first_features = np.load(first_feature_path)

    if feature_ids is None:
        # Use all features
        num_features = first_features.shape[-1]
        feature_ids = list(range(num_features))
        log.info(f"Using all {num_features} features")
    else:
        num_features = len(feature_ids)
        log.info(f"Using {num_features} selected features: {feature_ids}")

    #! Initialize accumulators
    # Histograms: (num_classes, num_features, num_bins)
    histograms = np.zeros((num_classes, num_features, num_bins), dtype=np.float64)

    # Track number of samples per class for normalization
    num_samples = np.zeros(num_classes, dtype=np.int64)

    # Bin edges are fixed at [0, 1] since features are normalized
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    log.info("Starting streaming histogram extraction...")

    #! Stream through images
    processed_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=total_images, desc="Processing images"):
        try:
            #! Load features and mask
            features = np.load(row["feature_path"])  # Shape: (H, W, F)
            mask = np.load(row["mask_path"])  # Shape: (H, W)

            #! Select features strictly
            features_selected = features[
                ..., feature_ids
            ]  # Shape: (H, W, num_features)

            #! Flatten for histogram computation
            h, w, f = features_selected.shape
            features_flat = features_selected.reshape(
                -1, num_features
            )  # (H*W, num_features)
            mask_flat = mask.reshape(-1)  # (H*W,)

            #! Accumulate histograms for each class
            for class_id in range(num_classes):
                # Get pixels belonging to this class
                class_mask = mask_flat == class_id
                class_pixels = features_flat[class_mask]  # (N_class, num_features)

                if class_pixels.shape[0] == 0:
                    continue

                # Update sample count
                num_samples[class_id] += class_pixels.shape[0]

                #! Compute histograms for each feature
                for feat_idx in range(num_features):
                    feat_values = class_pixels[:, feat_idx]

                    # Accumulate histogram counts
                    counts, _ = np.histogram(feat_values, bins=bin_edges)
                    histograms[class_id, feat_idx, :] += counts

            processed_count += 1

        except Exception as e:
            error_count += 1
            log.warning(f"Error processing {row['img_id']}: {str(e)}")
            continue

    log.info(f"Finished processing: {processed_count} successful, {error_count} errors")

    #! Apply Gaussian smoothing to approximate KDE
    log.info("Applying Gaussian smoothing to histograms...")
    kde = np.zeros_like(histograms)

    for class_id in range(num_classes):
        for feat_idx in range(num_features):
            # Smooth along bin axis
            smoothed = gaussian_filter1d(
                histograms[class_id, feat_idx, :], sigma=sigma, mode="nearest"
            )
            kde[class_id, feat_idx, :] = smoothed

    #! Normalize KDE to sum to 1 for each class-feature pair
    for class_id in range(num_classes):
        for feat_idx in range(num_features):
            total = kde[class_id, feat_idx, :].sum()
            if total > 0:
                kde[class_id, feat_idx, :] /= total

    log.info("KDE normalization complete")

    #! Prepare output dictionary
    result = {
        "histograms": histograms,
        "kde": kde,
        "bin_edges": np.tile(
            bin_edges, (num_features, 1)
        ),  # (num_features, num_bins + 1)
        "feature_ids": feature_ids,
        "num_samples": num_samples,
    }

    #! Save if output path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **result)
        log.info(f"Saved histograms and KDE to: {output_path}")

    #! Summary statistics
    log.info("=" * 70)
    log.info("EXTRACTION SUMMARY")
    log.info("=" * 70)
    log.info(f"Total pixels per class:")
    for class_id in range(num_classes):
        class_name = ClassInfo.CLASS_NAMES.get(class_id, f"Class_{class_id}")
        log.info(f"  {class_name}: {num_samples[class_id]:,} pixels")
    log.info("=" * 70)

    return result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_histograms(path: str) -> Dict[str, np.ndarray]:
    """
    Load pre-computed histograms and KDE from disk.

    Args:
        path: Path to .npz file saved by extract_histograms_streaming

    Returns:
        Dictionary with keys: 'histograms', 'kde', 'bin_edges', 'feature_ids', 'num_samples'
    """
    data = np.load(path, allow_pickle=True)

    result = {
        "histograms": data["histograms"],
        "kde": data["kde"],
        "bin_edges": data["bin_edges"],
        "feature_ids": data["feature_ids"].tolist(),
        "num_samples": data["num_samples"],
    }

    log.info(f"Loaded histograms from: {path}")
    log.info(f"  Shape: {result['kde'].shape}")
    log.info(f"  Features: {result['feature_ids']}")

    return result


def visualize_kde(
    kde_data: Dict[str, np.ndarray],
    feature_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize KDE distributions for all classes for a specific feature.

    Args:
        kde_data: Dictionary returned by extract_histograms_streaming or load_histograms
        feature_idx: Index of feature to visualize (in selected features)
        save_path: Optional path to save figure
    """

    kde = kde_data["kde"]
    bin_edges = kde_data["bin_edges"][feature_idx]
    feature_ids = kde_data["feature_ids"]

    num_classes = kde.shape[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(12, 6))

    for class_id in range(num_classes):
        class_name = ClassInfo.CLASS_NAMES.get(class_id, f"Class_{class_id}")
        plt.plot(
            bin_centers, kde[class_id, feature_idx, :], label=class_name, linewidth=2
        )

    plt.xlabel("Feature Value (Normalized)")
    plt.ylabel("Probability Density")
    plt.title(f"KDE for Feature {feature_ids[feature_idx]}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved KDE visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from src.cste import DataPath, FeatureInfo

    # Example 1: Extract histograms for selected features (RGB + NDVI + corners)
    log.info("Example 1: Extract histograms with selected features")

    # kde_data = extract_histograms_streaming(
    #     csv_path=DataPath.CSV_FEATURE_MASK_MAPPING_TRAIN,
    # )

    # Example 2: Load and visualize
    log.info("\nExample 2: Load and visualize KDE")

    loaded_data = load_histograms(
        f"{DataPath.MODEL_DIR}histogram/histogram_kde_streaming.npz"
    )

    # Visualize first feature (Red channel)
    for feature in FeatureInfo.FEATURE_ALL:
        visualize_kde(
            kde_data=loaded_data,
            feature_idx=feature,
            save_path=f"{DataPath.REPORT_PATH}kde_feature_{feature}.png",
        )

    log.info("Examples complete!")
