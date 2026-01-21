"""
Evaluation and inference using histogram-based KDE for image segmentation.

This module provides functions to score pixels based on pre-computed KDE
distributions and convert probability maps to segmentation masks.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import os

from src.cste import ClassInfo, DataPath, HistogramConfig
from src.logger import get_logger
from post_treatement import posttreat_pipeline

log = get_logger("histogram_evaluation")


# ============================================================================
# CORE CONSTANTS
# ============================================================================

LOG_EPSILON = 1e-10  #! Numerical stability floor for log operations


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================


def score_image_kde(
    features: np.ndarray,
    kde_data: Dict[str, np.ndarray],
    use_global_prior: bool = False,
    class_temperatures: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute class probability scores for an image using pre-computed KDE.

    Uses numerically stable log-likelihood computation and optional priors.

    Args:
        features: Image features, shape (H, W, F_selected)
        kde_data: Dictionary with keys 'kde', 'bin_edges', 'feature_ids', optionally 'class_priors'
        use_global_prior: If True, incorporate global dataset class priors
        class_temperatures: Optional temperature scaling per class for calibration

    Returns:
        Score matrix of shape (H, W, N) with values in [0, 1]
    """
    h, w, num_features = features.shape
    kde = kde_data["kde"]
    bin_edges = kde_data["bin_edges"]

    num_classes = kde.shape[0]
    num_bins = kde.shape[2]

    #! Initialize log-score matrix
    log_scores = np.zeros((h, w, num_classes), dtype=np.float64)

    #! Flatten features for vectorized processing
    features_flat = features.reshape(-1, num_features)

    #! Compute log-likelihood for each class
    for class_id in range(num_classes):
        log_probs = np.zeros(h * w, dtype=np.float64)

        for feat_idx in range(num_features):
            feat_values = features_flat[:, feat_idx]
            edges = bin_edges[feat_idx]

            #! Clip values to [0, 1]
            feat_values = np.clip(feat_values, 0.0, 1.0)

            #! Find bin indices
            bin_indices = np.digitize(feat_values, edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            #! Look up KDE values
            kde_values = kde[class_id, feat_idx, bin_indices]

            #! Apply epsilon floor and take log
            kde_values = np.maximum(kde_values, LOG_EPSILON)
            log_probs += np.log(kde_values)

        #! Average log probabilities
        log_scores[:, :, class_id] = (log_probs / num_features).reshape(h, w)

    #! Add global prior if enabled
    if use_global_prior and "class_priors" in kde_data:
        class_priors = kde_data["class_priors"]
        log_priors = np.log(np.maximum(class_priors, LOG_EPSILON))
        log_scores += log_priors.reshape(1, 1, num_classes)

    #! Apply temperature scaling if provided
    if class_temperatures is not None:
        log_scores = log_scores / class_temperatures.reshape(1, 1, num_classes)

    #! Convert to probability scores
    log_scores_max = log_scores.max(axis=2, keepdims=True)
    scores = np.exp(log_scores - log_scores_max)

    #! Normalize to [0, 1]
    score_sum = scores.sum(axis=2, keepdims=True)
    score_sum = np.maximum(score_sum, LOG_EPSILON)
    scores = scores / score_sum

    return scores


def scores_to_mask(
    scores: np.ndarray, tie_priority: Optional[List[int]] = None
) -> np.ndarray:
    """
    Convert class probability scores to segmentation mask.

    Args:
        scores: Score matrix, shape (H, W, N)
        tie_priority: Optional class priority order

    Returns:
        Segmentation mask, shape (H, W), dtype int32
    """
    h, w, num_classes = scores.shape

    if tie_priority is None:
        tie_priority = list(range(num_classes))

    #! Initialize mask
    mask = np.zeros((h, w), dtype=np.int32)

    #! Find maximum score for each pixel
    max_scores = scores.max(axis=2)

    #! Assign pixels in priority order
    for class_id in reversed(tie_priority):
        class_wins = scores[:, :, class_id] >= max_scores
        mask[class_wins] = class_id

    return mask


def compute_class_ratios(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute class ratios from a segmentation mask.

    Args:
        mask: Segmentation mask, shape (H, W)
        num_classes: Total number of classes

    Returns:
        Array of class ratios, shape (num_classes,)
    """
    total_pixels = mask.size
    ratios = np.zeros(num_classes, dtype=np.float64)

    for class_id in range(num_classes):
        ratios[class_id] = (mask == class_id).sum() / total_pixels

    return ratios


def apply_adaptive_prior(
    scores: np.ndarray, predicted_ratios: np.ndarray, target_ratios: np.ndarray
) -> np.ndarray:
    """
    Apply adaptive per-image class prior to scores.

    Computes bias term delta_c = log((target_ratio_c + eps) / (predicted_ratio_c + eps))
    and adds it to class scores.

    Args:
        scores: Current score matrix, shape (H, W, N)
        predicted_ratios: Predicted class ratios, shape (N,)
        target_ratios: Target class ratios, shape (N,)

    Returns:
        Adjusted score matrix, shape (H, W, N)
    """
    num_classes = scores.shape[2]

    #! Compute bias term
    delta = np.log(
        np.maximum(target_ratios, LOG_EPSILON)
        / np.maximum(predicted_ratios, LOG_EPSILON)
    )

    #! Add bias to scores (in log space)
    log_scores = np.log(np.maximum(scores, LOG_EPSILON))
    log_scores += delta.reshape(1, 1, num_classes)

    #! Convert back to probability space
    log_scores_max = log_scores.max(axis=2, keepdims=True)
    adjusted_scores = np.exp(log_scores - log_scores_max)

    #! Normalize
    score_sum = adjusted_scores.sum(axis=2, keepdims=True)
    score_sum = np.maximum(score_sum, LOG_EPSILON)
    adjusted_scores = adjusted_scores / score_sum

    return adjusted_scores


def infer_single_image_adaptive(
    features: np.ndarray,
    kde_dict: Dict[str, np.ndarray],
    target_ratios: Optional[np.ndarray] = None,
    num_iterations: int = 2,
    use_global_prior: bool = False,
    class_temperatures: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run adaptive inference on a single image with iterative ratio correction.

    Args:
        features: Image features, shape (H, W, F)
        kde_dict: KDE data dictionary
        target_ratios: Optional target class ratios. If None, uses global priors if available
        num_iterations: Number of adaptive iterations (1-3 recommended)
        use_global_prior: Whether to use global dataset prior
        class_temperatures: Optional temperature scaling per class

    Returns:
        mask: Final segmentation mask, shape (H, W)
        scores: Final score matrix, shape (H, W, N)
    """
    num_classes = kde_dict["kde"].shape[0]

    #! Set target ratios
    if target_ratios is None:
        if "class_priors" in kde_dict:
            target_ratios = kde_dict["class_priors"]
        else:
            target_ratios = np.ones(num_classes) / num_classes

    #! Initial inference
    scores = score_image_kde(
        features,
        kde_dict,
        use_global_prior=use_global_prior,
        class_temperatures=class_temperatures,
    )

    #! Iterative adaptive correction
    for iteration in range(num_iterations):
        #! Generate current mask
        mask = scores_to_mask(scores)

        #! Compute predicted ratios
        predicted_ratios = compute_class_ratios(mask, num_classes)

        #! Apply adaptive prior
        scores = apply_adaptive_prior(scores, predicted_ratios, target_ratios)

    #! Final mask
    mask = scores_to_mask(scores)

    return mask, scores


# ============================================================================
# BATCH INFERENCE
# ============================================================================


def run_histogram_inference(
    csv_path: str,
    kde_path: str,
    output_dir: str,
    feature_ids: Optional[List[int]] = None,
    save_scores: bool = False,
    batch_log_interval: int = 20,
    output_size: Optional[Tuple[int, int]] = (512, 512),
    postreatment: bool = False,
    use_global_prior: bool = False,
    use_adaptive_prior: bool = False,
    adaptive_iterations: int = 2,
    class_temperatures: Optional[List[float]] = None,
    target_class_ratios: Optional[List[float]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run inference on all images in CSV using histogram-based KDE.

    Args:
        csv_path: Path to CSV with 'img_id', 'feature_path', 'mask_path' columns
        kde_path: Path to .npz file with pre-computed KDE
        output_dir: Directory to save predicted masks
        feature_ids: Optional list of feature indices
        save_scores: If True, also save raw score matrices
        batch_log_interval: Log progress every N images
        output_size: Optional output size for masks
        postreatment: If True, apply post-processing pipeline
        use_global_prior: If True, incorporate global dataset class priors
        use_adaptive_prior: If True, use adaptive per-image prior correction
        adaptive_iterations: Number of adaptive iterations (1-3 recommended)
        class_temperatures: Optional temperature scaling per class [T1, T2, ...]
        target_class_ratios: Optional target class ratios [r1, r2, ...]
        **kwargs: Additional arguments passed to posttreat_pipeline

    Returns:
        DataFrame with columns: img_id, prediction_path, (optional) score_path
    """
    log.info("=" * 70)
    log.info("HISTOGRAM-BASED KDE INFERENCE")
    log.info("=" * 70)
    log.info(f"CSV path: {csv_path}")
    log.info(f"KDE path: {kde_path}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Use global prior: {use_global_prior}")
    log.info(f"Use adaptive prior: {use_adaptive_prior}")
    if use_adaptive_prior:
        log.info(f"Adaptive iterations: {adaptive_iterations}")
    if class_temperatures is not None:
        log.info(f"Class temperatures: {class_temperatures}")
    log.info("=" * 70)

    #! Load KDE data
    kde_data = np.load(kde_path, allow_pickle=True)
    kde_dict = {
        "kde": kde_data["kde"],
        "bin_edges": kde_data["bin_edges"],
        "feature_ids": kde_data["feature_ids"].tolist(),
    }

    if "class_priors" in kde_data:
        kde_dict["class_priors"] = kde_data["class_priors"]

    if feature_ids is None:
        feature_ids = kde_dict["feature_ids"]

    log.info(f"Using features: {feature_ids}")

    #! Verify feature consistency
    if feature_ids != kde_dict["feature_ids"]:
        log.warning("Feature IDs do not match KDE data - make sure this is intended!")

    #! Prepare temperature array
    temp_array = None
    if class_temperatures is not None:
        temp_array = np.array(class_temperatures, dtype=np.float64)

    #! Prepare target ratios
    target_ratios = None
    if target_class_ratios is not None:
        target_ratios = np.array(target_class_ratios, dtype=np.float64)
        target_ratios = target_ratios / target_ratios.sum()

    #! Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if save_scores:
        score_dir = os.path.join(output_dir, "scores")
        os.makedirs(score_dir, exist_ok=True)

    #! Read CSV
    df = pd.read_csv(csv_path)
    total_images = len(df)
    log.info(f"Processing {total_images} images...")

    #! Process images
    results = []
    processed_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=total_images, desc="Running inference"):
        try:
            img_id = row["img_id"]

            #! Load features
            features = np.load(row["feature_path"])
            features_selected = features[..., feature_ids]

            #! Run inference
            if use_adaptive_prior:
                mask, scores = infer_single_image_adaptive(
                    features_selected,
                    kde_dict,
                    target_ratios=target_ratios,
                    num_iterations=adaptive_iterations,
                    use_global_prior=use_global_prior,
                    class_temperatures=temp_array,
                )
            else:
                scores = score_image_kde(
                    features_selected,
                    kde_dict,
                    use_global_prior=use_global_prior,
                    class_temperatures=temp_array,
                )
                mask = scores_to_mask(scores)

            #! Resize mask to original size if needed
            if output_size is not None:
                pred_mask = cv2.resize(
                    mask, output_size, interpolation=cv2.INTER_NEAREST
                )
            else:
                pred_mask = mask

            #! Post-treatment if requested
            if postreatment:
                pred_mask = posttreat_pipeline(pred_mask, **kwargs)

            #! Save mask as PNG
            mask_path = os.path.join(output_dir, f"{img_id}.png")
            mask_uint8 = pred_mask.astype("uint8")
            cv2.imwrite(mask_path, mask_uint8)

            result_record = {"img_id": img_id, "prediction_path": mask_path}

            #! Save scores if requested
            if save_scores:
                score_path = os.path.join(score_dir, f"{img_id}_scores.npy")
                np.save(score_path, scores)
                result_record["score_path"] = score_path

            results.append(result_record)
            processed_count += 1

            #! Periodic logging
            if (processed_count % batch_log_interval) == 0:
                log.info(f"Processed {processed_count}/{total_images} images...")

        except Exception as e:
            error_count += 1
            log.warning(f"Error processing {row['img_id']}: {str(e)}")
            continue

    log.info(f"Inference complete: {processed_count} successful, {error_count} errors")

    #! Save results CSV
    df_results = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "inference_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    log.info(f"Saved results to: {results_csv_path}")

    return df_results


# ============================================================================
# SINGLE IMAGE INFERENCE
# ============================================================================


def predict_single_image(
    feature_path: str,
    kde_path: str,
    feature_ids: Optional[List[int]] = None,
    return_scores: bool = False,
    use_global_prior: bool = False,
    use_adaptive_prior: bool = False,
    adaptive_iterations: int = 2,
    class_temperatures: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run inference on a single image.

    Args:
        feature_path: Path to .npy file with features (H, W, F)
        kde_path: Path to .npz file with pre-computed KDE
        feature_ids: Optional list of feature indices
        return_scores: If True, also return score matrix
        use_global_prior: If True, incorporate global dataset class priors
        use_adaptive_prior: If True, use adaptive per-image prior correction
        adaptive_iterations: Number of adaptive iterations
        class_temperatures: Optional temperature scaling per class

    Returns:
        mask: Predicted segmentation mask (H, W)
        scores: Optional score matrix (H, W, N) if return_scores=True
    """
    #! Load KDE
    kde_data = np.load(kde_path, allow_pickle=True)
    kde_dict = {
        "kde": kde_data["kde"],
        "bin_edges": kde_data["bin_edges"],
        "feature_ids": kde_data["feature_ids"].tolist(),
    }

    if "class_priors" in kde_data:
        kde_dict["class_priors"] = kde_data["class_priors"]

    if feature_ids is None:
        feature_ids = kde_dict["feature_ids"]

    #! Prepare temperature array
    temp_array = None
    if class_temperatures is not None:
        temp_array = np.array(class_temperatures, dtype=np.float64)

    #! Load features
    features = np.load(feature_path)
    features_selected = features[..., feature_ids]

    #! Run inference
    if use_adaptive_prior:
        mask, scores = infer_single_image_adaptive(
            features_selected,
            kde_dict,
            num_iterations=adaptive_iterations,
            use_global_prior=use_global_prior,
            class_temperatures=temp_array,
        )
    else:
        scores = score_image_kde(
            features_selected,
            kde_dict,
            use_global_prior=use_global_prior,
            class_temperatures=temp_array,
        )
        mask = scores_to_mask(scores)

    if return_scores:
        return mask, scores
    else:
        return mask, None


# ============================================================================
# EVALUATION HELPERS
# ============================================================================


def compute_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int = ClassInfo.NUM_CLASSES,
) -> Dict[int, float]:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred_mask: Predicted mask (H, W)
        true_mask: Ground truth mask (H, W)
        num_classes: Total number of classes

    Returns:
        Dictionary mapping class_id -> IoU score
    """
    iou_scores = {}

    for class_id in range(num_classes):
        pred_class = pred_mask == class_id
        true_class = true_mask == class_id

        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()

        if union > 0:
            iou_scores[class_id] = intersection / union
        else:
            iou_scores[class_id] = 0.0

    return iou_scores


def evaluate_predictions(
    csv_path: str, prediction_dir: str, num_classes: int = ClassInfo.NUM_CLASSES
) -> pd.DataFrame:
    """
    Evaluate predictions against ground truth masks.

    Args:
        csv_path: Path to CSV with 'img_id', 'mask_path' columns
        prediction_dir: Directory containing predicted masks
        num_classes: Total number of classes

    Returns:
        DataFrame with per-image IoU scores
    """
    log.info("Evaluating predictions...")

    df = pd.read_csv(csv_path)
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            img_id = row["img_id"]

            #! Load ground truth
            true_mask = np.load(row["mask_path"])

            #! Load prediction
            pred_path = os.path.join(prediction_dir, f"{img_id}.npy")
            pred_mask = np.load(pred_path)

            #! Compute IoU
            iou_scores = compute_iou(pred_mask, true_mask, num_classes)

            record = {"img_id": img_id}
            for class_id, iou in iou_scores.items():
                class_name = ClassInfo.CLASS_NAMES.get(class_id, f"class_{class_id}")
                record[f"iou_{class_name}"] = iou

            record["mean_iou"] = np.mean(list(iou_scores.values()))

            results.append(record)

        except Exception as e:
            log.warning(f"Error evaluating {row['img_id']}: {str(e)}")
            continue

    df_results = pd.DataFrame(results)

    #! Log summary
    log.info("=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)
    for col in df_results.columns:
        if col.startswith("iou_"):
            mean_val = df_results[col].mean()
            log.info(f"{col}: {mean_val:.4f}")
    log.info(f"Overall mean IoU: {df_results['mean_iou'].mean():.4f}")
    log.info("=" * 70)

    return df_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from src.cste import FeatureInfo

    log.info("Example: Adaptive inference with global prior")

    results_df = run_histogram_inference(
        csv_path=DataPath.CSV_FEATURE_MASK_MAPPING_TEST,
        kde_path=r"data/models/histogram/histogram_kde_streaming.npz",
        output_dir=r"data/predictions/histogram_adaptive_inference",
        feature_ids=FeatureInfo.FEATURE_ALL,
        save_scores=False,
        batch_log_interval=20,
        output_size=(512, 512),
        postreatment=False,
        use_global_prior=True,
        use_adaptive_prior=True,
        adaptive_iterations=2,
    )

    log.info("Inference complete!")
