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
# SCORING FUNCTIONS
# ============================================================================

def score_image_kde(
    features: np.ndarray,
    kde_data: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute class probability scores for an image using pre-computed KDE.
    
    This function uses simple linear interpolation to look up probability
    densities from the discretized KDE for each pixel and feature.
    
    Args:
        features: Image features, shape (H, W, F_selected)
                  Must match the number of features in kde_data
        kde_data: Dictionary with keys 'kde', 'bin_edges', 'feature_ids'
                  from histogram_extraction.extract_histograms_streaming
        
    Returns:
        Score matrix of shape (H, W, N) with values in [0, 1]
        Higher scores indicate higher probability of belonging to each class
    """
    h, w, num_features = features.shape
    kde = kde_data['kde']  # Shape: (num_classes, num_features, num_bins)
    bin_edges = kde_data['bin_edges']  # Shape: (num_features, num_bins + 1)
    
    num_classes = kde.shape[0]
    num_bins = kde.shape[2]
    
    #! Initialize score matrix
    scores = np.zeros((h, w, num_classes), dtype=np.float32)
    
    #! Flatten features for vectorized processing
    features_flat = features.reshape(-1, num_features)  # (H*W, num_features)
    
    #! Compute scores for each class
    for class_id in range(num_classes):
        # Accumulate log probabilities (more numerically stable)
        log_probs = np.zeros(h * w, dtype=np.float64)
        
        for feat_idx in range(num_features):
            feat_values = features_flat[:, feat_idx]  # (H*W,)
            
            #! Get bin edges for this feature
            edges = bin_edges[feat_idx]
            
            #! Clip values to [0, 1] to handle edge cases
            feat_values = np.clip(feat_values, 0.0, 1.0)
            
            #! Find bin indices using digitize
            # digitize returns indices in [1, num_bins], we need [0, num_bins-1]
            bin_indices = np.digitize(feat_values, edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            
            #! Look up KDE values
            kde_values = kde[class_id, feat_idx, bin_indices]
            
            #! Add small epsilon to avoid log(0)
            kde_values = np.maximum(kde_values, 1e-10)
            
            #! Accumulate log probabilities
            log_probs += np.log(kde_values)
        
        #! Convert back to probabilities
        # Use exp of mean log prob to avoid numerical overflow
        scores_flat = np.exp(log_probs / num_features)
        scores[:, :, class_id] = scores_flat.reshape(h, w)
    
    #! Normalize scores to [0, 1] range
    # This makes scores comparable across images
    score_max = scores.max(axis=2, keepdims=True)
    score_max = np.maximum(score_max, 1e-10)  # Avoid division by zero
    scores = scores / score_max
    
    return scores


def scores_to_mask(
    scores: np.ndarray,
    tie_priority: Optional[List[int]] = None
) -> np.ndarray:
    """
    Convert class probability scores to segmentation mask.
    
    For each pixel, selects the class with highest score.
    In case of ties, uses priority ordering (lower index = higher priority).
    
    Args:
        scores: Score matrix, shape (H, W, N)
        tie_priority: Optional class priority order. If None, uses [0, 1, ..., N-1]
                      Lower index in list = higher priority in ties
        
    Returns:
        Segmentation mask, shape (H, W), dtype int32
        Each pixel contains class ID in range [0, N-1]
    """
    h, w, num_classes = scores.shape
    
    if tie_priority is None:
        #! Default: class 0 has highest priority, class N-1 lowest
        tie_priority = list(range(num_classes))
    
    #! Initialize mask
    mask = np.zeros((h, w), dtype=np.int32)
    
    #! Find maximum score for each pixel
    max_scores = scores.max(axis=2)  # (H, W)
    
    #! For each class in priority order, assign pixels where it has max score
    for class_id in reversed(tie_priority):
        # This class wins where it has the max score
        # By processing in reverse priority, higher priority classes overwrite
        class_wins = (scores[:, :, class_id] >= max_scores)
        mask[class_wins] = class_id
    
    return mask


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
    **kwargs
) -> pd.DataFrame:
    """
    Run inference on all images in CSV using histogram-based KDE.
    
    Args:
        csv_path: Path to CSV with 'img_id', 'feature_path', 'mask_path' columns
        kde_path: Path to .npz file with pre-computed KDE
        output_dir: Directory to save predicted masks
        feature_ids: Optional list of feature indices. If None, uses all from KDE
        save_scores: If True, also save raw score matrices
        batch_log_interval: Log progress every N images
        
    Returns:
        DataFrame with columns: img_id, prediction_path, (optional) score_path
    """
    log.info("=" * 70)
    log.info("HISTOGRAM-BASED KDE INFERENCE")
    log.info("=" * 70)
    log.info(f"CSV path: {csv_path}")
    log.info(f"KDE path: {kde_path}")
    log.info(f"Output directory: {output_dir}")
    log.info("=" * 70)
    
    #! Load KDE data
    kde_data = np.load(kde_path, allow_pickle=True)
    kde_dict = {
        'kde': kde_data['kde'],
        'bin_edges': kde_data['bin_edges'],
        'feature_ids': kde_data['feature_ids'].tolist()
    }
    
    if feature_ids is None:
        feature_ids = kde_dict['feature_ids']
    
    log.info(f"Using features: {feature_ids}")
    
    #! Verify feature consistency
    if feature_ids != kde_dict['feature_ids']:
        log.warning("Feature IDs do not match KDE data - make sure this is intended!")
    
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
            img_id = row['img_id']
            
            #! Load features
            features = np.load(row['feature_path'])  # (H, W, F)
            
            #! Select features strictly
            features_selected = features[..., feature_ids]  # (H, W, num_features)
            
            #! Compute scores
            scores = score_image_kde(features_selected, kde_dict)
            
            #! Convert to mask
            mask = scores_to_mask(scores)

            #! Resize mask to original size if needed
            if output_size is not None:
                pred_mask = cv2.resize(
                    mask,
                    output_size,
                    interpolation=cv2.INTER_NEAREST
                )
            
            #! Post-treatment if requested
            if postreatment:
                pred_mask = posttreat_pipeline(pred_mask, **kwargs)
            
            #! Save mask as PNG
            mask_path = os.path.join(output_dir, f"{img_id}.png")
            # Convert mask to uint8 if necessary
            mask_uint8 = pred_mask.astype('uint8')
            cv2.imwrite(mask_path, mask_uint8)
            
            result_record = {
                'img_id': img_id,
                'prediction_path': mask_path
            }
            
            #! Save scores if requested
            if save_scores:
                score_path = os.path.join(score_dir, f"{img_id}_scores.npy")
                np.save(score_path, scores)
                result_record['score_path'] = score_path
            
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
    return_scores: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run inference on a single image.
    
    Args:
        feature_path: Path to .npy file with features (H, W, F)
        kde_path: Path to .npz file with pre-computed KDE
        feature_ids: Optional list of feature indices
        return_scores: If True, also return score matrix
        
    Returns:
        mask: Predicted segmentation mask (H, W)
        scores: Optional score matrix (H, W, N) if return_scores=True
    """
    #! Load KDE
    kde_data = np.load(kde_path, allow_pickle=True)
    kde_dict = {
        'kde': kde_data['kde'],
        'bin_edges': kde_data['bin_edges'],
        'feature_ids': kde_data['feature_ids'].tolist()
    }
    
    if feature_ids is None:
        feature_ids = kde_dict['feature_ids']
    
    #! Load features
    features = np.load(feature_path)
    features_selected = features[..., feature_ids]
    
    #! Compute scores
    scores = score_image_kde(features_selected, kde_dict)
    
    #! Convert to mask
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
    num_classes: int = ClassInfo.NUM_CLASSES
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
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        
        if union > 0:
            iou_scores[class_id] = intersection / union
        else:
            iou_scores[class_id] = 0.0
    
    return iou_scores


def evaluate_predictions(
    csv_path: str,
    prediction_dir: str,
    num_classes: int = ClassInfo.NUM_CLASSES
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
            img_id = row['img_id']
            
            #! Load ground truth
            true_mask = np.load(row['mask_path'])
            
            #! Load prediction
            pred_path = os.path.join(prediction_dir, f"{img_id}.npy")
            pred_mask = np.load(pred_path)
            
            #! Compute IoU
            iou_scores = compute_iou(pred_mask, true_mask, num_classes)
            
            record = {'img_id': img_id}
            for class_id, iou in iou_scores.items():
                class_name = ClassInfo.CLASS_NAMES.get(class_id, f"class_{class_id}")
                record[f"iou_{class_name}"] = iou
            
            # Mean IoU
            record['mean_iou'] = np.mean(list(iou_scores.values()))
            
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
        if col.startswith('iou_'):
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
    
    # Example 1: Run batch inference
    log.info("Example 1: Batch inference on test set")
    
    results_df = run_histogram_inference(
        csv_path=DataPath.CSV_FEATURE_MASK_MAPPING_TEST,
        kde_path=r"data/models/histogram/histogram_kde_streaming.npz",
        output_dir=DataPath.HISTOGRAM_DIR,
        feature_ids=FeatureInfo.FEATURE_ALL,
        save_scores=False,
        batch_log_interval=20,
        output_size=(512, 512),
        postreatment=False
    )
    
    # # Example 2: Evaluate predictions
    # log.info("\nExample 2: Evaluate predictions")
    
    # eval_df = evaluate_predictions(
    #     csv_path=DataPath.CSV_FEATURE_MASK_MAPPING_TEST,
    #     prediction_dir=f"{DataPath.RESULT_PATH}histogram_predictions/",
    #     num_classes=ClassInfo.NUM_CLASSES
    # )
    
    # # Save evaluation results
    # eval_csv_path = f"{DataPath.RESULT_PATH}histogram_evaluation.csv"
    # eval_df.to_csv(eval_csv_path, index=False)
    # log.info(f"Saved evaluation results to: {eval_csv_path}")
    
    # # Example 3: Single image prediction
    # log.info("\nExample 3: Single image prediction")
    
    # # Get first test image
    # df_test = pd.read_csv(DataPath.CSV_FEATURE_MASK_MAPPING_TEST)
    # first_img = df_test.iloc[0]
    
    # mask, scores = predict_single_image(
    #     feature_path=first_img['feature_path'],
    #     kde_path=f"{DataPath.MODEL_DIR}histogram_kde_selected.npz",
    #     feature_ids=FeatureInfo.FEATURE_UNET_SELECTION,
    #     return_scores=True
    # )
    
    # log.info(f"Predicted mask shape: {mask.shape}")
    # log.info(f"Score matrix shape: {scores.shape}")
    # log.info(f"Unique classes in prediction: {np.unique(mask)}")
    
    # log.info("Examples complete!")
