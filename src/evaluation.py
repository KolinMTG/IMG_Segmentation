"""
Evaluation metrics for imbalanced multi-class semantic segmentation.

This module implements IoU, Dice, and Recall metrics with macro-averaging
to ensure equal contribution from all classes regardless of pixel frequency.
"""

import numpy as np
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from functools import partial

from typing import Dict, List, Optional, Tuple
import csv
import os
from datetime import datetime
from PIL import Image

from io_utils import list_dir_endwith, load_image
from cste import ResultPath, DataPath, GeneralConfig

from logger import get_logger
log = get_logger(__name__)


# ============================================================================
# PER-CLASS METRICS
# ============================================================================

def compute_per_class_iou(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Compute Intersection over Union (IoU) for each class.
    
    IoU = TP / (TP + FP + FN) = Intersection / Union
    
    Args:
        predicted: Predicted segmentation mask (H, W) with integer class IDs
        ground_truth: Ground truth segmentation mask (H, W) with integer class IDs
        classes: List of class IDs to evaluate. If None, uses all classes
                present in either predicted or ground_truth
                
    Returns:
        Dictionary {class_id: iou_score}
        - iou_score in [0, 1], higher is better
        - Returns 1.0 if both masks have zero pixels for that class (perfect agreement)
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1]])
        >>> gt = np.array([[0, 1], [1, 1]])
        >>> iou = compute_per_class_iou(pred, gt)
        >>> # Class 0: TP=1, FP=1, FN=1 -> IoU=1/3
        >>> # Class 1: TP=2, FP=0, FN=0 -> IoU=1.0
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}")
    
    # Determine classes to evaluate
    if classes is None:
        classes = np.unique(np.concatenate([
            np.unique(predicted),
            np.unique(ground_truth)
        ]))
        # Exclude negative classes (background/void)
        classes = classes[classes >= 0]
    
    iou_scores = {}
    
    for cls in classes:
        # Binary masks for this class
        pred_mask = (predicted == cls)
        gt_mask = (ground_truth == cls)
        
        # Compute intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # Handle edge case: both masks empty for this class
        if union == 0:
            iou = 1.0  # Perfect agreement (both absent)
        else:
            iou = float(intersection) / float(union)
        
        iou_scores[cls] = iou
    
    return iou_scores


def compute_per_class_dice(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Compute Dice coefficient (F1-score) for each class.
    
    Dice = 2 × TP / (2 × TP + FP + FN) = 2 × Intersection / (Pred + GT)
    
    Args:
        predicted: Predicted segmentation mask (H, W) with integer class IDs
        ground_truth: Ground truth segmentation mask (H, W) with integer class IDs
        classes: List of class IDs to evaluate. If None, uses all classes
                
    Returns:
        Dictionary {class_id: dice_score}
        - dice_score in [0, 1], higher is better
        - Returns 1.0 if both masks have zero pixels for that class
        
    Note:
        Dice is related to IoU: Dice = 2*IoU / (1+IoU)
        Dice scores are always >= IoU scores
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1]])
        >>> gt = np.array([[0, 1], [1, 1]])
        >>> dice = compute_per_class_dice(pred, gt)
        >>> # Class 0: 2*1/(2+2) = 0.5
        >>> # Class 1: 2*2/(3+2) = 0.8
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}")
    
    if classes is None:
        classes = np.unique(np.concatenate([
            np.unique(predicted),
            np.unique(ground_truth)
        ]))
        classes = classes[classes >= 0]
    
    dice_scores = {}
    
    for cls in classes:
        pred_mask = (predicted == cls)
        gt_mask = (ground_truth == cls)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        pred_pixels = pred_mask.sum()
        gt_pixels = gt_mask.sum()
        
        # Handle edge case: both masks empty
        if pred_pixels + gt_pixels == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_pixels + gt_pixels)
        
        dice_scores[cls] = dice
    
    return dice_scores


def compute_per_class_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Compute Recall (Sensitivity, True Positive Rate) for each class.
    
    Recall = TP / (TP + FN) = TP / Total_GT
    
    Args:
        predicted: Predicted segmentation mask (H, W) with integer class IDs
        ground_truth: Ground truth segmentation mask (H, W) with integer class IDs
        classes: List of class IDs to evaluate. If None, uses all classes
                
    Returns:
        Dictionary {class_id: recall_score}
        - recall_score in [0, 1], higher is better
        - Returns NaN if class not present in ground truth
        
    Note:
        Recall answers: "Of all ground truth pixels of class C, 
        what percentage did we correctly predict?"
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1]])
        >>> gt = np.array([[0, 1], [1, 1]])
        >>> recall = compute_per_class_recall(pred, gt)
        >>> # Class 0: TP=1, FN=1, Recall=0.5
        >>> # Class 1: TP=2, FN=0, Recall=1.0
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}")
    
    if classes is None:
        classes = np.unique(np.concatenate([
            np.unique(predicted),
            np.unique(ground_truth)
        ]))
        classes = classes[classes >= 0]
    
    recall_scores = {}
    
    for cls in classes:
        pred_mask = (predicted == cls)
        gt_mask = (ground_truth == cls)
        
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        total_gt = gt_mask.sum()
        
        # Handle edge case: class not in ground truth
        if total_gt == 0:
            recall = np.nan  # Undefined
        else:
            recall = float(true_positives) / float(total_gt)
        
        recall_scores[cls] = recall
    
    return recall_scores


def compute_per_class_precision(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Compute Precision (Positive Predictive Value) for each class.
    
    Precision = TP / (TP + FP) = TP / Total_Pred
    
    Args:
        predicted: Predicted segmentation mask (H, W) with integer class IDs
        ground_truth: Ground truth segmentation mask (H, W) with integer class IDs
        classes: List of class IDs to evaluate. If None, uses all classes
                
    Returns:
        Dictionary {class_id: precision_score}
        - precision_score in [0, 1], higher is better
        - Returns NaN if class not present in prediction
        
    Note:
        Precision answers: "Of all pixels predicted as class C,
        what percentage were correct?"
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}")
    
    if classes is None:
        classes = np.unique(np.concatenate([
            np.unique(predicted),
            np.unique(ground_truth)
        ]))
        classes = classes[classes >= 0]
    
    precision_scores = {}
    
    for cls in classes:
        pred_mask = (predicted == cls)
        gt_mask = (ground_truth == cls)
        
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        total_pred = pred_mask.sum()
        
        if total_pred == 0:
            precision = np.nan
        else:
            precision = float(true_positives) / float(total_pred)
        
        precision_scores[cls] = precision
    
    return precision_scores


# ============================================================================
# MACRO-AVERAGED AGGREGATION
# ============================================================================

def macro_average(
    per_class_scores: Dict[int, float],
    ignore_nan: bool = True
) -> float:
    """
    Compute macro-average of per-class scores.
    
    Macro-average gives equal weight to each class regardless of frequency.
    
    Args:
        per_class_scores: Dictionary {class_id: score}
        ignore_nan: If True, exclude NaN values from average
        
    Returns:
        Macro-averaged score in [0, 1]
        Returns NaN if all scores are NaN
        
    Example:
        >>> scores = {0: 0.8, 1: 0.6, 2: 0.4}
        >>> macro_average(scores)
        0.6  # (0.8 + 0.6 + 0.4) / 3
    """
    scores = np.array(list(per_class_scores.values()))
    
    if ignore_nan:
        scores = scores[~np.isnan(scores)]
    
    if len(scores) == 0:
        return np.nan
    
    return float(np.mean(scores))


def compute_miou(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> float:
    """
    Compute macro-averaged Intersection over Union (mIoU).
    
    mIoU = mean(IoU_c for all classes c)
    
    Each class contributes equally regardless of pixel frequency.
    
    Args:
        predicted: Predicted segmentation mask (H, W)
        ground_truth: Ground truth segmentation mask (H, W)
        classes: List of class IDs to evaluate
        
    Returns:
        mIoU score in [0, 1], higher is better
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1]])
        >>> gt = np.array([[0, 1], [1, 1]])
        >>> miou = compute_miou(pred, gt)
        >>> # mIoU = (IoU_0 + IoU_1) / 2 = (0.333 + 1.0) / 2 = 0.667
    """
    iou_scores = compute_per_class_iou(predicted, ground_truth, classes)
    return macro_average(iou_scores, ignore_nan=True)


def compute_mdice(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: Optional[List[int]] = None
) -> float:
    """
    Compute macro-averaged Dice coefficient (mDice).
    
    mDice = mean(Dice_c for all classes c)
    
    Args:
        predicted: Predicted segmentation mask (H, W)
        ground_truth: Ground truth segmentation mask (H, W)
        classes: List of class IDs to evaluate
        
    Returns:
        mDice score in [0, 1], higher is better
    """
    dice_scores = compute_per_class_dice(predicted, ground_truth, classes)
    return macro_average(dice_scores, ignore_nan=True)


# ============================================================================
# FREQUENCY-WEIGHTED AGGREGATION (OPTIONAL)
# ============================================================================

def frequency_weighted_average(
    per_class_scores: Dict[int, float],
    class_frequencies: Dict[int, float]
) -> float:
    """
    Compute frequency-weighted average of per-class scores.
    
    Weighted_avg = Σ (frequency_c × score_c)
    
    Args:
        per_class_scores: Dictionary {class_id: score}
        class_frequencies: Dictionary {class_id: frequency} 
                          Frequencies should sum to 1.0
                          
    Returns:
        Frequency-weighted average score
        
    Example:
        >>> scores = {0: 0.9, 1: 0.5, 2: 0.3}
        >>> freqs = {0: 0.7, 1: 0.2, 2: 0.1}
        >>> frequency_weighted_average(scores, freqs)
        0.76  # 0.7*0.9 + 0.2*0.5 + 0.1*0.3
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    for cls, score in per_class_scores.items():
        if cls in class_frequencies and not np.isnan(score):
            weight = class_frequencies[cls]
            weighted_sum += weight * score
            total_weight += weight
    
    if total_weight == 0:
        return np.nan
    
    return weighted_sum / total_weight


def compute_frequency_weighted_miou(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    class_frequencies: Dict[int, float],
    classes: Optional[List[int]] = None
) -> float:
    """
    Compute frequency-weighted mIoU.
    
    Weights each class by its pixel frequency in the dataset.
    
    Args:
        predicted: Predicted segmentation mask (H, W)
        ground_truth: Ground truth segmentation mask (H, W)
        class_frequencies: Dictionary {class_id: pixel_frequency}
        classes: List of class IDs to evaluate
        
    Returns:
        Frequency-weighted mIoU
        
    Note:
        This metric is biased toward frequent classes by design.
        Use for reporting, not primary model selection.
    """
    iou_scores = compute_per_class_iou(predicted, ground_truth, classes)
    return frequency_weighted_average(iou_scores, class_frequencies)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_segmentation(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    image_id: str,
    classes: Optional[List[int]] = None,
    class_frequencies: Optional[Dict[int, float]] = None
) -> Dict[str, any]:
    """
    Complete evaluation of segmentation prediction.
    
    Computes all metrics defined in evaluation_strategy.md:
    - Primary: mIoU, mDice
    - Diagnostic: per-class IoU, Dice, Recall, Precision
    - Optional: frequency-weighted mIoU
    
    Args:
        predicted_mask: Predicted segmentation (H, W) with integer class IDs
        ground_truth_mask: Ground truth segmentation (H, W) with integer class IDs
        image_id: Unique identifier for this image
        classes: List of class IDs to evaluate (default: auto-detect)
        class_frequencies: Optional dict {class_id: frequency} for weighted metrics
        
    Returns:
        Dictionary containing:
        {
            'image_id': str,
            'miou': float,              # Primary metric
            'mdice': float,             # Secondary metric
            'mean_recall': float,       # Diagnostic
            'mean_precision': float,    # Diagnostic
            'per_class_iou': Dict[int, float],
            'per_class_dice': Dict[int, float],
            'per_class_recall': Dict[int, float],
            'per_class_precision': Dict[int, float],
            'frequency_weighted_miou': float  # If class_frequencies provided
        }
        
    Example:
        >>> pred = np.random.randint(0, 5, (512, 512))
        >>> gt = np.random.randint(0, 5, (512, 512))
        >>> results = evaluate_segmentation(pred, gt, "image_001")
        >>> print(f"mIoU: {results['miou']:.3f}")
    """
    # Validate inputs
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_mask.shape} "
            f"vs ground truth {ground_truth_mask.shape}"
        )
    
    if predicted_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D masks, got {predicted_mask.ndim}D"
        )
    
    # Compute per-class metrics
    iou_scores = compute_per_class_iou(predicted_mask, ground_truth_mask, classes)
    dice_scores = compute_per_class_dice(predicted_mask, ground_truth_mask, classes)
    recall_scores = compute_per_class_recall(predicted_mask, ground_truth_mask, classes)
    precision_scores = compute_per_class_precision(predicted_mask, ground_truth_mask, classes)
    
    # Compute macro-averaged metrics
    miou = macro_average(iou_scores, ignore_nan=True)
    mdice = macro_average(dice_scores, ignore_nan=True)
    mean_recall = macro_average(recall_scores, ignore_nan=True)
    mean_precision = macro_average(precision_scores, ignore_nan=True)
    
    # Build results dictionary
    results = {
        'image_id': image_id,
        'miou': miou,
        'mdice': mdice,
        'mean_recall': mean_recall,
        'mean_precision': mean_precision,
        'per_class_iou': iou_scores,
        'per_class_dice': dice_scores,
        'per_class_recall': recall_scores,
        'per_class_precision': precision_scores
    }
    
    # Optional: frequency-weighted metrics
    if class_frequencies is not None:
        fw_miou = frequency_weighted_average(iou_scores, class_frequencies)
        results['frequency_weighted_miou'] = fw_miou
    
    return results


# ============================================================================
# CSV LOGGING
# ============================================================================

def log_evaluation_to_csv(
    results: Dict[str, any],
    csv_path: str,
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Append evaluation results to CSV file.
    
    Creates CSV file with headers if it doesn't exist.
    Each row contains:
    - image_id
    - Global metrics (miou, mdice, mean_recall, mean_precision)
    - Per-class metrics (flattened: iou_class0, iou_class1, ...)
    
    Args:
        results: Output from evaluate_segmentation()
        csv_path: Path to CSV file (created if doesn't exist)
        class_names: Optional mapping {class_id: name} for column headers
        
    Example:
        >>> results = evaluate_segmentation(pred, gt, "img_001")
        >>> log_evaluation_to_csv(results, "eval_results.csv")
    """
    # Extract class IDs (sorted for consistent column order)
    class_ids = sorted(results['per_class_iou'].keys())
    
    # Create column headers
    headers = ['image_id', 'miou', 'mdice', 'mean_recall', 'mean_precision']
    
    # Add per-class columns
    for cls in class_ids:
        class_label = class_names.get(cls, f'class_{cls}') if class_names else f'class_{cls}'
        headers.extend([
            f'iou_{class_label}',
            f'dice_{class_label}',
            f'recall_{class_label}',
            f'precision_{class_label}'
        ])
    
    # Add frequency-weighted metric if present
    if 'frequency_weighted_miou' in results:
        headers.append('frequency_weighted_miou')
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    # Prepare row data
    row = [
        results['image_id'],
        results['miou'],
        results['mdice'],
        results['mean_recall'],
        results['mean_precision']
    ]
    
    # Add per-class metrics
    for cls in class_ids:
        row.extend([
            results['per_class_iou'][cls],
            results['per_class_dice'][cls],
            results['per_class_recall'][cls],
            results['per_class_precision'][cls]
        ])
    
    # Add frequency-weighted metric if present
    if 'frequency_weighted_miou' in results:
        row.append(results['frequency_weighted_miou'])
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(headers)
        
        # Write data row
        writer.writerow(row)


def aggregate_csv_results(
    csv_path: str
) -> Dict[str, float]:
    """
    Compute aggregate statistics from evaluation CSV.
    
    Args:
        csv_path: Path to CSV file with evaluation results
        
    Returns:
        Dictionary with mean values across all images
        
    Example:
        >>> aggregate = aggregate_csv_results("eval_results.csv")
        >>> print(f"Dataset mIoU: {aggregate['miou']:.3f}")
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Compute means for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    aggregates = df[numeric_cols].mean().to_dict()
    
    return aggregates


def _process_single_evaluation(
    pred_path: str,
    gt_path: str,
    class_names: Optional[Dict[int, str]] = None
) -> Optional[Dict]:
    """
    Process a single mask pair evaluation.
    
    Args:
        pred_path: Path to predicted mask
        gt_path: Path to ground truth mask
        class_names: Optional mapping of class indices to names
        
    Returns:
        Dictionary with evaluation results or None if error occurs
    """
    try:
        log.info(f"Evaluating prediction {pred_path}.")
        
        # Load masks (assuming numpy arrays stored as .npy or images)
        pred_path_obj = Path(pred_path)
        gt_path_obj = Path(gt_path)
        
        # Load predicted and ground truth masks
        # Adjust loading based on your file format
        if pred_path_obj.suffix == '.npy':
            pred = np.load(pred_path)
            gt = np.load(gt_path)
        else:
            # For image files, use appropriate loading (PIL, cv2, etc.)

            pred = load_image(pred_path, one_channel=True) #use io_utils load_image
            gt = load_image(gt_path, one_channel=True) #use io_utils load_image
        
        # Get image ID from filename
        image_id = pred_path_obj.stem
        
        # Evaluate
        results = evaluate_segmentation(
            predicted_mask=pred,
            ground_truth_mask=gt,
            image_id=image_id
        )
        return results
        
    except Exception as e:
        log.error(f"Error in file {pred_path}: {str(e)}")
        return None


def evaluation_pipeline(
    pred_mask_paths: List[str],
    gt_mask_paths: List[str],
    csv_output_path: str,
    class_names: Optional[Dict[int, str]] = None,
    n_jobs: Optional[int] = GeneralConfig.NB_JOBS
) -> pd.DataFrame:
    """
    Evaluate multiple prediction-groundtruth mask pairs in parallel.
    
    Args:
        pred_mask_paths: List of paths to predicted masks
        gt_mask_paths: List of paths to ground truth masks
        csv_output_path: Path where results CSV will be saved
        class_names: Optional mapping of class indices to names
        n_jobs: Number of parallel jobs (default: all CPU cores)
        
    Returns:
        DataFrame containing all evaluation results
    """
    log.info(f"Starting evaluation pipeline with {len(pred_mask_paths)} samples.")
    
    if len(pred_mask_paths) != len(gt_mask_paths):
        raise ValueError("Number of predicted and ground truth masks must match")
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    log.info(f"Using {n_jobs} parallel workers.")
    
    # Prepare arguments for parallel processing
    eval_args = list(zip(pred_mask_paths, gt_mask_paths))
    
    # Create partial function with class_names
    eval_func = partial(_process_single_evaluation, class_names=class_names)
    
    # Parallel processing
    with mp.Pool(processes=n_jobs) as pool:
        results_list = pool.starmap(eval_func, eval_args)
    
    # Filter out None results (failed evaluations)
    valid_results = [r for r in results_list if r is not None]
    
    log.info(f"Successfully evaluated {len(valid_results)}/{len(pred_mask_paths)} samples.")
    
    # Convert results to DataFrame
    all_rows = []
    for result in valid_results:
        row = {
            'image_id': result['image_id'],
            'miou': result['miou'],
            'mdice': result['mdice'],
            'mean_recall': result['mean_recall'],
            'mean_precision': result['mean_precision']
        }
        
        # Add per-class metrics
        for cls in result['per_class_iou'].keys():
            class_name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
            row[f'{class_name}_iou'] = result['per_class_iou'][cls]
            row[f'{class_name}_dice'] = result['per_class_dice'][cls]
            row[f'{class_name}_recall'] = result['per_class_recall'][cls]
            row[f'{class_name}_precision'] = result['per_class_precision'][cls]
        
        all_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Save to CSV
    log.info(f"Saving results to {csv_output_path}.")
    df.to_csv(csv_output_path, index=False)
    
    log.info("Evaluation pipeline completed.")
    
    return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Paths
    pred_dir = ResultPath.PREDICTION_PATH
    gt_dir = DataPath.LABEL_TEST

    # List all predicted and ground truth mask files
    pred_files = list_dir_endwith(pred_dir, suffixes=['.png', '.npy', '.jpg'])[:4]
    gt_files = list_dir_endwith(gt_dir, suffixes=['.png', '.npy', '.jpg'])[:4]

    evaluation_pipeline(
        pred_mask_paths=pred_files,
        gt_mask_paths=gt_files,
        csv_output_path=ResultPath.EVALUATION_CSV_PATH,
        n_jobs=4
    )

