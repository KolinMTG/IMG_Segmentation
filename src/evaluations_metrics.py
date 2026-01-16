"""Evaluation metrics for segmentation models."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import confusion_matrix, accuracy_score
import json
from pathlib import Path

from logger import get_logger
log = get_logger("metrics")


def compute_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    per_class: bool = True
) -> Dict[str, float]:
    """
    Compute Intersection over Union (IoU) metric.
    
    Args:
        y_true: Ground truth masks, shape (N, H, W) or (N*H*W,)
        y_pred: Predicted masks, shape (N, H, W) or (N*H*W,)
        num_classes: Number of classes
        per_class: If True, return per-class IoU
        
    Returns:
        Dictionary with IoU scores
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    iou_scores = {}
    class_ious = []
    
    for c in range(num_classes):
        #! True positives, false positives, false negatives
        tp = np.sum((y_true_flat == c) & (y_pred_flat == c))
        fp = np.sum((y_true_flat != c) & (y_pred_flat == c))
        fn = np.sum((y_true_flat == c) & (y_pred_flat != c))
        
        #! IoU = TP / (TP + FP + FN)
        union = tp + fp + fn
        if union > 0:
            iou = tp / union
        else:
            iou = 0.0
        
        class_ious.append(iou)
        if per_class:
            iou_scores[f'iou_class_{c}'] = iou
    
    #! Mean IoU
    iou_scores['mean_iou'] = np.mean(class_ious)
    
    return iou_scores


def compute_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    per_class: bool = True
) -> Dict[str, float]:
    """
    Compute Dice coefficient (F1-score).
    
    Args:
        y_true: Ground truth masks, shape (N, H, W) or (N*H*W,)
        y_pred: Predicted masks, shape (N, H, W) or (N*H*W,)
        num_classes: Number of classes
        per_class: If True, return per-class Dice
        
    Returns:
        Dictionary with Dice scores
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    dice_scores = {}
    class_dice = []
    
    for c in range(num_classes):
        #! True positives, false positives, false negatives
        tp = np.sum((y_true_flat == c) & (y_pred_flat == c))
        fp = np.sum((y_true_flat != c) & (y_pred_flat == c))
        fn = np.sum((y_true_flat == c) & (y_pred_flat != c))
        
        #! Dice = 2*TP / (2*TP + FP + FN)
        denom = 2 * tp + fp + fn
        if denom > 0:
            dice = 2 * tp / denom
        else:
            dice = 0.0
        
        class_dice.append(dice)
        if per_class:
            dice_scores[f'dice_class_{c}'] = dice
    
    #! Mean Dice
    dice_scores['mean_dice'] = np.mean(class_dice)
    
    return dice_scores


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    cm = confusion_matrix(
        y_true_flat,
        y_pred_flat,
        labels=list(range(num_classes))
    )
    
    return cm


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a segmentation model.
    
    Args:
        model: Trained segmentation model (must have .predict() method)
        X_test: Test features, shape (N, H, W, C)
        y_test: Test masks, shape (N, H, W)
        num_classes: Number of classes
        batch_size: Batch size for prediction (only for U-Net)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    log.info("Evaluating model...")
    
    #! Make predictions
    if hasattr(model, 'model_name') and model.model_name == 'UNet':
        y_pred = model.predict(X_test, batch_size=batch_size)
    else:
        y_pred = model.predict(X_test)
    
    #! Compute metrics
    results = {}
    
    #! Accuracy
    accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
    results['accuracy'] = accuracy
    
    #! IoU
    iou_metrics = compute_iou(y_test, y_pred, num_classes)
    results.update(iou_metrics)
    
    #! Dice
    dice_metrics = compute_dice(y_test, y_pred, num_classes)
    results.update(dice_metrics)
    
    #! Confusion matrix
    cm = compute_confusion_matrix(y_test, y_pred, num_classes)
    results['confusion_matrix'] = cm.tolist()
    
    #! Class-wise accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for c in range(num_classes):
        results[f'accuracy_class_{c}'] = class_accuracy[c]
    
    log.info(f"Accuracy: {accuracy:.4f}")
    log.info(f"Mean IoU: {results['mean_iou']:.4f}")
    log.info(f"Mean Dice: {results['mean_dice']:.4f}")
    
    return results


def save_evaluation_report(
    results: Dict[str, Any],
    save_dir: str,
    model_name: str
) -> None:
    """
    Save evaluation report to JSON file.
    
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save report
        model_name: Name of the model
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    report_path = save_path / f'{model_name}_evaluation.json'
    
    #! Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            serializable_results[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            serializable_results[key] = int(value)
        else:
            serializable_results[key] = value
    
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    log.info(f"Saved evaluation report to {report_path}")


def print_evaluation_summary(
    results: Dict[str, Any],
    num_classes: int
) -> None:
    """
    Print a formatted evaluation summary.
    
    Args:
        results: Evaluation results dictionary
        num_classes: Number of classes
    """
    log.info("\n" + "="*50)
    log.info("EVALUATION SUMMARY")
    log.info("="*50)
    
    #! Overall metrics
    log.info(f"\nOverall Metrics:")
    log.info(f"  Accuracy:  {results['accuracy']:.4f}")
    log.info(f"  Mean IoU:  {results['mean_iou']:.4f}")
    log.info(f"  Mean Dice: {results['mean_dice']:.4f}")
    
    #! Per-class metrics
    log.info(f"\nPer-Class Metrics:")
    for c in range(num_classes):
        log.info(f"  Class {c}:")
        log.info(f"    Accuracy: {results.get(f'accuracy_class_{c}', 0):.4f}")
        log.info(f"    IoU:      {results.get(f'iou_class_{c}', 0):.4f}")
        log.info(f"    Dice:     {results.get(f'dice_class_{c}', 0):.4f}")
    
    #! Confusion matrix
    if 'confusion_matrix' in results:
        log.info(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        log.info(f"{cm}")
    
    log.info("="*50 + "\n")
