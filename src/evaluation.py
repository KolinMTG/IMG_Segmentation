"""
Semantic Image Segmentation Evaluation Module

This module provides comprehensive evaluation metrics and visualizations
for semantic segmentation tasks comparing ground truth and predicted masks.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import logging
from src.cste import ClassInfo, DataPath
from src.io_utils import load_image

# Assume log is already defined
log = logging.getLogger(__name__)


def compute_confusion_matrix(
    gt_masks: np.ndarray, pred_masks: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix for segmentation predictions.

    Args:
        gt_masks: Ground truth masks (flattened or 3D)
        pred_masks: Predicted masks (flattened or 3D)
        num_classes: Number of semantic classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    #! Flatten masks to 1D for easier processing
    gt_flat = gt_masks.flatten()
    pred_flat = pred_masks.flatten()

    #! Compute confusion matrix using bincount trick
    mask = (gt_flat >= 0) & (gt_flat < num_classes)
    confusion = np.bincount(
        num_classes * gt_flat[mask].astype(int) + pred_flat[mask].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)

    return confusion


def compute_iou(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Intersection over Union (IoU) metrics.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        Tuple of (per-class IoU array, mean IoU)
    """
    #! IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    #! Avoid division by zero
    iou = np.zeros(len(intersection))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]

    mean_iou = np.mean(iou[valid]) if valid.any() else 0.0

    return iou, mean_iou


def compute_dice(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Dice coefficient (F1-score) metrics.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        Tuple of (per-class Dice array, mean Dice)
    """
    #! Dice = 2*TP / (2*TP + FP + FN)
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)

    dice = np.zeros(len(intersection))
    denominator = ground_truth_set + predicted_set
    valid = denominator > 0
    dice[valid] = (2.0 * intersection[valid]) / denominator[valid]

    mean_dice = np.mean(dice[valid]) if valid.any() else 0.0

    return dice, mean_dice


def compute_precision_recall(
    confusion_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall per class.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        Tuple of (precision array, recall array)
    """
    true_positive = np.diag(confusion_matrix)
    predicted_positive = confusion_matrix.sum(axis=0)
    actual_positive = confusion_matrix.sum(axis=1)

    #! Precision = TP / (TP + FP)
    precision = np.zeros(len(true_positive))
    valid_prec = predicted_positive > 0
    precision[valid_prec] = true_positive[valid_prec] / predicted_positive[valid_prec]

    #! Recall = TP / (TP + FN)
    recall = np.zeros(len(true_positive))
    valid_rec = actual_positive > 0
    recall[valid_rec] = true_positive[valid_rec] / actual_positive[valid_rec]

    return precision, recall


def compute_pixel_accuracy(confusion_matrix: np.ndarray) -> float:
    """
    Compute overall pixel accuracy.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        Overall pixel accuracy
    """
    correct = np.diag(confusion_matrix).sum()
    total = confusion_matrix.sum()
    return correct / total if total > 0 else 0.0


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Dict[int, str],
    output_path: str,
    normalized: bool = True,
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix to visualize
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save the figure
        normalized: Whether to normalize the confusion matrix
    """
    if normalized:
        #! Normalize by row (ground truth)
        cm_norm = confusion_matrix.astype("float") / (
            confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10
        )
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        cm_norm = confusion_matrix
        title = "Confusion Matrix"
        fmt = "d"

    labels = [class_names[i] for i in range(len(class_names))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Count" if normalized else "Count"},
    )
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Saved confusion matrix to {output_path}")


def plot_metric_bars(
    metrics: np.ndarray, class_names: Dict[int, str], metric_name: str, output_path: str
) -> None:
    """
    Plot bar chart for per-class metrics.

    Args:
        metrics: Array of metric values per class
        class_names: Dictionary mapping class IDs to names
        metric_name: Name of the metric (for title and labels)
        output_path: Path to save the figure
    """
    labels = [class_names[i] for i in range(len(class_names))]
    colors = [
        np.array(ClassInfo.CLASS_COLORS[i]) / 255.0 for i in range(len(class_names))
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, metrics, color=colors, edgecolor="black", linewidth=1.2)

    #! Add value labels on top of bars
    for bar, value in zip(bars, metrics):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel(metric_name)
    plt.xlabel("Class")
    plt.title(f"{metric_name} per Class")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Saved {metric_name} plot to {output_path}")


def plot_metrics_summary(
    metrics_dict: Dict, class_names: Dict[int, str], output_path: str
) -> None:
    """
    Create a summary table figure with all metrics.

    Args:
        metrics_dict: Dictionary containing all computed metrics
        class_names: Dictionary mapping class IDs to names
        output_path: Path to save the figure
    """
    num_classes = len(class_names)
    labels = [class_names[i] for i in range(num_classes)]

    #! Prepare table data
    table_data = []
    for i in range(num_classes):
        row = [
            labels[i],
            f"{metrics_dict['iou_per_class'][i]:.3f}",
            f"{metrics_dict['dice_per_class'][i]:.3f}",
            f"{metrics_dict['precision'][i]:.3f}",
            f"{metrics_dict['recall'][i]:.3f}",
            f"{int(metrics_dict['support'][i])}",
        ]
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(12, num_classes * 0.6 + 2))
    ax.axis("tight")
    ax.axis("off")

    col_labels = ["Class", "IoU", "Dice", "Precision", "Recall", "Support"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    #! Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    #! Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title("Per-Class Metrics Summary", fontsize=14, weight="bold", pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Saved metrics summary table to {output_path}")


def generate_report(
    metrics_dict: Dict, class_names: Dict[int, str], num_images: int, output_path: str
) -> None:
    """
    Generate a comprehensive evaluation report.

    Args:
        metrics_dict: Dictionary containing all computed metrics
        class_names: Dictionary mapping class IDs to names
        num_images: Number of images evaluated
        output_path: Path to save the report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SEMANTIC SEGMENTATION EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    #! Dataset info
    report_lines.append(f"Number of images evaluated: {num_images}")
    report_lines.append("")

    #! Global metrics
    report_lines.append("-" * 80)
    report_lines.append("GLOBAL METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Overall Pixel Accuracy: {metrics_dict['pixel_accuracy']:.4f}")
    report_lines.append(f"Mean IoU (mIoU):        {metrics_dict['mean_iou']:.4f}")
    report_lines.append(f"Mean Dice (F1-score):   {metrics_dict['mean_dice']:.4f}")
    report_lines.append("")

    #! Per-class metrics
    report_lines.append("-" * 80)
    report_lines.append("PER-CLASS METRICS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for i in range(len(class_names)):
        class_name = class_names[i]
        report_lines.append(f"Class: {class_name}")
        report_lines.append(
            f"  IoU:              {metrics_dict['iou_per_class'][i]:.4f}"
        )
        report_lines.append(
            f"  Dice (F1-score):  {metrics_dict['dice_per_class'][i]:.4f}"
        )
        report_lines.append(f"  Precision:        {metrics_dict['precision'][i]:.4f}")
        report_lines.append(f"  Recall:           {metrics_dict['recall'][i]:.4f}")
        report_lines.append(f"  Support (pixels): {int(metrics_dict['support'][i])}")
        report_lines.append("")

    report_lines.append("=" * 80)

    #! Save text report
    report_text = "\n".join(report_lines)
    txt_path = os.path.join(output_path, "evaluation_report.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    log.info(f"Saved text report to {txt_path}")

    #! Save JSON report
    json_data = {
        "num_images": num_images,
        "global_metrics": {
            "pixel_accuracy": float(metrics_dict["pixel_accuracy"]),
            "mean_iou": float(metrics_dict["mean_iou"]),
            "mean_dice": float(metrics_dict["mean_dice"]),
        },
        "per_class_metrics": {},
    }

    for i in range(len(class_names)):
        class_name = class_names[i]
        json_data["per_class_metrics"][class_name] = {
            "iou": float(metrics_dict["iou_per_class"][i]),
            "dice": float(metrics_dict["dice_per_class"][i]),
            "precision": float(metrics_dict["precision"][i]),
            "recall": float(metrics_dict["recall"][i]),
            "support": int(metrics_dict["support"][i]),
        }

    json_path = os.path.join(output_path, "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    log.info(f"Saved JSON report to {json_path}")


def evaluate_predictions(
    ground_truth_dir: str, predictions_dir: str, output_path: str
) -> None:
    """
    Evaluate semantic segmentation predictions against ground truth.

    This function computes comprehensive metrics comparing predicted segmentation
    masks to ground truth masks, generates visualizations, and produces detailed
    evaluation reports.

    Args:
        ground_truth_dir: Directory containing ground truth masks (.png or .npy)
        predictions_dir: Directory containing predicted masks (.png or .npy)
        output_path: Directory where reports and plots will be saved

    The function generates:
        - Confusion matrix heatmap (normalized)
        - Bar plots for IoU and Dice per class
        - Metrics summary table
        - Text and JSON evaluation reports
    """
    log.info("Starting segmentation evaluation...")

    #! Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    #! Get list of mask files
    gt_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith(".png")])
    pred_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith(".png")])

    #! Initialize confusion matrix
    num_classes = ClassInfo.NUM_CLASSES
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    #! Process all mask pairs
    for gt_file, pred_file in zip(gt_files, pred_files):

        print(f"Evaluating {gt_file} vs {pred_file}...")
        #! Load ground truth masks
        gt_mask = np.array(
            Image.open(os.path.join(ground_truth_dir, gt_file)).convert("L")
        )
        pred_mask = np.array(
            Image.open(os.path.join(predictions_dir, pred_file)).convert("L")
        )

        #! Verify shapes match
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(
                f"Shape mismatch for {gt_file} vs {pred_file}: "
                f"GT {gt_mask.shape} vs Pred {pred_mask.shape}"
            )

        #! Accumulate confusion matrix
        confusion = compute_confusion_matrix(gt_mask, pred_mask, num_classes)
        total_confusion += confusion

    log.info("Computed confusion matrix for all images")

    #! Compute all metrics
    pixel_accuracy = compute_pixel_accuracy(total_confusion)
    iou_per_class, mean_iou = compute_iou(total_confusion)
    dice_per_class, mean_dice = compute_dice(total_confusion)
    precision, recall = compute_precision_recall(total_confusion)
    support = total_confusion.sum(axis=1)  # Ground truth pixel counts

    metrics_dict = {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class,
        "precision": precision,
        "recall": recall,
        "support": support,
        "confusion_matrix": total_confusion,
    }

    log.info("Computed all evaluation metrics")

    #! Generate plots
    plot_confusion_matrix(
        total_confusion,
        ClassInfo.CLASS_NAMES,
        os.path.join(output_path, "confusion_matrix.png"),
        normalized=True,
    )

    plot_metric_bars(
        iou_per_class,
        ClassInfo.CLASS_NAMES,
        "IoU",
        os.path.join(output_path, "iou_per_class.png"),
    )

    plot_metric_bars(
        dice_per_class,
        ClassInfo.CLASS_NAMES,
        "Dice Score",
        os.path.join(output_path, "dice_per_class.png"),
    )

    plot_metrics_summary(
        metrics_dict,
        ClassInfo.CLASS_NAMES,
        os.path.join(output_path, "metrics_summary.png"),
    )

    #! Generate reports
    generate_report(metrics_dict, ClassInfo.CLASS_NAMES, len(gt_files), output_path)

    log.info(f"Evaluation complete. Results saved to {output_path}")
    log.info(f"Overall Pixel Accuracy: {pixel_accuracy:.4f}")
    log.info(f"Mean IoU: {mean_iou:.4f}")
    log.info(f"Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    #! Example usage
    # evaluate_predictions(
    #     ground_truth_dir=DataPath.LABEL_TEST,
    #     predictions_dir=DataPath.UNET_INFERENCE_DIR_POSTTREATMENT,
    #     output_path=r"data/results/evaluation_report/unet_posttreatment/"
    # )

    # evaluate_predictions(
    #     ground_truth_dir=DataPath.LABEL_TEST,
    #     predictions_dir=DataPath.HISTOGRAM_DIR,
    #     output_path=r"data/results/evaluation_report/histogram_inference/"
    # )

    evaluate_predictions(
        ground_truth_dir=DataPath.LABEL_TEST,
        predictions_dir=r"data/results/histogram_adaptive_inference",
        output_path=r"data/results/evaluation_report/histogram_adaptive_inference/",
    )
