"""
Module for generating comprehensive segmentation metrics reports from CSV data.

This module provides functionality to analyze image segmentation metrics,
generate statistical summaries, create visualizations, and produce a markdown report.
"""

import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cste import ResultPath, DataPath


def generate_report_from_csv(csv_path: str, output_path: str) -> None:
    """
    Generate a comprehensive markdown report from segmentation metrics CSV.
    
    Args:
        csv_path: Path to the input CSV file containing segmentation metrics.
        output_path: Directory path where the report and plots will be saved.
    """
    # ! Load the CSV data
    df = pd.read_csv(csv_path)
    
    # ! Create output directories
    os.makedirs(output_path, exist_ok=True)
    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # ! Extract class information
    classes = _extract_classes(df)
    
    # ! Generate all analysis components
    dataset_overview = _generate_dataset_overview(df, classes)
    per_class_analysis = _generate_per_class_analysis(df, classes)
    image_analysis = _generate_image_analysis(df)
    correlation_analysis = _generate_correlation_analysis(df)
    
    # ! Create all plots
    _create_distribution_plots(df, plots_dir)
    _create_per_class_plots(df, classes, plots_dir)
    _create_correlation_heatmap(df, plots_dir)
    
    # ! Generate summary
    summary = _generate_summary(df, classes, per_class_analysis)
    
    # ! Write markdown report
    _write_markdown_report(
        output_path, dataset_overview, per_class_analysis,
        image_analysis, correlation_analysis, summary
    )


def _extract_classes(df: pd.DataFrame) -> List[str]:
    """Extract unique class identifiers from column names."""
    classes = set()
    for col in df.columns:
        if col.startswith('class_') and '_iou' in col:
            class_id = col.replace('class_', '').replace('_iou', '')
            classes.add(class_id)
    return sorted(list(classes))


def _generate_dataset_overview(df: pd.DataFrame, classes: List[str]) -> Dict:
    """Generate dataset overview statistics."""
    global_metrics = ['miou', 'mdice', 'mean_recall', 'mean_precision']
    
    overview = {
        'total_images': len(df),
        'classes': classes,
        'num_classes': len(classes),
        'stats': {}
    }
    
    # ! Calculate statistics for global metrics
    for metric in global_metrics:
        if metric in df.columns:
            overview['stats'][metric] = {
                'mean': df[metric].mean(),
                'median': df[metric].median(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max()
            }
    
    return overview


def _generate_per_class_analysis(df: pd.DataFrame, classes: List[str]) -> Dict:
    """Generate per-class statistical analysis."""
    metrics = ['iou', 'dice', 'recall', 'precision']
    per_class = {}
    
    for cls in classes:
        per_class[cls] = {}
        for metric in metrics:
            col_name = f'class_{cls}_{metric}'
            if col_name in df.columns:
                per_class[cls][metric] = {
                    'mean': df[col_name].mean(),
                    'median': df[col_name].median(),
                    'std': df[col_name].std(),
                    'min': df[col_name].min(),
                    'max': df[col_name].max()
                }
    
    # ! Identify best and worst performing classes
    iou_means = {cls: per_class[cls]['iou']['mean'] for cls in classes if 'iou' in per_class[cls]}
    dice_means = {cls: per_class[cls]['dice']['mean'] for cls in classes if 'dice' in per_class[cls]}
    
    analysis = {
        'per_class': per_class,
        'best_iou': max(iou_means, key=iou_means.get) if iou_means else None,
        'worst_iou': min(iou_means, key=iou_means.get) if iou_means else None,
        'best_dice': max(dice_means, key=dice_means.get) if dice_means else None,
        'worst_dice': min(dice_means, key=dice_means.get) if dice_means else None
    }
    
    return analysis


def _generate_image_analysis(df: pd.DataFrame) -> Dict:
    """Generate image-level analysis (top and bottom performers)."""
    analysis = {}
    
    # ! Top and bottom 5 by mIoU
    if 'miou' in df.columns:
        top_miou = df.nlargest(5, 'miou')[['image_id', 'miou', 'mdice']]
        bottom_miou = df.nsmallest(5, 'miou')[['image_id', 'miou', 'mdice']]
        analysis['top_miou'] = top_miou
        analysis['bottom_miou'] = bottom_miou
    
    # ! Top and bottom 5 by mDice
    if 'mdice' in df.columns:
        top_mdice = df.nlargest(5, 'mdice')[['image_id', 'miou', 'mdice']]
        bottom_mdice = df.nsmallest(5, 'mdice')[['image_id', 'miou', 'mdice']]
        analysis['top_mdice'] = top_mdice
        analysis['bottom_mdice'] = bottom_mdice
    
    return analysis


def _generate_correlation_analysis(df: pd.DataFrame) -> Dict:
    """Generate correlation analysis for global metrics."""
    global_metrics = ['miou', 'mdice', 'mean_recall', 'mean_precision']
    available_metrics = [m for m in global_metrics if m in df.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = df[available_metrics].corr()
        return {'correlation_matrix': corr_matrix}
    
    return {'correlation_matrix': None}


def _create_distribution_plots(df: pd.DataFrame, plots_dir: str) -> None:
    """Create distribution plots for mIoU and mDice."""
    metrics = [('miou', 'mIoU'), ('mdice', 'mDice')]
    
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        
        # ! Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[metric], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {label}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_histogram.png'), dpi=300)
        plt.close()
        
        # ! Boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(df[metric], vert=True)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Boxplot of {label}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_boxplot.png'), dpi=300)
        plt.close()


def _create_per_class_plots(df: pd.DataFrame, classes: List[str], plots_dir: str) -> None:
    """Create per-class bar plots for all metrics."""
    metrics = [
        ('iou', 'IoU'),
        ('dice', 'Dice'),
        ('recall', 'Recall'),
        ('precision', 'Precision')
    ]
    
    for metric, label in metrics:
        means = []
        class_labels = []
        
        for cls in classes:
            col_name = f'class_{cls}_{metric}'
            if col_name in df.columns:
                means.append(df[col_name].mean())
                class_labels.append(cls)
        
        if means:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(class_labels)), means, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel(f'Mean {label}', fontsize=12)
            ax.set_title(f'Per-Class Mean {label}', fontsize=14)
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'per_class_{metric}.png'), dpi=300)
            plt.close()


def _create_correlation_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    """Create correlation heatmap for global metrics."""
    global_metrics = ['miou', 'mdice', 'mean_recall', 'mean_precision']
    available_metrics = [m for m in global_metrics if m in df.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = df[available_metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix of Global Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()


def _generate_summary(df: pd.DataFrame, classes: List[str], per_class_analysis: Dict) -> str:
    """Generate summary paragraph highlighting key findings."""
    summary_parts = []
    
    # ! Overall performance
    if 'miou' in df.columns:
        mean_miou = df['miou'].mean()
        summary_parts.append(
            f"The dataset contains {len(df)} images across {len(classes)} classes, "
            f"achieving an overall mean IoU of {mean_miou:.4f}."
        )
    
    # ! Best and worst classes
    if per_class_analysis.get('best_iou') and per_class_analysis.get('worst_iou'):
        best_cls = per_class_analysis['best_iou']
        worst_cls = per_class_analysis['worst_iou']
        best_val = per_class_analysis['per_class'][best_cls]['iou']['mean']
        worst_val = per_class_analysis['per_class'][worst_cls]['iou']['mean']
        
        summary_parts.append(
            f"Class {best_cls} shows the best performance with mean IoU of {best_val:.4f}, "
            f"while class {worst_cls} has the lowest mean IoU of {worst_val:.4f}."
        )
    
    # ! Variability
    if 'miou' in df.columns:
        std_miou = df['miou'].std()
        summary_parts.append(
            f"The standard deviation of {std_miou:.4f} in mIoU indicates "
            f"{'moderate' if std_miou > 0.1 else 'low'} variability in model performance across images."
        )
    
    return ' '.join(summary_parts)


def _write_markdown_report(
    output_path: str,
    dataset_overview: Dict,
    per_class_analysis: Dict,
    image_analysis: Dict,
    correlation_analysis: Dict,
    summary: str
) -> None:
    """Write the complete markdown report."""
    report_path = os.path.join(output_path, 'report.md')
    
    with open(report_path, 'w') as f:
        # ! Title
        f.write("# Image Segmentation Metrics Report\n\n")
        
        # ! A. Dataset Overview
        f.write("## A. Dataset Overview\n\n")
        f.write(f"- **Total Images**: {dataset_overview['total_images']}\n")
        f.write(f"- **Number of Classes**: {dataset_overview['num_classes']}\n")
        f.write(f"- **Classes**: {', '.join(dataset_overview['classes'])}\n\n")
        
        f.write("### Global Metrics Summary\n\n")
        for metric, stats in dataset_overview['stats'].items():
            f.write(f"**{metric.upper()}**:\n")
            f.write(f"- Mean: {stats['mean']:.4f}\n")
            f.write(f"- Median: {stats['median']:.4f}\n")
            f.write(f"- Std: {stats['std']:.4f}\n")
            f.write(f"- Min: {stats['min']:.4f}\n")
            f.write(f"- Max: {stats['max']:.4f}\n\n")
        
        # ! B. Per-Class Analysis
        f.write("## B. Per-Class Analysis\n\n")
        
        for cls in sorted(per_class_analysis['per_class'].keys()):
            f.write(f"### Class {cls}\n\n")
            for metric, stats in per_class_analysis['per_class'][cls].items():
                f.write(f"**{metric.upper()}**: ")
                f.write(f"Mean={stats['mean']:.4f}, ")
                f.write(f"Median={stats['median']:.4f}, ")
                f.write(f"Std={stats['std']:.4f}, ")
                f.write(f"Min={stats['min']:.4f}, ")
                f.write(f"Max={stats['max']:.4f}\n\n")
        
        f.write("### Class Performance Highlights\n\n")
        if per_class_analysis.get('best_iou'):
            f.write(f"- **Highest Mean IoU**: Class {per_class_analysis['best_iou']}\n")
            f.write(f"- **Lowest Mean IoU**: Class {per_class_analysis['worst_iou']}\n")
            f.write(f"- **Highest Mean Dice**: Class {per_class_analysis['best_dice']}\n")
            f.write(f"- **Lowest Mean Dice**: Class {per_class_analysis['worst_dice']}\n\n")
        
        # ! C. Distribution Plots
        f.write("## C. Distribution Analysis\n\n")
        
        f.write("### mIoU Distribution\n\n")
        f.write("![mIoU Histogram](plots/miou_histogram.png)\n\n")
        f.write("![mIoU Boxplot](plots/miou_boxplot.png)\n\n")
        
        f.write("### mDice Distribution\n\n")
        f.write("![mDice Histogram](plots/mdice_histogram.png)\n\n")
        f.write("![mDice Boxplot](plots/mdice_boxplot.png)\n\n")
        
        f.write("### Per-Class Metrics\n\n")
        f.write("![Per-Class IoU](plots/per_class_iou.png)\n\n")
        f.write("![Per-Class Dice](plots/per_class_dice.png)\n\n")
        f.write("![Per-Class Recall](plots/per_class_recall.png)\n\n")
        f.write("![Per-Class Precision](plots/per_class_precision.png)\n\n")
        
        # ! D. Image-Level Analysis
        f.write("## D. Image-Level Analysis\n\n")
        
        if 'top_miou' in image_analysis:
            f.write("### Top 5 Images by mIoU\n\n")
            f.write("| Image ID | mIoU | mDice |\n")
            f.write("|----------|------|-------|\n")
            for _, row in image_analysis['top_miou'].iterrows():
                f.write(f"| {row['image_id']} | {row['miou']:.4f} | {row['mdice']:.4f} |\n")
            f.write("\n")
            
            f.write("### Bottom 5 Images by mIoU\n\n")
            f.write("| Image ID | mIoU | mDice |\n")
            f.write("|----------|------|-------|\n")
            for _, row in image_analysis['bottom_miou'].iterrows():
                f.write(f"| {row['image_id']} | {row['miou']:.4f} | {row['mdice']:.4f} |\n")
            f.write("\n")
        
        if 'top_mdice' in image_analysis:
            f.write("### Top 5 Images by mDice\n\n")
            f.write("| Image ID | mIoU | mDice |\n")
            f.write("|----------|------|-------|\n")
            for _, row in image_analysis['top_mdice'].iterrows():
                f.write(f"| {row['image_id']} | {row['miou']:.4f} | {row['mdice']:.4f} |\n")
            f.write("\n")
            
            f.write("### Bottom 5 Images by mDice\n\n")
            f.write("| Image ID | mIoU | mDice |\n")
            f.write("|----------|------|-------|\n")
            for _, row in image_analysis['bottom_mdice'].iterrows():
                f.write(f"| {row['image_id']} | {row['miou']:.4f} | {row['mdice']:.4f} |\n")
            f.write("\n")
        
        # ! E. Correlation Analysis
        f.write("## E. Correlation Analysis\n\n")
        
        if correlation_analysis['correlation_matrix'] is not None:
            f.write("### Correlation Matrix\n\n")
            f.write("![Correlation Heatmap](plots/correlation_heatmap.png)\n\n")
        
        # ! F. Summary
        f.write("## F. Summary\n\n")
        f.write(f"{summary}\n")


if __name__ == "__main__":
    # Example usage
    generate_report_from_csv(
        csv_path=ResultPath.EVALUATION_CSV_PATH,
        output_path=DataPath.REPORT_PATH
    )