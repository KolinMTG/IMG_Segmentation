"""
Example usage of probabilistic segmentation system.

This script demonstrates:
1. Training models from labeled data
2. Running inference on test images
3. Visualizing results
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict

from train_classifier import train_models_from_directory
from field import process_field
from building import process_building
from woodland import process_woodland
from water import process_water
from road import process_road
from cste import ClassInfo, DataPath


def train_models() -> None:
    """
    Train GMM models for all classes.
    
    ! Run this once before inference
    """
    train_models_from_directory(
        img_dir=DataPath.IMG_TRAIN,
        label_dir=DataPath.LABEL_TRAIN,
        model_dir="data/models/",
        n_components=3
    )


def run_inference(img_path: str) -> Dict[int, np.ndarray]:
    """
    Run inference on a single image.
    
    Args:
        img_path: Path to input image
        
    Returns:
        Dictionary mapping class ID to probability map
    """
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # ! Run all class detectors
    prob_maps = {
        0: process_field(img),
        1: process_building(img),
        2: process_woodland(img),
        3: process_water(img),
        4: process_road(img),
    }
    
    return prob_maps


def aggregate_predictions(prob_maps: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Aggregate per-class probabilities into final segmentation.
    
    ! This uses simple argmax - your existing pipeline may have
    ! additional steps (smoothing, morphology, etc.)
    
    Args:
        prob_maps: Dictionary of class ID to probability map
        
    Returns:
        Segmentation mask (H, W) with class IDs 0-4
    """
    # Stack probability maps
    prob_stack = np.stack([prob_maps[i] for i in range(5)], axis=-1)
    
    # Argmax to get most likely class per pixel
    segmentation = np.argmax(prob_stack, axis=-1).astype(np.uint8)
    
    return segmentation


def visualize_results(
    img: np.ndarray,
    prob_maps: Dict[int, np.ndarray],
    segmentation: np.ndarray,
    save_path: str = None
) -> None:
    """
    Visualize segmentation results.
    
    Args:
        img: Original RGB image in [0, 1]
        prob_maps: Per-class probability maps
        segmentation: Final segmentation mask
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Probability maps for each class
    for i, (class_id, class_name) in enumerate(ClassInfo.CLASS_NAMES.items()):
        row = (i + 1) // 4
        col = (i + 1) % 4
        axes[row, col].imshow(prob_maps[class_id], cmap='hot')
        axes[row, col].set_title(f'{class_name} Probability')
        axes[row, col].axis('off')
    
    # Final segmentation
    axes[1, 2].imshow(segmentation, cmap='tab10', vmin=0, vmax=4)
    axes[1, 2].set_title('Final Segmentation')
    axes[1, 2].axis('off')
    
    # Color legend
    axes[1, 3].axis('off')
    legend_text = "Classes:\n"
    for class_id, class_name in ClassInfo.CLASS_NAMES.items():
        legend_text += f"{class_id}: {class_name}\n"
    axes[1, 3].text(0.1, 0.5, legend_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def batch_inference(img_dir: str, output_dir: str = "data/results/") -> None:
    """
    Run inference on all images in a directory.
    
    Args:
        img_dir: Directory containing input images
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    
    for img_path in img_paths:
        # Run inference
        prob_maps = run_inference(str(img_path))
        
        # Aggregate
        segmentation = aggregate_predictions(prob_maps)
        
        # Save result
        output_path = Path(output_dir) / f"{img_path.stem}_seg.png"
        cv2.imwrite(str(output_path), segmentation)


if __name__ == "__main__":
    # ========================================================================
    # STEP 1: Train models (run once)
    # ========================================================================
    
    # ! Uncomment to train models
    # train_models()
    
    # ========================================================================
    # STEP 2: Run inference on test image
    # ========================================================================
    
    test_img_path = "data/images/test/M-33-7-A-d-2-3_19.jpg"
    
    # Load image
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Get probability maps
    prob_maps = run_inference(test_img_path)
    
    # Aggregate into final segmentation
    segmentation = aggregate_predictions(prob_maps)
    
    # ========================================================================
    # STEP 3: Visualize
    # ========================================================================
    
    visualize_results(
        img,
        prob_maps,
        segmentation,
        save_path="data/results/example_segmentation.png"
    )
    
    # ========================================================================
    # STEP 4: Batch processing (optional)
    # ========================================================================
    
    # ! Uncomment to process all test images
    # batch_inference(
    #     img_dir=DataPath.IMG_TEST,
    #     output_dir="data/results/predictions/"
    # )