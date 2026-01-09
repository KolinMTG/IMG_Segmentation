"""Useful functions to get information about the dataset. Or visualize images and labels."""


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

from src.cste import ClassInfo
from src.logger import get_logger


log = get_logger("data_info.log")

def show_img_labels(img_path: str, label_path: str):
    """
    Visualize image and segmentation mask.
    
    Args:
        img_path: Path to the RGB image
        label_path: Path to the segmentation mask
    """
    # Load RGB image
    img = Image.open(img_path).convert("RGB")

    # Load mask as single-channel
    label = Image.open(label_path).convert("L")

    # Convert to numpy arrays
    img_array = np.array(img)
    label_array = np.array(label)

    log.info(f"Image array shape: {img_array.shape}")
    log.info(f"Label array shape: {label_array.shape}")

    # Get colors and class names from constants
    colors = ClassInfo.CLASS_COLOR
    class_names = ClassInfo.CLASS_NAME 

    # Build colored mask from segmentation labels
    colored_mask = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

    for class_id, color in colors.items():
        colored_mask[label_array == class_id] = color

    # Create overlay by blending image and mask
    alpha = 0.5
    overlay = (img_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    # Display results in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title("Colored Mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay")
    axes[1, 0].axis("off")

    # Add legend with class statistics
    axes[1, 1].axis("off")
    legend_text = "Legend:\n\n"

    for class_id, name in class_names.items():
        count = np.sum(label_array == class_id)
        percentage = (count / label_array.size) * 100
        legend_text += f"- {name} ({class_id}): {percentage:.1f}%\n"

    axes[1, 1].text(0.05, 0.9, legend_text, va="top", family="monospace")
    plt.tight_layout()
    plt.show()


def class_proportion(labels_path_folder: str) -> None:
    """
    Calculate and display the total proportion of each class across all label images.
    
    Args:
        labels_path_folder: Path to the folder containing label images
    """
    # Dictionary to store total pixel count per class
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_pixels = 0

    # Get class names and colors from constants
    class_names = ClassInfo.CLASS_NAME
    colors = ClassInfo.CLASS_COLOR
    
    # List all label image files in the folder
    label_files = [f for f in os.listdir(labels_path_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    log.info(f"Analyzing {len(label_files)} label images...")
    
    # Process each label image
    for label_file in tqdm(label_files, desc="Processing label images"):
        label_path = os.path.join(labels_path_folder, label_file)
        
        # Load mask as grayscale
        label = Image.open(label_path).convert("L")
        label_array = np.array(label)
        
        # Count pixels for each class
        for class_id in class_counts.keys():
            class_counts[class_id] += np.sum(label_array == class_id)
        
        total_pixels += label_array.size
    
    # Calculate and display proportions
    log.info("\n" + "="*60)
    log.info("CLASS PROPORTIONS ACROSS ENTIRE DATASET")
    log.info("="*60 + "\n")
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        log.info(f"{class_names[class_id]:15} (class {class_id}): "
                 f"{percentage:6.2f}% ({count:,} pixels)")
    
    log.info(f"\n{'Total':15}: {total_pixels:,} pixels")
    log.info("="*60)
    
    # Visualization with bar chart and pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for visualization
    classes = [class_names[i] for i in sorted(class_counts.keys())]
    percentages = [(class_counts[i] / total_pixels) * 100 
                   for i in sorted(class_counts.keys())]
    bar_colors = [np.array(colors[i]) / 255.0 for i in sorted(class_counts.keys())]
    
    # Bar chart
    bars = ax1.bar(classes, percentages, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Proportion (%)', fontsize=12)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage values on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Pie chart
    ax2.pie(percentages, labels=classes, colors=bar_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def class_proportion_by_image(labels_path_folder: str) -> None:
    """
    Calculate and display the proportion of images containing at least one pixel of each class.
    Focuses on class presence rather than pixel count.
    
    Args:
        labels_path_folder: Path to the folder containing label images
    """
    # Dictionary to store the number of images containing each class
    class_presence = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_images = 0
    
    # Get class names and colors from constants
    class_names = ClassInfo.CLASS_NAME
    colors = ClassInfo.CLASS_COLOR
    
    # List all label image files in the folder
    label_files = [f for f in os.listdir(labels_path_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    log.info(f"Analyzing presence of classes in {len(label_files)} images...")
    
    # Process each label image
    for label_file in tqdm(label_files, desc="Processing label images"):
        label_path = os.path.join(labels_path_folder, label_file)
        
        # Load mask as grayscale
        label = Image.open(label_path).convert("L")
        label_array = np.array(label)
        
        # Check presence of each class (at least one pixel)
        for class_id in class_presence.keys():
            if np.any(label_array == class_id):
                class_presence[class_id] += 1
        
        total_images += 1
    
    # Calculate and display proportions
    log.info("\n" + "="*60)
    log.info("CLASS PRESENCE ACROSS IMAGES")
    log.info("="*60 + "\n")
    
    for class_id in sorted(class_presence.keys()):
        count = class_presence[class_id]
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        log.info(f"{class_names[class_id]:15} (class {class_id}): "
                 f"{percentage:6.2f}% ({count}/{total_images} images)")
    
    log.info(f"\n{'Total images':15}: {total_images}")
    log.info("="*60)
    
    # Visualization with bar chart and pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for visualization
    classes = [class_names[i] for i in sorted(class_presence.keys())]
    percentages = [(class_presence[i] / total_images) * 100 
                   for i in sorted(class_presence.keys())]
    bar_colors = [np.array(colors[i]) / 255.0 for i in sorted(class_presence.keys())]
    
    # Bar chart
    bars = ax1.bar(classes, percentages, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Presence in Images (%)', fontsize=12)
    ax1.set_title('Class Presence Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage values and counts on top of bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        count = class_presence[sorted(class_presence.keys())[i]]
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n({count}/{total_images})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Pie chart
    ax2.pie(percentages, labels=classes, colors=bar_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Class Presence Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Example usage: display a single image with its label
    # img_path = os.path.join(DataPath.IMG_TRAIN, 'M-34-51-C-d-4-1_191.jpg')
    # label_path = os.path.join(DataPath.LABEL_TRAIN, 'M-34-51-C-d-4-1_191_m.png')
    # show_img_labels(img_path, label_path)
    
    # Analyze class proportions across the entire training dataset
    log.info("\n")
    # class_proportion(DataPath.LABEL_TRAIN)
    # class_proportion_by_image(DataPath.LABEL_TRAIN)

    show_img_labels(r"data/images/test/M-33-20-D-c-4-2_0.png", r"data/results/predictions/M-33-20-D-c-4-2_0_seg.png")