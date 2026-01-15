"""Useful functions to get information about the dataset. Or visualize images and labels."""


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from typing import List
import seaborn as sns
import pandas as pd
from src.cste import ClassInfo, FeatureInfo, DataPath
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
    colors = ClassInfo.CLASS_COLORS
    class_names = ClassInfo.CLASS_NAMES 

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


def show_feature(feature_path: str, feature_id: List[int]) -> None:
    """
    Visualize one to three feature maps from a .npy feature tensor.

    The feature file must have shape (H, W, F).
    With F being the number of features.
    The feature values are stored in cste.py file (class FeatureInfo)

    Args:
        feature_path: Path to .npy feature file
        feature_id: List of feature indices to visualize (max length = 3)
    """
    # Load feature tensor
    features = np.load(feature_path)

    if features.ndim != 3:
        raise ValueError("Feature array must have shape (H, W, F)")

    h, w, f = features.shape

    if len(feature_id) == 0 or len(feature_id) > 3:
        raise ValueError("feature_id must contain 1 to 3 feature indices")

    for idx in feature_id:
        if idx < 0 or idx >= f:
            raise IndexError(f"Feature index {idx} out of bounds (0, {f - 1})")

    # Extract selected features
    selected = features[:, :, feature_id]

    # Normalize each feature independently (min-max)
    selected_norm = np.zeros_like(selected, dtype=np.float32)
    for i in range(selected.shape[2]):
        channel = selected[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            selected_norm[:, :, i] = (channel - min_val) / (max_val - min_val)
        else:
            selected_norm[:, :, i] = 0.0

    # Visualization
    plt.figure(figsize=(6, 6))

    if selected_norm.shape[2] == 1:
        # Single feature: heatmap
        sns.heatmap(
            selected_norm[:, :, 0],
            cmap="viridis",
            cbar=True,
            square=True
        )
        plt.title(f"Feature {FeatureInfo.FEATURE_NAMES[feature_id[0]]} Heatmap")

    else:
        # 2 or 3 features: RGB projection
        if selected_norm.shape[2] == 2:
            # Pad third channel with zeros
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0:2] = selected_norm
        else:
            rgb = selected_norm

        plt.imshow(rgb)
        plt.axis("off")
        feature_names = [FeatureInfo.FEATURE_NAMES[idx] for idx in feature_id]
        plt.title(f"Features {feature_names} in RGB projection.")

    plt.tight_layout()
    plt.show()


def show_all_features(feature_path: str, feature_info_cls = FeatureInfo) -> None:
    """
    Display a series of relevant visualizations for all feature types
    from a single .npy feature file (H, W, n_features).
    
    Args:
        feature_path: Path to the .npy feature file
        feature_info_cls: Class containing feature indices as attributes
    """
    feats = np.load(feature_path)
    
    # Dynamically generate groups using FeatureInfo class attributes
    groups = {
        "RGB": [feature_info_cls.RED, feature_info_cls.GREEN, feature_info_cls.BLUE],
        "HSV": [feature_info_cls.HUE, feature_info_cls.SATURATION, feature_info_cls.VALUE],
        "Grayscale & Blurs": [feature_info_cls.GRAYSCALE, feature_info_cls.BLUR_SIGMA_1, feature_info_cls.BLUR_SIGMA_5],
        "Gradient": [feature_info_cls.GRADIENT_MAG, feature_info_cls.GRADIENT_ORIENT],
        "Texture": [feature_info_cls.LOCAL_VARIANCE, feature_info_cls.LOCAL_ENTROPY, feature_info_cls.LBP],
        "Spectral": [feature_info_cls.NDVI, feature_info_cls.WATER_INDEX],
        "Geometry": [feature_info_cls.ANISOTROPY, feature_info_cls.CORNER_DENSITY]
    }
    
    for group_name, feature_ids in groups.items():
        log.info(f"Displaying features for group: {group_name}")
        show_feature(feature_path, feature_ids)



def csv_class_statistic_info(stat_csv_path:str= DataPath.CSV_CLASS_STATISTICS_TRAIN, save: str = None):
    """
    Plot statistics from a CSV containing class proportions and counts per image.

    Parameters:
        stat_csv_path (str): Path to CSV with header:
            img_id,img_path,label_path,prop_class_0..4,count_class_0..4
        save (str or None): If None, display the plots. Otherwise, save plots to the specified folder.
    """
    # Load CSV
    df = pd.read_csv(stat_csv_path)

    num_classes = ClassInfo.NUM_CLASSES  # As per header

    # ---------- 1. Distribution of proportions ----------
    plt.figure(figsize=(10, 6))
    for c in range(num_classes):
        sns.kdeplot(df[f"prop_class_{c}"], label=f"Class {ClassInfo.CLASS_NAMES[c]}", fill=True)
    plt.title("Distribution of class proportions per image")
    plt.xlabel("Proportion")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save:
        os.makedirs(save, exist_ok=True)
        plt.savefig(os.path.join(save, "class_proportion_distribution.png"))
        plt.close()
    else:
        plt.show()

    # ---------- 2. Number of images containing each class ----------
    presence = [(df[f"count_class_{c}"] > 0).sum() for c in range(num_classes)]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f"Class {ClassInfo.CLASS_NAMES[c]}" for c in range(num_classes)], y=presence)
    plt.title("Number of images containing each class")
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save, "images_per_class.png"))
        plt.close()
    else:
        plt.show()

    # ---------- 3. Histogram of absolute pixel counts ----------
    plt.figure(figsize=(12, 6))
    for c in range(num_classes):
        sns.histplot(df[f"count_class_{c}"], bins=50, label=f"Class {ClassInfo.CLASS_NAMES[c]}", kde=False, alpha=0.5)
    plt.title("Histogram of absolute pixel counts per class")
    plt.xlabel("Pixel count")
    plt.ylabel("Number of images")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save, "pixel_count_histograms.png"))
        plt.close()
    else:
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

    # show_img_labels(r"data/images/test/M-33-20-D-c-4-2_105.jpg",
    #                 r"data/results/predictions/M-33-20-D-c-4-2_105_pred.png")

    # show_feature(r"data/features/train/M-33-7-A-d-2-3_0.npy", [0, 1, 2])  # Example feature visualization

    csv_class_statistic_info(stat_csv_path=DataPath.CSV_CLASS_STATISTICS_TRAIN)