import numpy as np
from typing import List
from PIL import Image
import csv
import pandas as pd

from src.cste import DataPath, ClassInfo, ImgSelectionRule
from src.logger import get_logger



log = get_logger("data_utils")


def mask_0n_to_onehot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a mask with integer class labels (0 to n-1) into one-hot encoding.

    Parameters:
        mask (np.ndarray): Input 2D array of shape (H, W) with integer class labels.
        num_classes (int): Total number of classes (n).

    Returns:
        np.ndarray: One-hot encoded array of shape (H, W, num_classes).
    """
    # Ensure mask contains integers in the valid range
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError("Mask contains invalid class indices for the specified num_classes.")

    # Convert to one-hot encoding using NumPy advanced indexing
    onehot = np.eye(num_classes, dtype=np.uint8)[mask]  # shape (H, W, num_classes)
    
    return onehot





def compute_class_proportions(mask: np.ndarray, num_classes: int) -> list:
    """
    Compute the proportion of each class in a mask.

    Parameters:
        mask (np.ndarray): 2D array of shape (H, W) with integer class labels 0..num_classes-1
        num_classes (int): Total number of classes

    Returns:
        List[float]: Proportion of pixels belonging to each class. Sum = 1.0
    """
    total_pixels = mask.size
    proportions = [(mask == c).sum() / total_pixels for c in range(num_classes)]
    return proportions

def compute_class_pixel_counts(mask: np.ndarray, num_classes: int) -> list:
    """
    Compute the absolute number of pixels for each class in a mask.

    Parameters:
        mask (np.ndarray): 2D array of shape (H, W) with integer class labels 0..num_classes-1
        num_classes (int): Total number of classes

    Returns:
        List[int]: Number of pixels belonging to each class.
    """
    counts = [(mask == c).sum() for c in range(num_classes)]
    return counts


def save_class_statistics(mapping_csv_path: str = DataPath.CSV_MAPPING_TRAIN, output_csv: str = DataPath.CSV_CLASS_STATISTICS_TRAIN, num_classes: int=ClassInfo.NUM_CLASSES, mask_ext: str = ".png"):
    """
    Compute class proportions and count for all masks listed in a mapping CSV and save to CSV.

    Parameters:
        mapping_csv_path (str): Path to CSV with header 'img_id,img_path,label_path,feature_path'
        output_csv (str): Path to CSV file to save the results
        num_classes (int): Total number of classes
        mask_ext (str): Expected mask file extension (default ".png")
    """
    # Read the mapping CSV
    df_mapping = pd.read_csv(mapping_csv_path)

    # Prepare CSV header
    header = ["img_id", "img_path", "label_path"] + [f"prop_class_{i}" for i in range(num_classes)] + [f"count_class_{i}" for i in range(num_classes)]

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, row in df_mapping.iterrows():
            mask_path = row["label_path"]

            try:
                # Load mask

                mask = np.array(Image.open(mask_path), dtype=np.uint8)

                # Compute proportions
                props = compute_class_proportions(mask, num_classes)
                counts = compute_class_pixel_counts(mask, num_classes)

                # Write row
                writer.writerow([row["img_id"], row["img_path"], mask_path] + props + counts)

            except Exception as e:
                log.warning(f"Could not open mask '{mask_path}': {str(e)} in line {idx+2} of mapping CSV.")
                continue



def select_img(mapping_csv: str= DataPath.CSV_MAPPING_TRAIN, class_statistics_csv: str = DataPath.CSV_CLASS_STATISTICS_TRAIN, rule: dict = ImgSelectionRule.BUILDING_OR_ROAD_OR_WATER, output_csv: str = DataPath.CSV_SELECTED_IMAGES_TRAIN):
    """
    Select images from a mapping CSV based on class statistics rules with logic defined in the rule.

    Parameters:
        mapping_csv (str): Path to CSV with columns: img_id,img_path,label_path,feature_path
        class_statistics_csv (str): CSV with class counts/proportions for each image
        rule (dict): Dictionary defining selection rules.
                     Special key "__logic__" = "AND" or "OR" (default = "AND")
                     Other keys = class IDs (int) with condition strings, e.g., ">=1", "0"
        output_csv (str): Path to save filtered mapping CSV
    """
    # Load CSVs
    log.info(f"Selecting images from '{mapping_csv}' using rules: {rule}... and logic '{rule.get('logic', rule.get('__logic__', 'AND'))}'")
    df_mapping = pd.read_csv(mapping_csv)
    df_stats = pd.read_csv(class_statistics_csv)

    # Merge mapping with stats on label_path
    df = pd.merge(df_mapping, df_stats, on="label_path", how="inner")

    # Determine logic
    logic = rule.pop("__logic__", "AND").upper()
    if logic not in ["AND", "OR"]:
        raise ValueError("Logic in rule must be 'AND' or 'OR'")

    # Build individual masks
    masks = []
    for class_id, condition in rule.items():
        col_name = f"count_class_{class_id}"
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in CSV")
        current_mask = pd.Series(True, index=df.index)

        # Parse condition
        if condition.startswith(">="):
            value = int(condition[2:])
            current_mask &= df[col_name] >= value
        elif condition.startswith(">"):
            value = int(condition[1:])
            current_mask &= df[col_name] > value
        elif condition.startswith("<="):
            value = int(condition[2:])
            current_mask &= df[col_name] <= value
        elif condition.startswith("<"):
            value = int(condition[1:])
            current_mask &= df[col_name] < value
        elif condition == "0":
            current_mask &= df[col_name] == 0
        elif condition == ">=1":
            current_mask &= df[col_name] >= 1
        else:
            raise ValueError(f"Unknown condition: {condition} for class {class_id}")

        masks.append(current_mask)

    # Combine masks according to logic
    if logic == "AND":
        final_mask = pd.Series(True, index=df.index)
        for m in masks:
            final_mask &= m
    elif logic == "OR":
        final_mask = pd.Series(False, index=df.index)
        for m in masks:
            final_mask |= m

    # Filter and save
    mapping_cols = ["img_id_x", "img_path_x", "label_path", "feature_path"]
    df_filtered = df.loc[final_mask, mapping_cols]
    df_filtered.columns = ["img_id", "img_path", "label_path", "feature_path"]
    df_filtered.to_csv(output_csv, index=False)
    log.info(f"{len(df_filtered)} images selected and saved to {output_csv}")






if __name__ == "__main__":
    # save_class_statistics()
    select_img()
# ============================================================================
