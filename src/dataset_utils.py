"""Dataset utilities for satellite image segmentation."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
from logger import get_logger
log = get_logger("dataset_utils")

class SegmentationDataset:
    """
    Dataset loader for satellite image segmentation.
    
    Loads handcrafted features and segmentation masks from disk.
    Supports feature selection and one-hot encoding.
    """
    
    def __init__(
        self,
        csv_path: str,
        num_classes: int,
        feature_ids: Optional[List[int]] = None,
        one_hot: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with mask_path and feature_path columns
            num_classes: Number of segmentation classes
            feature_ids: List of feature indices to load (None = all features)
            one_hot: If True, convert masks to one-hot encoding
        """
        self.df = pd.read_csv(csv_path)
        self.num_classes = num_classes
        self.feature_ids = feature_ids
        self.one_hot = one_hot
        
        #! Validate required columns
        required_cols = {'mask_path', 'feature_path'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        log.info(f"Loaded dataset with {len(self.df)} samples")
        log.info(f"Classes: {num_classes}, One-hot: {one_hot}")
        if feature_ids:
            log.info(f"Selected features: {feature_ids}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, mask)
            - features: shape (H, W, C) where C = len(feature_ids)
            - mask: shape (H, W) or (H, W, N) if one_hot=True
        """
        row = self.df.iloc[idx]
        
        #! Load feature array
        features = np.load(row['feature_path'])  # (H, W, F)
        
        #! Select subset of features if specified
        if self.feature_ids is not None:
            features = features[..., self.feature_ids]
        
        #! Load mask
        mask = np.load(row['mask_path'])  # (H, W)
        
        #! Convert to one-hot if requested
        if self.one_hot:
            mask = self._to_one_hot(mask)
        
        return features, mask
    
    def _to_one_hot(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert integer mask to one-hot encoding.
        
        Args:
            mask: Integer mask of shape (H, W)
            
        Returns:
            One-hot mask of shape (H, W, N)
        """
        H, W = mask.shape
        one_hot = np.zeros((H, W, self.num_classes), dtype=np.float32)
        for c in range(self.num_classes):
            one_hot[..., c] = (mask == c).astype(np.float32)
        return one_hot
    
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (features_batch, masks_batch)
            - features_batch: shape (B, H, W, C)
            - masks_batch: shape (B, H, W) or (B, H, W, N)
        """
        batch_features = []
        batch_masks = []
        
        for idx in indices:
            features, mask = self[idx]
            batch_features.append(features)
            batch_masks.append(mask)
        
        return np.array(batch_features), np.array(batch_masks)
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all data into memory.
        
        WARNING: Use only if dataset fits in memory.
        
        Returns:
            Tuple of (all_features, all_masks)
        """
        log.info("Loading all data into memory...")
        return self.get_batch(list(range(len(self))))
    
    def compute_class_weights(self) -> np.ndarray:
        """
        Compute class weights for imbalanced datasets.
        
        Returns:
            Array of class weights, shape (num_classes,)
        """
        log.info("Computing class weights...")
        
        class_counts = np.zeros(self.num_classes)
        total_pixels = 0
        
        for idx in range(len(self)):
            mask = np.load(self.df.iloc[idx]['mask_path'])
            for c in range(self.num_classes):
                class_counts[c] += np.sum(mask == c)
            total_pixels += mask.size
        
        #! Compute inverse frequency weights
        weights = total_pixels / (self.num_classes * class_counts)
        
        log.info(f"Class distribution: {class_counts}")
        log.info(f"Class weights: {weights}")
        
        return weights


def create_train_val_split(
    csv_path: str,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Split dataset into train and validation sets.
    
    Args:
        csv_path: Path to full dataset CSV
        val_split: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_csv_path, val_csv_path)
    """
    df = pd.read_csv(csv_path)
    
    #! Shuffle and split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_split))
    
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    #! Save splits
    base_path = Path(csv_path).parent
    train_path = str(base_path / 'train.csv')
    val_path = str(base_path / 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    log.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    return train_path, val_path
