"""Random Forest classifier for pixel-wise segmentation."""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseSegmentationModel


class RandomForestSegmentation(BaseSegmentationModel):
    """
    Random Forest classifier for pixel-wise segmentation.
    
    Trains on individual pixels with their feature vectors.
    Optimized for CPU with multiprocessing.
    """
    
    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = 20,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize Random Forest model.
        
        Args:
            num_classes: Number of segmentation classes
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            n_jobs: Number of parallel jobs (-1 = use all cores)
            random_state: Random seed for reproducibility
        """
        super().__init__(num_classes, 'RandomForest')
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        #! Store configuration
        self.config = {
            'num_classes': num_classes,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': n_jobs,
            'random_state': random_state
        }
        
        #! Initialize model with balanced class weights
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            verbose=1
        )
        
        log.info(f"Initialized Random Forest: trees={n_estimators}, depth={max_depth}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_fraction: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train Random Forest on pixel features.
        
        Args:
            X_train: Training features, shape (N, H, W, C)
            y_train: Training masks, shape (N, H, W)
            X_val: Validation features (not used, for API consistency)
            y_val: Validation masks (not used, for API consistency)
            sample_fraction: Fraction of pixels to sample for training (memory efficient)
            
        Returns:
            Training metrics dictionary
        """
        log.info(f"Training Random Forest with {sample_fraction*100:.1f}% pixel sampling")
        
        #! Reshape data to (num_pixels, num_features)
        N, H, W, C = X_train.shape
        X_train_flat = X_train.reshape(-1, C)
        y_train_flat = y_train.reshape(-1)
        
        #! Sample pixels for memory efficiency (16GB RAM constraint)
        if sample_fraction < 1.0:
            n_samples = int(len(X_train_flat) * sample_fraction)
            indices = np.random.choice(len(X_train_flat), n_samples, replace=False)
            X_train_flat = X_train_flat[indices]
            y_train_flat = y_train_flat[indices]
            log.info(f"Sampled {n_samples:,} / {N*H*W:,} pixels for training")
        
        #! Train model (CPU-optimized with multiprocessing)
        log.info("Training Random Forest (this may take a few minutes)...")
        self.model.fit(X_train_flat, y_train_flat)
        
        #! Compute training accuracy
        train_acc = self.model.score(X_train_flat, y_train_flat)
        log.info(f"Training accuracy: {train_acc:.4f}")
        
        #! Feature importance
        feature_importance = self.model.feature_importances_
        log.info(f"Feature importance: {feature_importance}")
        
        metrics = {
            'train_accuracy': train_acc,
            'feature_importance': feature_importance.tolist(),
            'n_samples_used': len(X_train_flat)
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on images.
        
        Args:
            X: Input features, shape (N, H, W, C)
            
        Returns:
            Predicted masks, shape (N, H, W)
        """
        N, H, W, C = X.shape
        
        #! Reshape to pixels
        X_flat = X.reshape(-1, C)
        
        #! Predict
        log.info(f"Predicting on {N} images...")
        y_pred_flat = self.model.predict(X_flat)
        
        #! Reshape back to images
        y_pred = y_pred_flat.reshape(N, H, W)
        
        return y_pred
    
    def save(self, save_dir: str) -> None:
        """
        Save model and configuration.
        
        Args:
            save_dir: Directory to save model artifacts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        #! Save model with pickle
        model_path = save_path / 'random_forest.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        log.info(f"Saved model to {model_path}")
        
        #! Save configuration
        self._save_config(save_dir)
    
    def load(self, save_dir: str) -> None:
        """
        Load model and configuration.
        
        Args:
            save_dir: Directory containing model artifacts
        """
        #! Load configuration
        self.config = self._load_config(save_dir)
        
        #! Load model
        model_path = Path(save_dir) / 'random_forest.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        log.info(f"Loaded model from {model_path}")
