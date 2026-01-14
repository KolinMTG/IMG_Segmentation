"""
Training module for probabilistic pixel-level classification.

This module trains one Gaussian Mixture Model (GMM) per class using
labeled training data. Models are saved to disk for inference.

Training workflow:
1. Load training images and ground truth masks
2. Extract features for all pixels
3. Sample pixels from each class
4. Fit GMM for each class
5. Save models to disk
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.mixture import GaussianMixture
import cv2

from general_processing import extract_features
from cste import ClassInfo, DataPath


class PixelClassifier:
    """
    Probabilistic pixel classifier using Gaussian Mixture Models.
    
    ! Each class is modeled as a GMM in feature space
    ! GMM allows for multi-modal distributions (e.g., dark vs bright water)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        max_samples_per_class: int = 50000
    ):
        """
        Initialize classifier.
        
        Args:
            n_components: Number of Gaussian components per class GMM
            max_samples_per_class: Maximum pixels to sample per class for training
        """
        self.n_components = n_components
        self.max_samples_per_class = max_samples_per_class
        self.models: Dict[int, GaussianMixture] = {}
        self.n_features: int = 17  # From extract_features()
    
    def _sample_pixels(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_id: int
    ) -> np.ndarray:
        """
        Sample pixels belonging to a specific class.
        
        Args:
            features: Feature tensor (H, W, N)
            labels: Ground truth labels (H, W)
            class_id: Target class ID
            
        Returns:
            Feature matrix (M, N) for sampled pixels
        """
        # Find pixels belonging to class
        mask = (labels == class_id)
        class_pixels = features[mask]
        
        # ! Randomly sample if too many pixels
        if len(class_pixels) > self.max_samples_per_class:
            indices = np.random.choice(
                len(class_pixels),
                self.max_samples_per_class,
                replace=False
            )
            class_pixels = class_pixels[indices]
        
        return class_pixels
    
    def train(
        self,
        image_paths: List[str],
        label_paths: List[str]
    ) -> None:
        """
        Train GMM models for all classes.
        
        Args:
            image_paths: List of training image file paths
            label_paths: List of corresponding label mask file paths
        """
        # ! Collect training samples for all classes
        class_samples: Dict[int, List[np.ndarray]] = {
            c: [] for c in ClassInfo.CLASS_NAMES.keys()
        }
        
        # Process each training image
        for img_path, label_path in zip(image_paths, label_paths):
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Load labels
            labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract features
            features = extract_features(img)
            
            # Sample pixels for each class
            for class_id in ClassInfo.CLASS_NAMES.keys():
                samples = self._sample_pixels(features, labels, class_id)
                if len(samples) > 0:
                    class_samples[class_id].append(samples)
        
        # ! Train one GMM per class
        for class_id, class_name in ClassInfo.CLASS_NAMES.items():
            # Concatenate all samples for this class
            if len(class_samples[class_id]) == 0:
                continue
            
            X = np.vstack(class_samples[class_id])
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='diag',  # Diagonal covariance for efficiency
                max_iter=100,
                random_state=42
            )
            
            gmm.fit(X)
            self.models[class_id] = gmm
    
    def save_models(self, model_dir: str = "data/models/") -> None:
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save model files
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        for class_id, model in self.models.items():
            class_name = ClassInfo.CLASS_NAMES[class_id]
            model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    def load_models(self, model_dir: str = "data/models/") -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        self.models = {}
        
        for class_id, class_name in ClassInfo.CLASS_NAMES.items():
            model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[class_id] = pickle.load(f)
    
    def predict_class_probability(
        self,
        features: np.ndarray,
        class_id: int
    ) -> np.ndarray:
        """
        Compute posterior probability P(class | features) for each pixel.
        
        ! Uses Bayes' rule with uniform prior:
        ! P(class | x) ∝ P(x | class)
        
        Args:
            features: Feature tensor (H, W, N)
            class_id: Target class ID
            
        Returns:
            Probability map (H, W) in [0, 1]
        """
        H, W, N = features.shape
        
        # Reshape to (H*W, N)
        X = features.reshape(-1, N)
        
        # Get log-likelihood from GMM
        model = self.models[class_id]
        log_prob = model.score_samples(X)
        
        # ! Convert log-likelihood to probability
        # ! Use softmax-like normalization across all classes
        prob = np.exp(log_prob)
        
        # Reshape back to (H, W)
        prob_map = prob.reshape(H, W)
        
        # Normalize to [0, 1]
        prob_min = prob_map.min()
        prob_max = prob_map.max()
        
        if prob_max - prob_min > 1e-8:
            prob_map = (prob_map - prob_min) / (prob_max - prob_min)
        else:
            prob_map = np.zeros_like(prob_map)
        
        return prob_map.astype(np.float32)


def train_models_from_directory(
    img_dir: str = DataPath.IMG_TRAIN,
    label_dir: str = DataPath.LABEL_TRAIN,
    model_dir: str = "data/models/",
    n_components: int = 3
) -> None:
    """
    Train models using all images in training directory.
    
    Args:
        img_dir: Directory containing training images
        label_dir: Directory containing training labels
        model_dir: Directory to save trained models
        n_components: Number of GMM components per class
    """
    # Collect all training files
    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    label_paths = sorted(Path(label_dir).glob("*.png"))
    
    # Convert to strings
    img_paths = [str(p) for p in img_paths]
    label_paths = [str(p) for p in label_paths]
    
    # Train classifier
    classifier = PixelClassifier(n_components=n_components)
    classifier.train(img_paths, label_paths)
    
    # Save models
    classifier.save_models(model_dir)


if __name__ == "__main__":
    # ! Execute training pipeline
    train_models_from_directory()
