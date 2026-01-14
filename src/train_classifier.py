"""
Training module for Bayesian probabilistic pixel-level classification.

This module trains one Gaussian Mixture Model (GMM) per class using
labeled training data. Models are saved to disk for inference.

Key features:
- Adaptive sampling per class based on class frequency
- Optional PCA dimensionality reduction
- Class priors P(c) for proper Bayesian inference
- Diagonal covariance GMMs for efficiency

Training workflow:
1. Load training images and ground truth masks
2. Extract features for all pixels
3. Compute class priors from training data
4. Sample pixels adaptively per class
5. Optional: Apply PCA for dimensionality reduction
6. Fit GMM for each class
7. Save models and metadata to disk
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import cv2
import pandas as pd


from cste import ClassInfo, DataPath, ProcessingConfig, CSVKeys
from logger import get_logger
from tqdm import tqdm

import os

log = get_logger("train_classifier.log")


class BayesianPixelClassifier:
    """
    Bayesian probabilistic pixel classifier using Gaussian Mixture Models.
    
    ! Each class is modeled as a GMM in feature space: P(x | class=c)
    ! Class priors P(c) are estimated from training data frequencies
    ! Final probabilities: P(c | x) = P(x | c) P(c) / sum_k P(x | k) P(k)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        use_pca: bool = False,
        n_pca_components: int = 12,
        adaptive_sampling: bool = True
    ):
        """
        Initialize Bayesian classifier.
        
        Args:
            n_components: Number of Gaussian components per class GMM
            use_pca: Whether to apply PCA dimensionality reduction
            n_pca_components: Number of PCA components if use_pca=True
            adaptive_sampling: Adaptive sampling based on class frequency
        """
        self.n_components = n_components
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.adaptive_sampling = adaptive_sampling
        
        self.models: Dict[int, GaussianMixture] = {}
        self.class_priors: Dict[int, float] = {}
        self.pca: Optional[PCA] = None
        self.n_features: int = 19
        log.info("Initialized BayesianPixelClassifier")
    
    def _compute_class_priors(
        self,
        label_paths: List[str]
    ) -> None:
        """
        Compute class priors P(c) from training data.
        
        ! Priors are class pixel frequencies across all training images
        
        Args:
            label_paths: List of corresponding label mask file paths
        """
        log.info(f"Computing class priors from training data, number of images: {len(label_paths)}")
        class_counts = {c: 0 for c in ClassInfo.CLASS_NAMES.keys()}
        total_pixels = 0
        
        for label_path in tqdm(label_paths):
            labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if labels is None:
                log.warning(f"Label file not found or unreadable: {label_path}, skipping.")
                continue
            
            for class_id in ClassInfo.CLASS_NAMES.keys():
                class_counts[class_id] += np.sum(labels == class_id)
            
            total_pixels += labels.size
        
        # Compute priors with Laplace smoothing
        for class_id in ClassInfo.CLASS_NAMES.keys():
            self.class_priors[class_id] = (
                (class_counts[class_id] + 1) / (total_pixels + len(ClassInfo.CLASS_NAMES))
            )
        log.info(f"Computed class priors: {self.class_priors}")
    
    def _adaptive_sample_size(self, class_id: int) -> int:
        """
        Compute adaptive sample size based on class prior.
        
        ! Rare classes get more samples per image to balance training
        
        Args:
            class_id: Target class ID
            
        Returns:
            Number of samples to draw for this class
        """
        if not self.adaptive_sampling:
            return 50000
        
        # ! Base sample size inversely proportional to prior
        # ! Ensures balanced representation across classes
        base_samples = 50000
        prior = self.class_priors.get(class_id, 0.2)
        
        # Scale inversely: rare classes get more samples
        sample_size = int(base_samples / (prior * 5 + 0.1))
        
        # Clamp to reasonable range
        return min(max(sample_size, 10000), 100000)
    
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
        mask = (labels == class_id)
        class_pixels = features[mask]
        
        if len(class_pixels) == 0:
            return np.array([]).reshape(0, features.shape[2])
        
        # ! Adaptive sampling based on class frequency
        max_samples = self._adaptive_sample_size(class_id)
        
        if len(class_pixels) > max_samples:
            indices = np.random.choice(
                len(class_pixels),
                max_samples,
                replace=False
            )
            class_pixels = class_pixels[indices]
        
        return class_pixels
    
    def train(
    self,
    image_paths: List[str],
    label_paths: List[str],
    feature_paths: List[str],
) -> None:
        """
        Train GMM models for all classes with proper Bayesian setup.
        Handles feature downsampling, label resizing, and errors robustly.

        Args:
            image_paths: List[str]
            label_paths: List[str]
            feature_paths: List[str]
        """
        log.info("Starting training of BayesianPixelClassifier")
        log.info(f"Images: {len(image_paths)}, Labels: {len(label_paths)}, Features: {len(feature_paths)}")
        # Step 1: Compute class priors
        self._compute_class_priors(label_paths)

        # Step 2: Initialize storage for class samples
        class_samples: Dict[int, List[np.ndarray]] = {c: [] for c in ClassInfo.CLASS_NAMES.keys()}

        for i, (img_path, label_path, feature_path) in enumerate(zip(image_paths, label_paths, feature_paths)):
            log.info(f"Processing {img_path} ({i+1}/{len(image_paths)})")

            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    log.warning(f"Image file not found or unreadable: {img_path}, skipping.")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # Load label
                labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if labels is None:
                    log.warning(f"Label file not found or unreadable: {label_path}, skipping.")
                    continue

                # Load feature
                if feature_path is None or not os.path.exists(feature_path):
                    log.warning(f"Feature file not found or unreadable: {feature_path}, skipping {img_path}.")
                    continue

                try:
                    feature = np.load(feature_path)
                except Exception as e:
                    log.warning(f"Failed to load feature file {feature_path}: {e}, skipping {img_path}.")
                    continue

                # Downsample labels if needed to match feature size
                if labels.shape[:2] != feature.shape[:2]:
                    log.warning(
                        f"Label shape {labels.shape[:2]} does not match feature shape {feature.shape[:2]} for {img_path}. Resizing."
                    )
                    labels = cv2.resize(labels, (feature.shape[1], feature.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Sample pixels for each class
                for class_id in ClassInfo.CLASS_NAMES.keys():
                    samples = self._sample_pixels(feature, labels, class_id)
                    if len(samples) > 0:
                        class_samples[class_id].append(samples)

            except Exception as e:
                log.warning(f"Unexpected error processing {img_path}: {e}")
                continue

        # Step 3: Optional PCA
        if self.use_pca:
            all_samples = [np.vstack(slist) for slist in class_samples.values() if len(slist) > 0]
            if len(all_samples) > 0:
                all_data = np.vstack(all_samples)
                if len(all_data) > ProcessingConfig.PCA_SUBSAMPLE_SIZE:
                    indices = np.random.choice(len(all_data), ProcessingConfig.PCA_SUBSAMPLE_SIZE, replace=False)
                    all_data = all_data[indices]
                self.pca = PCA(n_components=self.n_pca_components, whiten=True)
                self.pca.fit(all_data)
                log.info("PCA fitting completed")

        # Step 4: Train GMM per class
        for class_id, class_name in ClassInfo.CLASS_NAMES.items():
            if len(class_samples[class_id]) == 0:
                log.warning(f"No samples collected for class {class_name} ({class_id}), skipping GMM training.")
                continue

            X = np.vstack(class_samples[class_id])
            if self.use_pca and self.pca is not None:
                X = self.pca.transform(X)

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='diag',
                max_iter=100,
                random_state=42,
                reg_covar=1e-6
            )
            gmm.fit(X)
            self.models[class_id] = gmm
            log.info(f"GMM trained for class {class_name} ({class_id})")


    
    def save_models(self, model_dir: str = DataPath.MODEL_DIR) -> None:
        """
        Save trained models and metadata to disk.
        
        Args:
            model_dir: Directory to save model files
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save each GMM model
        for class_id, model in self.models.items():
            class_name = ClassInfo.CLASS_NAMES[class_id]
            model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # ! Save metadata (priors, PCA, etc.)
        metadata = {
            'class_priors': self.class_priors,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'n_features': self.n_features
        }
        
        metadata_path = Path(model_dir) / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_models(self, model_dir: str = DataPath.MODEL_DIR) -> None:
        """
        Load trained models and metadata from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        self.models = {}
        
        # Load GMM models
        for class_id, class_name in ClassInfo.CLASS_NAMES.items():
            model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[class_id] = pickle.load(f)
        
        # Load metadata
        metadata_path = Path(model_dir) / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.class_priors = metadata.get('class_priors', {})
                self.pca = metadata.get('pca', None)
                self.use_pca = metadata.get('use_pca', False)
                self.n_features = metadata.get('n_features', 19)
    
    def predict_class_probabilities(
        self,
        features: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute normalized Bayesian posterior probabilities for all classes.
        
        ! Implements: P(c | x) = P(x | c) P(c) / sum_k P(x | k) P(k)
        
        Args:
            features: Feature tensor (H, W, N)
            
        Returns:
            Dictionary mapping class ID to probability map (H, W)
        """
        H, W, N = features.shape
        X = features.reshape(-1, N)
        
        # Apply PCA if used during training
        if self.use_pca and self.pca is not None:
            X = self.pca.transform(X)
        
        # ! Compute log P(x | c) for all classes
        log_likelihoods = {}
        for class_id, model in self.models.items():
            log_likelihoods[class_id] = model.score_samples(X)
        
        # ! Add log priors: log P(c)
        log_priors = {
            c: np.log(self.class_priors.get(c, 1.0 / len(ClassInfo.CLASS_NAMES)))
            for c in self.models.keys()
        }
        
        # ! Compute log P(x | c) P(c) = log P(x | c) + log P(c)
        log_posteriors = {
            c: log_likelihoods[c] + log_priors[c]
            for c in self.models.keys()
        }
        
        # ! Normalize to get P(c | x)
        # ! Use log-sum-exp trick for numerical stability
        log_posterior_stack = np.stack(
            [log_posteriors[c] for c in sorted(self.models.keys())],
            axis=1
        )
        
        # Log-sum-exp normalization
        log_sum = np.logaddexp.reduce(log_posterior_stack, axis=1, keepdims=True)
        log_normalized = log_posterior_stack - log_sum
        
        # Convert to probabilities
        posteriors_normalized = np.exp(log_normalized)
        
        # ! Extract per-class probability maps
        prob_maps = {}
        for i, class_id in enumerate(sorted(self.models.keys())):
            prob_map = posteriors_normalized[:, i].reshape(H, W)
            prob_maps[class_id] = prob_map.astype(np.float32)
        
        return prob_maps


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def train_models_from_directory(
    csv_mapping: str = DataPath.CSV_MAPPING_TRAIN,
    model_dir: str = DataPath.MODEL_DIR,
    n_components: int = 3,
    use_pca: bool = False,
    n_pca_components: int = 12,
) -> None:
    """
    Train models using all images in training directory.
    
    Args:
        csv_mapping: Path to CSV file mapping images to labels and features
        model_dir: Directory to save trained models
        n_components: Number of GMM components per class
        use_pca: Whether to use PCA dimensionality reduction
        n_pca_components: Number of PCA components
    """
    log.info("train_models_from_directory: Starting training of BayesianPixelClassifier")

    # Gather file paths from CSV mapping
    df = pd.read_csv(csv_mapping)
    img_paths = df[CSVKeys.IMAGE_PATH].tolist()
    label_paths = df[CSVKeys.LABEL_PATH].tolist()
    feature_paths = df[CSVKeys.FEATURE_PATH].tolist()

    
    # Error handling for data availability
    if len(img_paths) == 0 or len(label_paths) != len(img_paths):
        log.error("train_models_from_directory: No training data found or mismatch in image/label counts")
        return
    log.info(f"train_models_from_directory: Found {len(img_paths)} training images")
    
    classifier = BayesianPixelClassifier(
        n_components=n_components,
        use_pca=use_pca,
        n_pca_components=n_pca_components,
        adaptive_sampling=True
    )
    
    classifier.train(img_paths, label_paths, feature_paths)
    classifier.save_models(model_dir)


if __name__ == "__main__":
    train_models_from_directory()
