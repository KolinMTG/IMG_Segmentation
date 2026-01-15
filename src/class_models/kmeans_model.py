"""K-Means clustering for unsupervised segmentation."""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from models.base_model import BaseSegmentationModel


class KMeansSegmentation(BaseSegmentationModel):
    """
    K-Means clustering for unsupervised image segmentation.
    
    Uses MiniBatchKMeans for memory efficiency.
    Does NOT use ground truth masks during training.
    """
    
    def __init__(
        self,
        num_classes: int,
        batch_size: int = 10000,
        n_init: int = 10,
        random_state: int = 42
    ):
        """
        Initialize K-Means model.
        
        Args:
            num_classes: Number of clusters (should match segmentation classes)
            batch_size: Batch size for mini-batch training
            n_init: Number of random initializations
            random_state: Random seed for reproducibility
        """
        super().__init__(num_classes, 'KMeans')
        
        self.batch_size = batch_size
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_to_class_map = None
        
        #! Store configuration
        self.config = {
            'num_classes': num_classes,
            'batch_size': batch_size,
            'n_init': n_init,
            'random_state': random_state
        }
        
        #! Initialize MiniBatchKMeans for memory efficiency
        self.model = MiniBatchKMeans(
            n_clusters=num_classes,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
            verbose=1
        )
        
        log.info(f"Initialized K-Means: clusters={num_classes}, batch_size={batch_size}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_fraction: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train K-Means clustering (unsupervised).
        
        Args:
            X_train: Training features, shape (N, H, W, C)
            y_train: Training masks (optional, used only for post-hoc mapping)
            X_val: Validation features (not used)
            y_val: Validation masks (not used)
            sample_fraction: Fraction of pixels to sample for clustering
            
        Returns:
            Training metrics dictionary
        """
        log.info(f"Training K-Means (unsupervised) with {sample_fraction*100:.1f}% pixel sampling")
        
        #! Reshape data to (num_pixels, num_features)
        N, H, W, C = X_train.shape
        X_train_flat = X_train.reshape(-1, C)
        
        #! Sample pixels for memory efficiency
        if sample_fraction < 1.0:
            n_samples = int(len(X_train_flat) * sample_fraction)
            indices = np.random.choice(len(X_train_flat), n_samples, replace=False)
            X_train_sampled = X_train_flat[indices]
            log.info(f"Sampled {n_samples:,} / {N*H*W:,} pixels for clustering")
        else:
            X_train_sampled = X_train_flat
        
        #! Train K-Means (unsupervised)
        log.info("Fitting K-Means clusters...")
        self.model.fit(X_train_sampled)
        
        #! Compute inertia and silhouette score
        inertia = self.model.inertia_
        
        #! Silhouette score on a sample (expensive for large datasets)
        if len(X_train_sampled) > 50000:
            sample_idx = np.random.choice(len(X_train_sampled), 50000, replace=False)
            X_silhouette = X_train_sampled[sample_idx]
        else:
            X_silhouette = X_train_sampled
        
        labels = self.model.predict(X_silhouette)
        silhouette = silhouette_score(X_silhouette, labels, sample_size=min(10000, len(X_silhouette)))
        
        log.info(f"Inertia: {inertia:.2f}")
        log.info(f"Silhouette score: {silhouette:.4f}")
        
        #! Optional: Create mapping from clusters to ground truth classes
        if y_train is not None:
            self.cluster_to_class_map = self._map_clusters_to_classes(
                X_train, y_train
            )
            log.info(f"Cluster to class mapping: {self.cluster_to_class_map}")
        
        metrics = {
            'inertia': inertia,
            'silhouette_score': silhouette,
            'cluster_to_class_map': self.cluster_to_class_map
        }
        
        return metrics
    
    def _map_clusters_to_classes(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[int, int]:
        """
        Map cluster IDs to ground truth classes (post-hoc).
        
        For each cluster, assign the majority ground truth class.
        
        Args:
            X: Features, shape (N, H, W, C)
            y: Ground truth masks, shape (N, H, W)
            
        Returns:
            Dictionary mapping cluster_id -> class_id
        """
        log.info("Creating cluster-to-class mapping...")
        
        #! Predict cluster assignments
        cluster_labels = self.predict(X)
        
        #! For each cluster, find the most common ground truth class
        mapping = {}
        for cluster_id in range(self.num_classes):
            mask = (cluster_labels == cluster_id)
            if mask.sum() == 0:
                mapping[cluster_id] = cluster_id  # Default to identity
                continue
            
            #! Find majority class in this cluster
            cluster_pixels = y[mask]
            counts = np.bincount(cluster_pixels.flatten(), minlength=self.num_classes)
            majority_class = np.argmax(counts)
            mapping[cluster_id] = int(majority_class)
        
        return mapping
    
    def predict(self, X: np.ndarray, use_mapping: bool = True) -> np.ndarray:
        """
        Make predictions on images.
        
        Args:
            X: Input features, shape (N, H, W, C)
            use_mapping: If True and mapping exists, map clusters to classes
            
        Returns:
            Predicted masks, shape (N, H, W)
        """
        N, H, W, C = X.shape
        
        #! Reshape to pixels
        X_flat = X.reshape(-1, C)
        
        #! Predict cluster IDs
        log.info(f"Predicting on {N} images...")
        cluster_ids = self.model.predict(X_flat)
        
        #! Apply mapping if available
        if use_mapping and self.cluster_to_class_map is not None:
            cluster_ids = np.array([
                self.cluster_to_class_map[cid] for cid in cluster_ids
            ])
        
        #! Reshape back to images
        y_pred = cluster_ids.reshape(N, H, W)
        
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
        model_path = save_path / 'kmeans.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        log.info(f"Saved model to {model_path}")
        
        #! Save cluster mapping if exists
        if self.cluster_to_class_map is not None:
            mapping_path = save_path / 'cluster_mapping.pkl'
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.cluster_to_class_map, f)
            log.info(f"Saved cluster mapping to {mapping_path}")
        
        #! Save configuration
        self.config['has_mapping'] = self.cluster_to_class_map is not None
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
        model_path = Path(save_dir) / 'kmeans.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        log.info(f"Loaded model from {model_path}")
        
        #! Load cluster mapping if exists
        if self.config.get('has_mapping', False):
            mapping_path = Path(save_dir) / 'cluster_mapping.pkl'
            with open(mapping_path, 'rb') as f:
                self.cluster_to_class_map = pickle.load(f)
            log.info(f"Loaded cluster mapping from {mapping_path}")
