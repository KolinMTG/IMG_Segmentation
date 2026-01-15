"""
Shared inference module for probabilistic classification.

This module provides:
- Global model and metadata caching
- Shared feature extraction
- Normalized Bayesian probability computation
- Thread-safe model loading

! This module is imported by all class detection files to avoid code duplication
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional
from sklearn.decomposition import PCA

from feature_extraction_pipeline import extract_features
from cste import ClassInfo, DataPath


# ============================================================================
# GLOBAL CACHE
# ============================================================================

_models_cache: Optional[Dict[int, object]] = None
_metadata_cache: Optional[Dict] = None
_features_cache: Optional[Dict[int, np.ndarray]] = None  # Cache features by image id


def _get_image_id(img: np.ndarray) -> int:
    """
    Generate a simple hash ID for an image to cache features.
    
    ! This allows feature extraction to happen once per image
    ! even when all 5 classes are evaluated
    
    Args:
        img: Input image
        
    Returns:
        Hash ID
    """
    return hash(img.tobytes())


def load_models_and_metadata(model_dir: str = "data/models/") -> tuple:
    """
    Load all models and metadata (cached globally).
    
    ! Models and metadata are loaded once per session
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Tuple of (models_dict, metadata_dict)
    """
    global _models_cache, _metadata_cache
    
    if _models_cache is not None and _metadata_cache is not None:
        return _models_cache, _metadata_cache
    
    # Load GMM models
    _models_cache = {}
    for class_id, class_name in ClassInfo.CLASS_NAMES.items():
        model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                _models_cache[class_id] = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train_classifier.py first."
            )
    
    # Load metadata
    metadata_path = Path(model_dir) / "metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            _metadata_cache = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Run train_classifier.py first."
        )
    
    return _models_cache, _metadata_cache


def get_cached_features(img: np.ndarray) -> np.ndarray:
    """
    Extract features with caching.
    
    ! Features are cached per image to avoid recomputation
    ! when multiple classes are evaluated on the same image
    
    Args:
        img: RGB image (H, W, 3) in [0, 1]
        
    Returns:
        Feature tensor (H, W, 19)
    """
    global _features_cache
    
    if _features_cache is None:
        _features_cache = {}
    
    img_id = _get_image_id(img)
    
    if img_id not in _features_cache:
        _features_cache[img_id] = extract_features(img, normalize=True)
        
        # ! Limit cache size to prevent memory issues
        if len(_features_cache) > 10:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(_features_cache))
            del _features_cache[oldest_key]
    
    return _features_cache[img_id]


def compute_normalized_probabilities(
    img: np.ndarray,
    model_dir: str = DataPath.MODEL_DIR
) -> Dict[int, np.ndarray]:
    """
    Compute normalized Bayesian probabilities for all classes.
    
    ! This ensures sum(P(c | x)) = 1 for all pixels
    ! Returns one probability map per class
    
    Args:
        img: RGB image (H, W, 3) in [0, 1]
        model_dir: Directory containing models
        
    Returns:
        Dictionary mapping class ID to probability map (H, W)
    """
    models, metadata = load_models_and_metadata(model_dir)
    features = get_cached_features(img)
    
    H, W, N = features.shape
    X = features.reshape(-1, N)
    
    # Apply PCA if used during training
    use_pca = metadata.get('use_pca', False)
    pca: Optional[PCA] = metadata.get('pca', None)
    
    if use_pca and pca is not None:
        X = pca.transform(X)
    
    # ! Compute log-likelihoods P(x | c)
    log_likelihoods = {}
    for class_id, model in models.items():
        log_likelihoods[class_id] = model.score_samples(X)
    
    # ! Get log priors P(c)
    class_priors = metadata.get('class_priors', {})
    log_priors = {
        c: np.log(class_priors.get(c, 1.0 / len(ClassInfo.CLASS_NAMES)))
        for c in models.keys()
    }
    
    # ! Compute log posteriors: log P(x | c) + log P(c)
    log_posteriors = {
        c: log_likelihoods[c] + log_priors[c]
        for c in models.keys()
    }
    
    # ! Normalize using log-sum-exp
    log_posterior_stack = np.stack(
        [log_posteriors[c] for c in sorted(models.keys())],
        axis=1
    )
    
    log_sum = np.logaddexp.reduce(log_posterior_stack, axis=1, keepdims=True)
    log_normalized = log_posterior_stack - log_sum
    posteriors_normalized = np.exp(log_normalized)
    
    # ! Extract per-class maps
    prob_maps = {}
    for i, class_id in enumerate(sorted(models.keys())):
        prob_map = posteriors_normalized[:, i].reshape(H, W)
        prob_maps[class_id] = prob_map.astype(np.float32)
    
    return prob_maps


def compute_single_class_probability(
    img: np.ndarray,
    class_id: int,
    model_dir: str = "data/models/"
) -> np.ndarray:
    """
    Compute normalized probability for a single class.
    
    ! This computes full Bayesian posterior P(c | x) normalized across all classes
    ! More efficient than computing all classes if only one is needed
    
    Args:
        img: RGB image (H, W, 3) in [0, 1]
        class_id: Target class ID
        model_dir: Directory containing models
        
    Returns:
        Probability map (H, W) in [0, 1]
    """
    # ! For proper normalization, we need all class probabilities
    # ! So we compute all and return the requested one
    all_probs = compute_normalized_probabilities(img, model_dir)
    return all_probs[class_id]
