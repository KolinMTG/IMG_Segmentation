"""
Water class detection (Class 3) using probabilistic classifier.

Strategy:
- Extract comprehensive feature vector for each pixel
- Compute P(pixel | Water) using trained GMM
- Return posterior probability map
"""

import numpy as np
import pickle
from pathlib import Path

from general_processing import extract_features
from cste import ClassInfo


# ! Global model cache to avoid reloading on every call
_water_model = None


def _load_model(model_dir: str = "data/models/"):
    """
    Load trained Water model from disk.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Trained GMM model for Water class
    """
    global _water_model
    
    if _water_model is None:
        class_name = ClassInfo.CLASS_NAMES[3]  # "Water"
        model_path = Path(model_dir) / f"{class_name.lower()}_model.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                _water_model = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train_classifier.py first."
            )
    
    return _water_model


def process_water(img: np.ndarray) -> np.ndarray:
    """
    Generate water likelihood score map using probabilistic model.
    
    Detection strategy:
    1. Extract comprehensive features (color, texture, gradients, indices)
    2. Evaluate P(features | Water) using trained GMM
    3. Return normalized probability map
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale score map (H, W) in [0, 1] representing water likelihood
    """
    # Load trained model
    model = _load_model()
    
    # Extract features
    features = extract_features(img)
    H, W, N = features.shape
    
    # Reshape to (H*W, N)
    X = features.reshape(-1, N)
    
    # ! Compute log-likelihood P(x | Water)
    log_prob = model.score_samples(X)
    
    # Convert to probability
    prob = np.exp(log_prob)
    
    # Reshape to (H, W)
    prob_map = prob.reshape(H, W)
    
    # ! Normalize to [0, 1] for compatibility with existing pipeline
    prob_min = prob_map.min()
    prob_max = prob_map.max()
    
    if prob_max - prob_min > 1e-8:
        prob_map = (prob_map - prob_min) / (prob_max - prob_min)
    else:
        prob_map = np.zeros_like(prob_map)
    
    return prob_map.astype(np.float32)
