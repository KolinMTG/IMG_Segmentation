"""
Woodland class detection (Class 2) using Bayesian probabilistic classifier.

Strategy:
- Extract 19-dimensional feature vector per pixel
- Compute normalized posterior P(Woodland | x) using GMM and class priors
- Return probability map compatible with aggregation pipeline
"""

import numpy as np

from classifier_inference import compute_single_class_probability


def process_woodland(img: np.ndarray) -> np.ndarray:
    """
    Generate woodland likelihood score map using Bayesian probabilistic model.
    
    Detection strategy:
    1. Extract comprehensive features (color, texture, gradients, indices, geometry)
    2. Evaluate P(x | Woodland) using trained GMM
    3. Apply Bayes' rule with class priors: P(Woodland | x) = P(x | Woodland) P(Woodland) / Z
    4. Return normalized probability map
    
    ! Probability is normalized across all classes (sum-to-one constraint)
    ! High NDVI, texture variance, and green channel are strong woodland indicators
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale probability map (H, W) in [0, 1] representing woodland likelihood
    """
    return compute_single_class_probability(img, class_id=2)
