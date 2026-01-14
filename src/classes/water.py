"""
Water class detection (Class 3) using Bayesian probabilistic classifier.

Strategy:
- Extract 19-dimensional feature vector per pixel
- Compute normalized posterior P(Water | x) using GMM and class priors
- Return probability map compatible with aggregation pipeline
"""

import numpy as np

from classifier_inference import compute_single_class_probability


def process_water(img: np.ndarray) -> np.ndarray:
    """
    Generate water likelihood score map using Bayesian probabilistic model.
    
    Detection strategy:
    1. Extract comprehensive features (color, texture, gradients, indices, geometry)
    2. Evaluate P(x | Water) using trained GMM
    3. Apply Bayes' rule with class priors: P(Water | x) = P(x | Water) P(Water) / Z
    4. Return normalized probability map
    
    ! Probability is normalized across all classes (sum-to-one constraint)
    ! Water index, blue channel, and low variance are strong water indicators
    
    Args:
        img: RGB image (H, W, 3) with values in [0, 1]
        
    Returns:
        Grayscale probability map (H, W) in [0, 1] representing water likelihood
    """
    return compute_single_class_probability(img, class_id=3)
