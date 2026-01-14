# Probabilistic Segmentation System - Migration Guide

## Overview

This document describes the migration from **hand-crafted filter-based detection** to **feature-space probabilistic classification** for satellite image segmentation.

## Architecture Changes

### Before (Filter-Based)
```
Image → Class-specific filters → Likelihood maps → Aggregation → Segmentation
```

### After (Probabilistic)
```
Image → Feature extraction → GMM evaluation → Probability maps → Aggregation → Segmentation
```

## Key Components

### 1. Feature Extraction (`general_processing.py`)

**New function**: `extract_features(img: np.ndarray) -> np.ndarray`

Extracts **17 features per pixel**:

| Feature Category | Features | Count |
|-----------------|----------|-------|
| **Color** | R, G, B, H, S, V | 6 |
| **Intensity** | Grayscale, Multi-scale blur (σ=1, 2.5, 5) | 4 |
| **Gradient** | Magnitude, Orientation | 2 |
| **Texture** | Local variance, Local entropy, LBP | 3 |
| **Spectral** | NDVI approximation, Water index | 2 |

**Returns**: Feature tensor of shape `(H, W, 17)`

### 2. Training Module (`train_classifier.py`)

**Purpose**: Train one Gaussian Mixture Model (GMM) per class

**Workflow**:
1. Load training images and ground truth masks
2. Extract features for all pixels
3. Sample up to 50,000 pixels per class
4. Fit GMM with 3 components (diagonal covariance)
5. Save models to `data/models/`

**Usage**:
```python
from train_classifier import train_models_from_directory

# Train all models
train_models_from_directory(
    img_dir="data/images/train/",
    label_dir="data/labels/train/",
    model_dir="data/models/",
    n_components=3
)
```

**Output files**:
- `field_model.pkl`
- `building_model.pkl`
- `woodland_model.pkl`
- `water_model.pkl`
- `road_model.pkl`

### 3. Class Detection Modules (`classes/*.py`)

**All `process_<class>()` functions rewritten** to:

1. Load trained GMM model (cached globally)
2. Extract features: `features = extract_features(img)`
3. Compute log-likelihood: `log_prob = model.score_samples(X)`
4. Convert to probability: `prob = exp(log_prob)`
5. Normalize to [0, 1]
6. Return probability map

**API remains identical**:
```python
def process_<class>(img: np.ndarray) -> np.ndarray:
    """
    Args:
        img: RGB image (H, W, 3) in [0, 1]
    Returns:
        Probability map (H, W) in [0, 1]
    """
```

## Statistical Model Details

### Gaussian Mixture Model (GMM)

Each class `c` is modeled as:

```
P(x | class=c) = Σᵢ wᵢ · N(x | μᵢ, Σᵢ)
```

Where:
- `wᵢ`: Component weights
- `μᵢ`: Component means (17-dimensional)
- `Σᵢ`: Diagonal covariance matrices

**Why GMM?**
- Handles multi-modal distributions (e.g., dark vs bright water)
- More flexible than single Gaussian
- Diagonal covariance for computational efficiency
- Works well with moderate training data

### Inference

For each pixel with features `x`:

1. Compute **class-conditional likelihood**:
   ```
   P(x | class=c) from GMM
   ```

2. Normalize to probability map for visualization

3. Post-processing pipeline applies:
   - Gaussian smoothing
   - Argmax across classes
   - Morphological cleanup
   - Confidence thresholding

## Migration Steps

### Step 1: Install Dependencies
```bash
pip install numpy opencv-python scikit-image scipy scikit-learn
```

### Step 2: Create Model Directory
```bash
mkdir -p data/models
```

### Step 3: Train Models
```python
python train_classifier.py
```

Expected output:
- 5 model files in `data/models/`
- Training uses all images in `data/images/train/`

### Step 4: Replace Class Files
- Backup old `classes/` directory
- Replace with new probabilistic versions

### Step 5: Update `general_processing.py`
- Add feature extraction functions
- Keep all existing utilities

### Step 6: Test
```python
import cv2
from field import process_field

# Load test image
img = cv2.imread("data/images/test/M-33-7-A-d-2-3_19.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

# Get probability map
prob_map = process_field(img)
```

## Backward Compatibility

✅ **Preserved**:
- All `process_<class>()` function signatures
- Output format: `(H, W)` arrays in `[0, 1]`
- Integration with existing post-processing pipeline
- No changes to aggregation, morphology, or visualization

✅ **Improved**:
- Better class discrimination
- Learned from data instead of hand-crafted
- Consistent feature extraction across classes
- Probabilistic interpretation

## Performance Considerations

### Memory
- Feature extraction: `H × W × 17 × 4 bytes`
- For 512×512 image: ~17 MB
- Models: ~1-2 MB per class

### Speed
- Feature extraction: ~0.5-1s per image
- GMM evaluation: ~0.1-0.2s per class
- **Total**: ~1-2s per image (5 classes)

### Optimization Tips
1. **Model caching**: Models loaded once per session (global cache)
2. **Batch processing**: Extract features once, reuse for all classes
3. **Parallel evaluation**: Can evaluate all classes in parallel

## Troubleshooting

### "Model not found" Error
```python
FileNotFoundError: Model not found: data/models/field_model.pkl
```
**Solution**: Run `train_classifier.py` first

### Poor Segmentation Quality
**Possible causes**:
- Insufficient training data
- Need more GMM components (`n_components > 3`)
- Feature normalization issues

**Solutions**:
1. Add more training images
2. Increase `n_components` in training
3. Check feature value ranges

### Memory Issues
**Solution**: Reduce `max_samples_per_class` in `PixelClassifier`

## Advanced Configuration

### Tuning GMM Parameters

In `train_classifier.py`:
```python
classifier = PixelClassifier(
    n_components=5,  # Increase for more complex distributions
    max_samples_per_class=100000  # More samples for better fitting
)
```

### Adding New Features

In `general_processing.py`, extend `extract_features()`:
```python
# Add new feature
new_feature = compute_new_feature(img)
features[:, :, 17] = normalize_to_01(new_feature)

# Update feature count
return features  # Now shape (H, W, 18)
```

Don't forget to update:
- `self.n_features` in `PixelClassifier`
- Feature documentation

## Validation

### Visual Inspection
```python
import matplotlib.pyplot as plt

# Compare old vs new
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(old_prob_map, cmap='gray')
axes[0].set_title('Old (Filter-based)')
axes[1].imshow(new_prob_map, cmap='gray')
axes[1].set_title('New (Probabilistic)')
plt.show()
```

### Quantitative Metrics
Use existing evaluation pipeline with ground truth masks to compare:
- IoU per class
- Overall accuracy
- F1-score

## Future Enhancements

### Potential Improvements
1. **Feature selection**: Use PCA or feature importance analysis
2. **Class balancing**: Weight samples inversely to class frequency
3. **Spatial features**: Add neighborhood context
4. **Ensemble methods**: Combine multiple models
5. **Online learning**: Update models with new data

### Alternative Models
- **Parzen window** (kernel density estimation)
- **Naive Bayes** (faster, assumes independence)
- **Random Forest** (handles non-linear boundaries better)

## Summary

This migration replaces hand-crafted filters with learned probabilistic models while maintaining:
- ✅ Identical API
- ✅ Same output format
- ✅ Compatible with existing pipeline
- ✅ No deep learning dependencies

**Result**: More accurate, data-driven segmentation with minimal code changes.
