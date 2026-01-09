# Satellite Image Segmentation Pipeline

A heuristic-based satellite image segmentation system for generating interpretable pre-label masks for machine learning models.

## Overview

This project implements a rule-based computer vision pipeline that segments RGB satellite images into 5 semantic classes:

- **Field** (Class 0): Agricultural plots, cropland, gardens
- **Building** (Class 1): Houses, apartments, warehouses, constructions
- **Woodland** (Class 2): Dense or semi-dense tree-covered areas
- **Water** (Class 3): Rivers, lakes, ponds
- **Road** (Class 4): Paved or unpaved roads, dirt paths

### Key Features

- **Independent class processing**: Each class has a specialized detection algorithm
- **Interpretable heuristics**: Uses classical computer vision techniques (no deep learning)
- **Modular architecture**: Easy to extend with new classes
- **High recall for critical classes**: Buildings and roads prioritized over other classes
- **Efficient processing**: Optimized for speed and memory

## Project Structure

```
project/
│
├── constants.py              # Configuration and class definitions
├── io_utils.py              # Image I/O operations
├── preprocessing.py         # Shared processing functions
├── pipeline.py              # Main segmentation pipeline
├── example.py               # Usage examples
│
├── classes/                 # Per-class detection modules
│   ├── __init__.py
│   ├── field.py            # Field detection (Class 0)
│   ├── building.py         # Building detection (Class 1)
│   ├── woodland.py         # Woodland detection (Class 2)
│   ├── water.py            # Water detection (Class 3)
│   └── road.py             # Road detection (Class 4)
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── results/
│       └── predictions/
│
└── .logs/
```

## Installation

### Requirements

```bash
pip install numpy opencv-python scikit-image scipy pillow
```

### Dependencies

- Python 3.8+
- numpy
- opencv-python (cv2)
- scikit-image
- scipy
- pillow (PIL)

## Quick Start

### Basic Usage

```python
from io_utils import load_image
from pipeline import segmentation_pipeline, merge_masks_majority_vote

# Load image
img = load_image("path/to/image.jpg", normalize=True)

# Process all classes
masks = segmentation_pipeline(img, class_ids=[0, 1, 2, 3, 4])

# Get individual class masks
field_mask = masks[0]      # Values in [0, 1]
building_mask = masks[1]   # Higher = more confident
woodland_mask = masks[2]
water_mask = masks[3]
road_mask = masks[4]

# Merge into final segmentation
final_segmentation = merge_masks_majority_vote(masks)
# final_segmentation contains class IDs (0-4) for each pixel
```

### Process Specific Classes

```python
# Only process buildings and roads (high priority)
masks = segmentation_pipeline(img, class_ids=[1, 4])

building_mask = masks[1]
road_mask = masks[4]
```

### Save Results

```python
from io_utils import save_mask, save_colored_mask
from constants import ClassInfo

# Save individual mask
save_mask(building_mask, "building_confidence.png")

# Save colored segmentation
save_colored_mask(
    final_segmentation,
    "segmentation_colored.png",
    ClassInfo.CLASS_COLORS
)
```

## Architecture

### Pipeline Flow

```
Input RGB Image (H, W, 3)
         ↓
┌────────────────────┐
│  Load & Normalize  │
│   [0, 255] → [0,1] │
└────────────────────┘
         ↓
    ┌────┴────┐
    │ Process │ (for each requested class)
    └────┬────┘
         ↓
┌─────────────────────────────────────┐
│   Class-Specific Detection          │
│   ├─ Field:    Smoothness + intensity│
│   ├─ Building: Edges + contrast     │
│   ├─ Woodland: Green + texture      │
│   ├─ Water:    Blue + smoothness    │
│   └─ Road:     Edges + linearity    │
└─────────────────────────────────────┘
         ↓
┌────────────────────┐
│  Score Maps [0,1]  │
│  One per class     │
└────────────────────┘
         ↓
┌────────────────────┐
│  Majority Vote     │
│  (optional merge)  │
└────────────────────┘
         ↓
  Final Segmentation
```

### Class Detection Strategies

#### Field (Class 0)
- **Key features**: Uniform texture, medium brightness
- **Techniques**: Gaussian smoothing, CLAHE, local variance
- **Logic**: Low variance → high smoothness score

#### Building (Class 1)
- **Key features**: Strong edges, high local contrast
- **Techniques**: Sobel edges, CLAHE, morphological closing
- **Logic**: High edge density + high contrast = building
- **Priority**: High recall (expanded detection)

#### Woodland (Class 2)
- **Key features**: High green intensity, irregular texture
- **Techniques**: Green channel emphasis, LBP texture, NDVI approximation
- **Logic**: Green + texture + vegetation index

#### Water (Class 3)
- **Key features**: Blue color, very smooth, low saturation
- **Techniques**: Blue emphasis, variance analysis, HSV conversion
- **Logic**: Blue + smoothness + low saturation

#### Road (Class 4)
- **Key features**: Linear structures, strong edges, orientation consistency
- **Techniques**: Canny edges, orientation analysis, morphological operations
- **Logic**: Linear edges + consistent orientation
- **Priority**: High recall (expanded detection)

## How It Works

### Input Requirements

Images must be:
- RGB format
- Numpy array type
- Float32 dtype
- Normalized to [0, 1]

```python
img = load_image(path, normalize=True)
# img.shape = (H, W, 3)
# img.dtype = np.float32
# img.min() >= 0, img.max() <= 1
```

### Output Format

Each class processor returns a confidence score map:
- Shape: (H, W)
- Dtype: float32
- Range: [0, 1]
- Interpretation: Higher values = higher confidence

### Mask Merging

The `merge_masks_majority_vote()` function:
1. Stacks all class masks
2. Finds class with maximum score per pixel
3. Returns integer mask with class IDs

## Extending the System

### Adding a New Class

To add a new semantic class (e.g., "Bridge"):

#### Step 1: Update Constants

```python
# constants.py
class ClassInfo:
    CLASS_NAMES = {
        0: "Field",
        1: "Building",
        2: "Woodland",
        3: "Water",
        4: "Road",
        5: "Bridge",  # New class
    }
    
    CLASS_COLORS = {
        # ... existing colors ...
        5: [255, 255, 0],  # Yellow for bridges
    }
```

#### Step 2: Create Detection Module

```python
# classes/bridge.py
"""
Bridge class detection (Class 5).

Strategy:
- Bridges connect over water or valleys
- Linear structure similar to roads
- Often have distinct edge patterns
"""

import numpy as np
from preprocessing import (
    to_grayscale,
    compute_canny_edges,
    normalize_to_01
)

def process_bridge(img: np.ndarray) -> np.ndarray:
    """
    Generate bridge likelihood score map.
    
    Args:
        img: RGB image (H, W, 3) in [0, 1]
        
    Returns:
        Score map (H, W) in [0, 1]
    """
    # 1. Convert to grayscale
    gray = to_grayscale(img)
    
    # 2. Detect edges
    edges = compute_canny_edges(gray)
    
    # 3. Your custom processing logic
    # ... add your heuristics here ...
    
    # 4. Return normalized score
    return normalize_to_01(score_map)
```

#### Step 3: Register in Pipeline

```python
# pipeline.py
from classes.bridge import process_bridge

CLASS_PROCESSORS = {
    0: process_field,
    1: process_building,
    2: process_woodland,
    3: process_water,
    4: process_road,
    5: process_bridge,  # Add new class
}
```

#### Step 4: Test

```python
masks = segmentation_pipeline(img, class_ids=[5])
bridge_mask = masks[5]
```

### Design Guidelines for New Classes

1. **Analyze visual characteristics**
   - What makes this class distinct?
   - Color patterns? Texture? Shape? Edges?

2. **Choose appropriate techniques**
   - Smooth surfaces → variance analysis
   - Textured regions → LBP, local std dev
   - Geometric shapes → edge detection
   - Color patterns → channel emphasis, HSV

3. **Combine multiple features**
   - Use 3-5 features per class
   - Weight them based on importance
   - Example: `0.4 * feature1 + 0.3 * feature2 + 0.3 * feature3`

4. **Normalize and smooth**
   - Always normalize final score to [0, 1]
   - Apply light Gaussian smoothing to reduce noise

5. **Consider priority**
   - High-priority classes: apply dilation to increase recall
   - Low-priority classes: focus on precision

## Common Functions

### Preprocessing (`preprocessing.py`)

```python
# Normalization
normalize_to_01(img)                    # Min-max scaling
to_grayscale(rgb_img)                   # RGB to gray

# Smoothing
apply_gaussian(img, sigma=1.0)          # Gaussian blur
apply_unsharp_mask(img, sigma, strength) # Sharpening

# Contrast
apply_clahe(img, clip_limit, tile_size) # Adaptive histogram

# Edges
compute_sobel_magnitude(gray)           # Sobel edges
compute_canny_edges(gray, low, high)    # Canny edges

# Morphology
apply_morphology(img, op, kernel, iter) # Dilate/erode/close/open

# Texture
compute_local_variance(gray, window)    # Local variance
compute_local_std(gray, window)         # Local std dev
compute_edge_density(edges, kernel)     # Edge concentration

# Color
rgb_to_hsv(rgb_img)                     # RGB to HSV
emphasize_channel(img, idx, factor)     # Boost R/G/B
```

## Performance Considerations

### Speed Optimization

- Use appropriate kernel/window sizes (smaller = faster)
- Process only required classes
- Consider downsampling for large images

### Memory Management

- Process images in batches
- Delete intermediate arrays when possible
- Use in-place operations where applicable

### Typical Processing Times

On a modern CPU (example: Intel i7):
- Single class: ~0.1-0.3 seconds per image
- All classes: ~0.5-1.5 seconds per image
- Image size: 512×512 pixels

## Troubleshooting

### Common Issues

**Issue**: Low confidence scores across all pixels
- **Solution**: Check image normalization (should be [0, 1])

**Issue**: One class dominates segmentation
- **Solution**: Adjust feature weights in class processor

**Issue**: Noisy outputs
- **Solution**: Increase smoothing sigma values

**Issue**: Missing small objects
- **Solution**: Reduce kernel/window sizes

**Issue**: False positives for high-priority classes
- **Solution**: Reduce dilation iterations or weights

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Version**: 1.0  
**Last Updated**: January 2026
