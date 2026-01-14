# Segmentation Strategy Document

This document provides a detailed explanation of the processing pipeline for each semantic class in the satellite image segmentation system.

## Table of Contents

- [Overview](#overview)
- [Class 0: Field](#class-0-field)
- [Class 1: Building](#class-1-building)
- [Class 2: Woodland](#class-2-woodland)
- [Class 3: Water](#class-3-water)
- [Class 4: Road](#class-4-road)
- [Parameter Reference](#parameter-reference)
- [Feature Combination Rationale](#feature-combination-rationale)

---

## Overview

Each class is detected using a specialized pipeline that combines multiple computer vision techniques. The general approach follows these principles:

1. **Visual Analysis**: Identify distinctive visual characteristics
2. **Feature Extraction**: Apply appropriate filters and transformations
3. **Score Computation**: Combine features into a confidence score
4. **Post-Processing**: Normalize and smooth the final score map

### Common Input/Output Format

**Input**: RGB satellite image
- Type: `np.ndarray`
- Shape: `(H, W, 3)`
- Dtype: `float32`
- Range: `[0, 1]`

**Output**: Confidence score map
- Type: `np.ndarray`
- Shape: `(H, W)`
- Dtype: `float32`
- Range: `[0, 1]`
- Interpretation: Higher values indicate higher class likelihood

---

## Class 0: Field

**Module**: `classes/field.py`  
**Function**: `process_field(img: np.ndarray) -> np.ndarray`

### Visual Characteristics

Fields exhibit:
- **Uniform texture** with low spatial variance
- **Medium brightness** (typically 0.4-0.6 intensity)
- **Regular patterns** from agricultural cultivation
- **Low edge density** compared to buildings or roads
- **Consistent color** within regions

### Processing Pipeline

#### Step 1: Light Smoothing
```python
smoothed = apply_gaussian(img, sigma=1.0)
```

**Purpose**: Reduce high-frequency noise while preserving field boundaries

**Technical Details**:
- Uses Gaussian filter with σ=1.0
- Applied to all 3 RGB channels independently
- Preserves range [0, 1]

**Why**: Fields contain subtle noise from varying vegetation density. Light smoothing helps identify the underlying uniform structure without over-blurring boundaries.

---

#### Step 2: Local Contrast Normalization
```python
normalized = apply_clahe(smoothed, clip_limit=2.0, tile_size=8)
```

**Purpose**: Enhance local contrast to distinguish field variations

**Technical Details**:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Clip limit: 2.0 (moderate enhancement)
- Tile size: 8×8 pixels
- Applied per-channel

**Why**: Agricultural fields may have varying illumination. CLAHE normalizes local contrast, making fields more uniformly detectable regardless of lighting conditions.

---

#### Step 3: Grayscale Conversion
```python
gray = to_grayscale(normalized)
# Weights: R=0.299, G=0.587, B=0.114
```

**Purpose**: Reduce dimensionality for texture analysis

**Technical Details**:
- Luminosity method (standard ITU-R BT.601)
- Emphasizes green channel (57% weight)

**Why**: Fields are primarily characterized by texture uniformity rather than color. Grayscale simplifies variance computation.

---

#### Step 4: Smoothness Score
```python
local_var = compute_local_variance(gray, window_size=7)
smoothness = 1.0 / (1.0 + local_var * 100)
```

**Purpose**: Quantify texture uniformity

**Technical Details**:
- Local variance computed over 7×7 windows
- Inverse relationship: high variance → low smoothness
- Scaling factor: 100 (adjusts sensitivity)

**Mathematical Formula**:
```
For each pixel (x, y):
  local_mean = avg(gray[window])
  local_var = avg((gray[window] - local_mean)²)
  smoothness = 1 / (1 + 100 × local_var)
```

**Why**: Fields have low spatial variance due to uniform crop coverage. Buildings and roads have high variance due to edges and textures.

---

#### Step 5: Intensity Preference
```python
intensity_center = 0.5
intensity_width = 0.3
intensity_score = exp(-((gray - 0.5)² / (2 × 0.3²)))
```

**Purpose**: Favor medium-brightness regions typical of fields

**Technical Details**:
- Gaussian preference centered at 0.5 intensity
- Width parameter: 0.3 (covers 0.2-0.8 range effectively)

**Mathematical Formula**:
```
intensity_score = exp(-(I - 0.5)² / 0.18)
```

**Why**: Fields typically appear medium-bright (neither very dark like shadows nor very bright like concrete). This Gaussian preference penalizes extreme intensities.

---

#### Step 6: Feature Combination
```python
combined = 0.7 × smoothness + 0.3 × intensity_score
```

**Purpose**: Merge smoothness and intensity cues

**Weight Rationale**:
- **70% smoothness**: Primary indicator (fields are smooth)
- **30% intensity**: Secondary indicator (fields are medium-bright)

**Why**: Smoothness is the strongest field indicator, but intensity helps distinguish fields from other smooth areas like water or roads.

---

#### Step 7: Score Smoothing
```python
score_smoothed = apply_gaussian(combined, sigma=1.5)
```

**Purpose**: Reduce pixel-level noise in score map

**Technical Details**:
- Gaussian smoothing with σ=1.5
- Slightly stronger than input smoothing

**Why**: Score maps can have noisy transitions. Smoothing creates more coherent field regions.

---

#### Step 8: Normalization
```python
final_score = normalize_to_01(score_smoothed)
```

**Purpose**: Ensure output is in [0, 1] range

**Technical Details**:
- Min-max scaling: `(x - min) / (max - min)`
- Handles edge case where min ≈ max

**Why**: Standardizes output range for comparison across classes and images.

---

### Summary Flow

```
RGB Image
    ↓ Gaussian smoothing (σ=1.0)
Smoothed Image
    ↓ CLAHE (clip=2.0, tile=8×8)
Contrast Normalized
    ↓ Grayscale conversion
Gray Image
    ↓ Local variance (window=7×7)
Smoothness Score ──┐
                   ├─→ Weighted combination (0.7 + 0.3)
Intensity Score ───┘
    ↓
Combined Score
    ↓ Gaussian smoothing (σ=1.5)
Smoothed Score
    ↓ Min-max normalization
Final Field Score [0, 1]
```

---

## Class 1: Building

**Module**: `classes/building.py`  
**Function**: `process_building(img: np.ndarray) -> np.ndarray`

### Visual Characteristics

Buildings exhibit:
- **Strong edges** at structure boundaries
- **High local contrast** between building and surroundings
- **Geometric shapes** (rectangles, polygons)
- **Clustered edge patterns** forming continuous structures
- **Distinct brightness** from background

### Priority Note

**High Priority Class**: Buildings are critical to detect. The pipeline favors **high recall** over precision to minimize false negatives.

### Processing Pipeline

#### Step 1: Grayscale Conversion
```python
gray = to_grayscale(img)
```

**Purpose**: Simplify for edge detection

**Why**: Building edges are luminance-based rather than color-based. Grayscale captures structural information efficiently.

---

#### Step 2: Contrast Enhancement
```python
contrast_enhanced = apply_clahe(gray, clip_limit=3.0, tile_size=8)
```

**Purpose**: Amplify local contrast for better edge detection

**Technical Details**:
- CLAHE with clip_limit=3.0 (stronger than field processing)
- 8×8 tile size for local adaptation

**Why**: Buildings often blend with surroundings. Strong contrast enhancement makes edges more prominent.

---

#### Step 3: Edge Detection (Sobel)
```python
edges = compute_sobel_magnitude(contrast_enhanced)
edges_norm = normalize_to_01(edges)
```

**Purpose**: Detect intensity gradients

**Technical Details**:
- Sobel operator computes gradients in x and y directions
- Magnitude: `sqrt(Gx² + Gy²)`
- 3×3 kernel size

**Mathematical Formula**:
```
Gx = Sobel_x ∗ image
Gy = Sobel_y ∗ image
magnitude = √(Gx² + Gy²)
```

**Why**: Buildings have strong edges at walls, roofs, and windows. Sobel captures these gradients effectively.

---

#### Step 4: Morphological Closing
```python
edges_closed = apply_morphology(
    edges_norm,
    operation='close',
    kernel_size=3,
    iterations=1
)
```

**Purpose**: Connect nearby edges into continuous structures

**Technical Details**:
- Closing = Dilation followed by Erosion
- 3×3 structuring element
- 1 iteration

**Why**: Building edges may have small gaps due to texture or lighting. Closing fills these gaps to form complete building outlines.

---

#### Step 5: Edge Density
```python
edge_density = compute_edge_density(edges_closed, kernel_size=9)
```

**Purpose**: Measure local edge concentration

**Technical Details**:
- Convolves edge map with uniform 9×9 kernel
- Produces local average of edge pixels

**Mathematical Formula**:
```
For each pixel (x, y):
  edge_density(x, y) = sum(edges[window]) / 81
```

**Why**: Buildings have high edge density (many edges in small area). Fields and water have low edge density.

---

#### Step 6: Local Contrast
```python
local_contrast = compute_local_std(contrast_enhanced, window_size=7)
local_contrast_norm = normalize_to_01(local_contrast)
```

**Purpose**: Measure intensity variation within neighborhoods

**Technical Details**:
- Standard deviation over 7×7 windows
- Higher std = higher contrast

**Why**: Buildings have high local contrast due to shadows, textures, and material differences.

---

#### Step 7: Gradient Magnitude
```python
gradient_mag = sqrt(sobel_x² + sobel_y²)
gradient_norm = normalize_to_01(gradient_mag)
```

**Purpose**: Additional edge information

**Technical Details**:
- Similar to Step 3 but on original gray image
- Captures finer edge details

**Why**: Provides complementary edge information that may be lost in closed edges.

---

#### Step 8: Feature Combination
```python
combined = (
    0.45 × edge_density +
    0.30 × local_contrast_norm +
    0.25 × gradient_norm
)
```

**Purpose**: Integrate multiple building indicators

**Weight Rationale**:
- **45% edge density**: Primary indicator (buildings = many edges)
- **30% local contrast**: Secondary indicator (buildings have contrast)
- **25% gradient**: Tertiary indicator (refines detection)

**Why**: Multiple features provide robustness. If one feature fails (e.g., low contrast building), others compensate.

---

#### Step 9: Dilation (High Recall)
```python
combined_expanded = apply_morphology(
    combined,
    operation='dilate',
    kernel_size=3,
    iterations=1
)
```

**Purpose**: Expand detected regions to increase recall

**Technical Details**:
- Dilation with 3×3 kernel
- 1 iteration (modest expansion)

**Why**: **Critical for high-priority class**. Slight over-detection is acceptable to ensure no buildings are missed.

---

#### Step 10: Normalization
```python
final_score = normalize_to_01(combined_expanded)
```

**Purpose**: Standardize output range

---

### Summary Flow

```
RGB Image
    ↓ Grayscale conversion
Gray Image
    ↓ CLAHE (clip=3.0, tile=8×8)
Contrast Enhanced
    ↓
    ├─→ Sobel edges → Close → Edge density ────────────┐
    │                                                   │
    ├─→ Local standard deviation ──────────────────────┤
    │                                                   ├─→ Weighted
    └─→ Gradient magnitude ────────────────────────────┘   combination
                                                            (0.45+0.30+0.25)
Combined Score
    ↓ Dilation (3×3, 1 iter) [HIGH RECALL BIAS]
Expanded Score
    ↓ Min-max normalization
Final Building Score [0, 1]
```

---

## Class 2: Woodland

**Module**: `classes/woodland.py`  
**Function**: `process_woodland(img: np.ndarray) -> np.ndarray`

### Visual Characteristics

Woodlands exhibit:
- **High green channel intensity** (vegetation signature)
- **Irregular texture** from trees and foliage
- **High spatial variance** compared to fields
- **Strong vegetation index** (NDVI-like)
- **Natural patterns** (non-geometric)

### Processing Pipeline

#### Step 1: Green Channel Emphasis
```python
green_emphasized = emphasize_channel(img, channel_idx=1, factor=1.4)
```

**Purpose**: Amplify vegetation signal

**Technical Details**:
- Multiplies green channel by 1.4
- Clips result to [0, 1]
- Red and blue channels unchanged

**Mathematical Formula**:
```
R' = R
G' = min(1.4 × G, 1.0)
B' = B
```

**Why**: Vegetation reflects strongly in green spectrum. Amplifying green makes woodlands more distinguishable from bare fields or urban areas.

---

#### Step 2: Sharpening (Unsharp Mask)
```python
sharpened = apply_unsharp_mask(
    green_emphasized,
    sigma=1.5,
    strength=0.6
)
```

**Purpose**: Enhance texture details

**Technical Details**:
- Gaussian blur with σ=1.5
- Sharpening: `original + 0.6 × (original - blurred)`
- Applied to all channels

**Mathematical Formula**:
```
blurred = Gaussian(img, σ=1.5)
sharpened = img + 0.6 × (img - blurred)
```

**Why**: Trees have fine texture (leaves, branches). Sharpening enhances these details, making texture analysis more effective.

---

#### Step 3: Green Channel Extraction
```python
green_channel = sharpened[:, :, 1]
```

**Purpose**: Isolate primary vegetation indicator

**Why**: After emphasis and sharpening, green channel contains strongest woodland signal.

---

#### Step 4: Texture Variance
```python
texture_var = compute_local_variance(green_channel, window_size=7)
texture_score = normalize_to_01(texture_var)
```

**Purpose**: Quantify texture irregularity

**Technical Details**:
- Local variance over 7×7 windows
- High variance indicates texture

**Why**: Woodlands have irregular texture from tree crowns. Fields have low variance (uniform crops).

---

#### Step 5: Local Binary Pattern (LBP)
```python
gray = to_grayscale(sharpened)
radius = 2
n_points = 8 × radius = 16
lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
lbp_norm = normalize_to_01(lbp)
```

**Purpose**: Capture fine-scale texture patterns

**Technical Details**:
- LBP with radius=2, 16 sampling points
- 'uniform' method: rotation-invariant patterns
- Produces texture descriptor per pixel

**How LBP Works**:
1. For each pixel, sample 16 neighbors in radius=2 circle
2. Compare each neighbor to center pixel (binary)
3. Form 16-bit binary pattern
4. Map to uniform pattern code

**Why**: LBP captures micro-texture (e.g., leaf patterns) that variance may miss. Complements variance analysis.

---

#### Step 6: NDVI Approximation
```python
red = img[:, :, 0]
green = img[:, :, 1]
ndvi_approx = (green - red) / (green + red + 1e-8)
ndvi_norm = normalize_to_01(ndvi_approx)
```

**Purpose**: Compute vegetation index

**Technical Details**:
- Approximates Normalized Difference Vegetation Index
- True NDVI uses near-infrared, but we only have RGB
- Uses green as proxy for infrared

**Mathematical Formula**:
```
NDVI ≈ (G - R) / (G + R)
Range: [-1, 1] → normalized to [0, 1]
```

**Why**: Vegetation has high (Green - Red) values. NDVI is standard remote sensing metric for vegetation detection.

---

#### Step 7: Feature Combination
```python
combined = (
    0.30 × green_channel +
    0.25 × texture_score +
    0.20 × lbp_norm +
    0.25 × ndvi_norm
)
```

**Purpose**: Integrate multiple woodland indicators

**Weight Rationale**:
- **30% green intensity**: Primary color indicator
- **25% texture variance**: Structural indicator
- **20% LBP**: Fine texture indicator
- **25% NDVI**: Vegetation health indicator

**Why**: Balanced combination ensures detection works even if one feature fails (e.g., dark shadows reducing green intensity).

---

#### Step 8: Score Smoothing
```python
score_smoothed = apply_gaussian(combined, sigma=1.5)
```

**Purpose**: Create coherent woodland regions

**Why**: Individual texture measures can be noisy. Smoothing consolidates woodland areas.

---

#### Step 9: Normalization
```python
final_score = normalize_to_01(score_smoothed)
```

---

### Summary Flow

```
RGB Image
    ↓ Emphasize green (×1.4)
Green Emphasized
    ↓ Unsharp mask (σ=1.5, strength=0.6)
Sharpened Image
    ↓
    ├─→ Extract green channel ─────────────────────────┐
    │                                                   │
    ├─→ Texture variance (7×7) ────────────────────────┤
    │                                                   │
    ├─→ LBP (radius=2, n=16) ──────────────────────────┤
    │                                                   ├─→ Weighted
Original RGB                                           │   combination
    └─→ NDVI approximation ────────────────────────────┘   (0.30+0.25+0.20+0.25)

Combined Score
    ↓ Gaussian smoothing (σ=1.5)
Smoothed Score
    ↓ Min-max normalization
Final Woodland Score [0, 1]
```

---

## Class 3: Water

**Module**: `classes/water.py`  
**Function**: `process_water(img: np.ndarray) -> np.ndarray`

### Visual Characteristics

Water bodies exhibit:
- **High blue channel intensity** (water reflection spectrum)
- **Very smooth texture** (minimal spatial variance)
- **Low saturation** (desaturated appearance)
- **Variable brightness** (dark in shadows, bright in sunlight)
- **Uniform color** within regions

### Processing Pipeline

#### Step 1: Heavy Smoothing
```python
smoothed = apply_gaussian(img, sigma=2.0)
```

**Purpose**: Reduce noise and emphasize smoothness

**Technical Details**:
- Strong Gaussian blur (σ=2.0, strongest in all classes)
- Applied to all channels

**Why**: Water is naturally smooth. Heavy smoothing removes ripples and reflections while preserving the underlying smooth structure.

---

#### Step 2: Blue Channel Emphasis
```python
blue_emphasized = emphasize_channel(smoothed, channel_idx=2, factor=1.5)
```

**Purpose**: Amplify water's spectral signature

**Technical Details**:
- Multiplies blue channel by 1.5
- Clips to [0, 1]

**Mathematical Formula**:
```
R' = R
G' = G
B' = min(1.5 × B, 1.0)
```

**Why**: Water reflects blue wavelengths strongly. Emphasis increases separation from non-water regions.

---

#### Step 3: Blue Channel Extraction
```python
blue_channel = blue_emphasized[:, :, 2]
```

**Purpose**: Isolate primary water indicator

---

#### Step 4: Smoothness Score
```python
gray = to_grayscale(blue_emphasized)
local_var = compute_local_variance(gray, window_size=9)
smoothness = 1.0 / (1.0 + local_var × 150)
```

**Purpose**: Quantify surface uniformity

**Technical Details**:
- Local variance over 9×9 windows (larger than field: 7×7)
- Scaling factor: 150 (stronger than field: 100)
- Inverse relationship

**Mathematical Formula**:
```
smoothness = 1 / (1 + 150 × variance)
```

**Why**: Water is smoother than fields. Larger window (9×9) and higher scaling (150) make detector more sensitive to smoothness differences.

---

#### Step 5: HSV Conversion & Saturation
```python
hsv = rgb_to_hsv(blue_emphasized)
saturation = hsv[:, :, 1]
low_saturation_score = 1.0 - saturation
```

**Purpose**: Detect desaturated regions

**Technical Details**:
- Converts RGB → HSV color space
- Extracts saturation channel
- Inverts (low saturation = high score)

**Why**: Water often appears gray/desaturated, especially in shadows or with sediment. High saturation indicates colorful objects (vegetation, buildings).

---

#### Step 6: Intensity Preference (Bimodal)
```python
value = hsv[:, :, 2]
dark_preference = exp(-((value - 0.3)² / (2 × 0.2²)))
bright_preference = exp(-((value - 0.6)² / (2 × 0.25²)))
intensity_preference = max(dark_preference, bright_preference)
```

**Purpose**: Handle varying lighting conditions

**Technical Details**:
- Two Gaussian preferences:
  - Dark water: centered at 0.3 (shadows)
  - Bright water: centered at 0.6 (sunlit)
- Takes maximum of both

**Mathematical Formula**:
```
dark = exp(-(V - 0.3)² / 0.08)
bright = exp(-(V - 0.6)² / 0.125)
preference = max(dark, bright)
```

**Why**: Water can be very dark (shadows, deep water) or medium-bright (sunlit surface). Bimodal preference captures both cases.

---

#### Step 7: Feature Combination
```python
combined = (
    0.40 × blue_channel +
    0.30 × smoothness +
    0.20 × low_saturation_score +
    0.10 × intensity_preference
)
```

**Purpose**: Integrate water indicators

**Weight Rationale**:
- **40% blue intensity**: Primary spectral indicator
- **30% smoothness**: Primary structural indicator
- **20% low saturation**: Secondary color indicator
- **10% intensity**: Lighting adaptation

**Why**: Blue and smoothness are most reliable. Saturation helps reject blue buildings. Intensity handles edge cases.

---

#### Step 8: Score Smoothing
```python
score_smoothed = apply_gaussian(combined, sigma=2.0)
```

**Purpose**: Create coherent water bodies

**Technical Details**:
- Strong smoothing (σ=2.0, matching input smoothing)

**Why**: Water bodies are naturally continuous. Strong smoothing consolidates fragmented detections.

---

#### Step 9: Normalization
```python
final_score = normalize_to_01(score_smoothed)
```

---

### Summary Flow

```
RGB Image
    ↓ Heavy Gaussian smoothing (σ=2.0)
Smoothed Image
    ↓ Emphasize blue (×1.5)
Blue Emphasized
    ↓
    ├─→ Extract blue channel ──────────────────────────┐
    │                                                   │
    ├─→ Grayscale → Local variance (9×9) → Smoothness ─┤
    │                                                   │
    ├─→ RGB to HSV → Saturation → Invert ──────────────┤
    │                                                   ├─→ Weighted
    └─→ HSV value → Bimodal intensity preference ──────┘   combination
                                                            (0.40+0.30+0.20+0.10)
Combined Score
    ↓ Heavy Gaussian smoothing (σ=2.0)
Smoothed Score
    ↓ Min-max normalization
Final Water Score [0, 1]
```

---

## Class 4: Road

**Module**: `classes/road.py`  
**Function**: `process_road(img: np.ndarray) -> np.ndarray`

### Visual Characteristics

Roads exhibit:
- **Strong linear edges** at road boundaries
- **Consistent orientation** along road length
- **Uniform texture within road surface**
- **High contrast** with surroundings
- **Linear/curved geometric shapes**

### Priority Note

**High Priority Class**: Roads are critical infrastructure. Pipeline favors **high recall** to minimize missed roads.

### Processing Pipeline

#### Step 1: Grayscale Conversion
```python
gray = to_grayscale(img)
```

**Purpose**: Simplify for edge and orientation analysis

---

#### Step 2: Strong Contrast Enhancement
```python
contrast_enhanced = apply_clahe(gray, clip_limit=3.5, tile_size=8)
```

**Purpose**: Maximize edge visibility

**Technical Details**:
- CLAHE with clip_limit=3.5 (strongest in all classes)
- 8×8 tile size

**Why**: Roads may have low contrast with surroundings (e.g., dirt roads on soil). Strong enhancement makes boundaries detectable.

---

#### Step 3: Canny Edge Detection
```python
canny_edges = compute_canny_edges(
    contrast_enhanced,
    low_thresh=30,
    high_thresh=100
)
```

**Purpose**: Detect linear structures

**Technical Details**:
- Canny edge detector with dual thresholds
- Lower thresholds (30/100) for sensitivity
- Produces binary edge map

**How Canny Works**:
1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

**Why**: Canny excels at detecting continuous linear edges (road boundaries) while suppressing noise.

---

#### Step 4: Edge Dilation
```python
edges_dilated = apply_morphology(
    canny_edges,
    operation='dilate',
    kernel_size=3,
    iterations=2
)
```

**Purpose**: Strengthen and connect edge detections

**Technical Details**:
- Dilation with 3×3 kernel
- 2 iterations (stronger than building: 1)

**Why**: Road edges may be fragmented. Dilation connects nearby edges into continuous road boundaries.

---

#### Step 5: Morphological Closing
```python
edges_closed = apply_morphology(
    edges_dilated,
    operation='close',
    kernel_size=5,
    iterations=1
)
```

**Purpose**: Fill gaps in linear structures

**Technical Details**:
- Closing with 5×5 kernel (larger than building: 3×3)
- Closing = Dilation + Erosion

**Why**: Roads have consistent width. Closing fills small gaps while maintaining road structure.

---

#### Step 6: Edge Density
```python
edge_density = compute_edge_density(edges_closed, kernel_size=11)
```

**Purpose**: Measure local edge concentration

**Technical Details**:
- Convolves with 11×11 uniform kernel (largest in all classes)
- Produces local edge count

**Why**: Roads have moderate edge density (two parallel edges). Large kernel (11×11) captures both edges of wide roads.

---

#### Step 7: Sobel Magnitude
```python
sobel_mag = compute_sobel_magnitude(contrast_enhanced)
sobel_norm = normalize_to_01(sobel_mag)
```

**Purpose**: Complementary edge information

**Why**: Sobel captures gradients that Canny may miss (e.g., weak but consistent edges).

---

#### Step 8: Orientation Analysis
```python
sobel_x = ndimage.sobel(gray, axis=1)
sobel_y = ndimage.sobel(gray, axis=0)
orientation_strength = sqrt(sobel_x² + sobel_y²)
orientation_norm = normalize_to_01(orientation_strength)
```

**Purpose**: Detect oriented structures

**Technical Details**:
- Separate x and y gradients
- Magnitude indicates edge strength
- Direction (not used explicitly) indicates orientation

**Why**: Roads maintain consistent orientation over local regions, unlike random edges.

---

#### Step 9: Orientation Consistency
```python
window = 7
sx_smooth = uniform_filter(sobel_x, size=window)
sy_smooth = uniform_filter(sobel_y, size=window)
consistency = sqrt(sx_smooth² + sy_smooth²)
consistency_norm = normalize_to_01(consistency)
```

**Purpose**: Measure directional coherence

**Technical Details**:
- Smooths gradient components over 7×7 windows
- If gradients align locally, smoothed magnitude is high
- If gradients cancel (random), smoothed magnitude is low

**Mathematical Formula**:
```
Gx_smooth = avg(Gx[window])
Gy_smooth = avg(Gy[window])
consistency = √(Gx_smooth² + Gy_smooth²)
```

**Why**: Roads have consistent gradient direction (perpendicular to road). Random textures have inconsistent gradients that cancel when averaged.

---

#### Step 10: Feature Combination
```python
combined = (
    0.35 × edge_density +
    0.25 × sobel_norm +
    0.20 × orientation_norm +
    0.20 × consistency_norm
)
```

**Purpose**: Integrate multiple road indicators

**Weight Rationale**:
- **35% edge density**: Primary indicator (roads have edges)
- **25% Sobel magnitude**: Secondary edge indicator
- **20% orientation strength**: Linear structure indicator
- **20% orientation consistency**: Coherence indicator

**Why**: Balanced weights ensure detection of various road types (paved, dirt, narrow, wide).

---

#### Step 11: Dilation (High Recall)
```python
combined_expanded = apply_morphology(
    combined,
    operation='dilate',
    kernel_size=3,
    iterations=1
)
```

**Purpose**: Expand detected regions for high recall

**Technical Details**:
- Dilation with 3×3 kernel
- 1 iteration

**Why**: **Critical for high-priority class**. Ensures narrow roads are not missed.

---

#### Step 12: Score Smoothing
```python
score_smoothed = apply_gaussian(combined_expanded, sigma=1.5)
```

**Purpose**: Create continuous road regions

---

#### Step 13: Normalization
```python
final_score = normalize_to_01(score_smoothed)
```

---

### Summary Flow

```
RGB Image
    ↓ Grayscale conversion
Gray Image
    ↓ CLAHE (clip=3.5, tile=8×8) [STRONGEST ENHANCEMENT]
Contrast Enhanced
    ↓
├─→ Canny edges → Dilate (2 iter) → Close (5×<function_calls>
├─→ Canny edges → Dilate (2 iter) → Close (5×5) → Edge density ───┐
│                                                                 │
├─→ Sobel magnitude ──────────────────────────────────────────────┤
│                                                                 │
├─→ Sobel x,y → Orientation strength ─────────────────────────────┤
│                                                                 ├─→ Weighted
└─→ Sobel x,y → Smooth → Orientation consistency ─────────────────┘   combination
(0.35+0.25+0.20+0.20)
Combined Score
↓ Dilation (3×3, 1 iter) [HIGH RECALL BIAS]
Expanded Score
↓ Gaussian smoothing (σ=1.5)
Smoothed Score
↓ Min-max normalization
Final Road Score [0, 1]
```

---

## Parameter Reference

### Smoothing Parameters

| Class     | Operation         | Sigma/Size | Purpose                  |
|-----------|-------------------|------------|--------------------------|
| Field     | Gaussian          | σ = 1.0    | Light noise reduction    |
| Building  | -                 | -          | No input smoothing       |
| Woodland  | Unsharp mask      | σ = 1.5    | Texture enhancement      |
| Water     | Gaussian          | σ = 2.0    | Heavy smoothing          |
| Road      | -                 | -          | No input smoothing       |

### Contrast Enhancement (CLAHE)

| Class     | Clip Limit | Tile Size | Purpose                     |
|-----------|------------|-----------|-----------------------------|
| Field     | 2.0        | 8×8       | Moderate normalization      |
| Building  | 3.0        | 8×8       | Strong edge enhancement     |
| Woodland  | -          | -         | Not used                    |
| Water     | -          | -         | Not used                    |
| Road      | 3.5        | 8×8       | Strongest enhancement       |

### Window Sizes

| Feature Type      | Field | Building | Woodland | Water | Road |
|-------------------|-------|----------|----------|-------|------|
| Local Variance    | 7×7   | -        | 7×7      | 9×9   | -    |
| Local Std Dev     | -     | 7×7      | -        | -     | -    |
| Edge Density      | -     | 9×9      | -        | -     | 11×11|
| Orientation       | -     | -        | -        | -     | 7×7  |

### Morphological Operations

| Class     | Operation | Kernel | Iterations | Purpose                  |
|-----------|-----------|--------|------------|--------------------------|
| Building  | Close     | 3×3    | 1          | Connect edges            |
| Building  | Dilate    | 3×3    | 1          | Expand detection (recall)|
| Road      | Dilate    | 3×3    | 2          | Strengthen edges         |
| Road      | Close     | 5×5    | 1          | Fill gaps                |
| Road      | Dilate    | 3×3    | 1          | Expand detection (recall)|

### Edge Detection

| Class     | Method | Parameters          | Purpose                    |
|-----------|--------|---------------------|----------------------------|
| Building  | Sobel  | 3×3 kernel          | General edges              |
| Road      | Canny  | thresh=(30, 100)    | Linear structures          |
| Road      | Sobel  | 3×3 kernel          | Complementary edges        |

### Channel Emphasis

| Class     | Channel | Factor | Rationale                       |
|-----------|---------|--------|---------------------------------|
| Woodland  | Green   | 1.4×   | Vegetation reflectance          |
| Water     | Blue    | 1.5×   | Water spectral signature        |

### Texture Analysis

| Class     | Method              | Parameters       | Purpose                  |
|-----------|---------------------|------------------|--------------------------|
| Woodland  | Local Variance      | window=7×7       | Measure irregularity     |
| Woodland  | LBP                 | radius=2, n=16   | Fine-scale patterns      |
| Field     | Local Variance      | window=7×7       | Measure smoothness       |
| Water     | Local Variance      | window=9×9       | Measure uniformity       |

---

## Feature Combination Rationale

### Weight Distribution Philosophy

Each class combines 3-4 features with weights that reflect:

1. **Reliability**: How consistently the feature detects the class
2. **Discriminability**: How well it separates from other classes
3. **Robustness**: How well it handles edge cases (lighting, shadows, etc.)

### Per-Class Weight Justification

#### Field (0.7 smoothness + 0.3 intensity)

**Why 70% smoothness?**
- Most reliable field indicator
- Distinguishes from roads (also smooth but less smooth)
- Distinguishes from woodlands (textured)

**Why 30% intensity?**
- Secondary discriminator
- Helps reject water (too dark/bright)
- Helps reject roads (often brighter or darker)

#### Building (0.45 edge + 0.30 contrast + 0.25 gradient)

**Why 45% edge density?**
- Primary indicator: buildings = many edges
- Most discriminative feature
- Directly measures building boundaries

**Why 30% local contrast?**
- Buildings have high internal contrast (windows, shadows)
- Complements edge detection
- Robust to building type variations

**Why 25% gradient magnitude?**
- Refines detection
- Captures weak edges missed by density
- Lower weight due to noise sensitivity

#### Woodland (0.30 green + 0.25 variance + 0.20 LBP + 0.25 NDVI)

**Why balanced weights?**
- No single dominant feature
- Robustness through diversity

**Why 30% green intensity?**
- Primary color indicator
- But not dominant (shadows reduce green)

**Why 25% texture variance + 20% LBP?**
- Combined 45% for texture
- Texture is highly reliable for woodlands
- Split between coarse (variance) and fine (LBP)

**Why 25% NDVI?**
- Strong vegetation indicator
- Independent of illumination
- Complements color features

#### Water (0.40 blue + 0.30 smoothness + 0.20 saturation + 0.10 intensity)

**Why 40% blue intensity?**
- Strongest single indicator
- Spectral signature of water
- Dominant weight justified by reliability

**Why 30% smoothness?**
- Second most reliable
- Water is very smooth
- Helps reject blue buildings

**Why 20% low saturation?**
- Discriminates from blue sky reflections
- Robust feature
- Lower weight (less reliable in sunlit water)

**Why 10% intensity preference?**
- Edge case handler
- Adapts to lighting variations
- Low weight (auxiliary role)

#### Road (0.35 edge + 0.25 Sobel + 0.20 orientation + 0.20 consistency)

**Why 35% edge density?**
- Primary indicator
- Roads have parallel edges
- Most reliable feature

**Why 25% Sobel magnitude?**
- Complementary edge info
- Captures weak edges
- Moderate weight for robustness

**Why 20% orientation strength + 20% consistency?**
- Combined 40% for linear structure
- Discriminates from buildings (random edges)
- Split between strength and coherence

---

## Cross-Class Discrimination

### How Classes Are Distinguished

| Feature           | Field | Building | Woodland | Water | Road |
|-------------------|-------|----------|----------|-------|------|
| Smoothness        | High  | Low      | Low      | Very High | Medium |
| Edge Density      | Low   | Very High| Medium   | Very Low | High |
| Green Intensity   | Medium| Low      | High     | Low   | Low  |
| Blue Intensity    | Low   | Low      | Low      | High  | Low  |
| Texture Variance  | Low   | High     | High     | Very Low | Medium |
| Linear Orientation| No    | No       | No       | No    | Yes  |

### Example Discriminations

**Field vs Water**:
- Both smooth
- Water: high blue, low saturation
- Field: medium brightness, medium green

**Building vs Road**:
- Both have edges
- Road: linear, consistent orientation
- Building: clustered, random orientation

**Woodland vs Field**:
- Both have vegetation
- Woodland: high texture, high variance
- Field: low texture, uniform

**Building vs Woodland**:
- Both have texture
- Building: geometric edges, high local contrast
- Woodland: irregular texture, high green

---

## Computational Complexity

### Per-Class Operation Counts

Approximate operation counts for 512×512 image:

| Class     | Major Operations | Approx. Time | Relative Speed |
|-----------|------------------|--------------|----------------|
| Field     | 8 steps          | ~50ms        | Fast           |
| Building  | 10 steps         | ~80ms        | Medium         |
| Woodland  | 9 steps          | ~120ms       | Slow (LBP)     |
| Water     | 9 steps          | ~70ms        | Medium         |
| Road      | 13 steps         | ~110ms       | Slow           |

**Bottlenecks**:
- **LBP** (Woodland): Most expensive operation
- **Morphological operations** (Building, Road): Multiple iterations
- **CLAHE** (Building, Road): Tile-based processing

**Optimization opportunities**:
- Downsample before LBP
- Reduce morphological iterations
- Use smaller CLAHE tile sizes

---

## Design Decisions

### Why Heuristic Approach?

1. **Interpretability**: Every decision is explainable
2. **Data efficiency**: No training data required
3. **Debugging**: Easy to trace failures
4. **Flexibility**: Quick to modify and tune
5. **Baseline**: Provides pre-labels for ML training

### Why Separate Processors?

1. **Modularity**: Easy to modify one class without affecting others
2. **Clarity**: Clear logic per class
3. **Extensibility**: Simple to add new classes
4. **Testing**: Can test classes independently

### Why These Specific Techniques?

**Chosen**: CLAHE, Sobel, Canny, LBP, Morphology
- Proven computer vision methods
- Fast and reliable
- Available in scikit-image/OpenCV
- Well-understood behavior

**Not Chosen**: Gabor filters, HOG, SIFT
- More complex
- Slower
- Harder to interpret
- Overkill for this task

### Why Normalized [0, 1] Outputs?

1. **Comparability**: Can compare scores across classes
2. **Fusion**: Enables majority voting and thresholding
3. **Visualization**: Easy to display as grayscale images
4. **Standardization**: Consistent interface

---

## Limitations and Future Work

### Current Limitations

1. **Fixed parameters**: No adaptive tuning per image
2. **No context**: Each pixel processed independently
3. **No shape priors**: Doesn't use geometric constraints
4. **RGB only**: Can't leverage infrared or multispectral
5. **Scale sensitivity**: May fail on very high/low resolution images

### Potential Improvements

1. **Adaptive parameters**:
   - Estimate noise level, adjust smoothing
   - Estimate global contrast, adjust CLAHE

2. **Contextual reasoning**:
   - Buildings near roads are more likely buildings
   - Water near existing water more likely water

3. **Shape constraints**:
   - Buildings are rectangular
   - Roads are linear and connected
   - Water bodies are continuous

4. **Multi-scale processing**:
   - Detect large features at coarse scale
   - Refine boundaries at fine scale

5. **Learning-based tuning**:
   - Use ML to optimize feature weights
   - Learn per-class thresholds

---

## Conclusion

This heuristic pipeline leverages classical computer vision to create interpretable, class-specific detection algorithms. Each class uses a tailored combination of:

- **Spectral features** (color channels)
- **Structural features** (edges, texture)
- **Contextual features** (smoothness, contrast)
- **Geometric features** (orientation, shape)

The result is a robust baseline system that:
- Generates high-quality pre-labels for ML training
- Provides interpretable confidence scores
- Handles diverse satellite imagery
- Balances precision and recall appropriately

Feature weights and parameters can be tuned based on validation results to optimize for specific datasets or geographic regions.</parameter>