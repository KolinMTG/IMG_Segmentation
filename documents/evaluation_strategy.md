# Evaluation Strategy for Imbalanced Semantic Segmentation

## Problem Statement

### Class Imbalance in Satellite Image Segmentation

Our dataset exhibits severe class imbalance across two dimensions:

1. **Pixel-level imbalance**: Surface area distribution
   - Field: 57.9% of all pixels
   - Woodland: 33.1% of all pixels
   - Water: 6.5% of all pixels
   - Road: 1.6% of all pixels
   - Building: 0.9% of all pixels

2. **Presence-level imbalance**: Frequency across images
   - Field: 91.4% of images
   - Woodland: 73.8% of images
   - Road: 37.3% of images
   - Water: 20.6% of images
   - Building: 16.0% of images

### Why Standard Metrics Fail

#### Pixel-Wise Accuracy is Inappropriate

**Trivial prediction**: Predict "Field" everywhere

```
Accuracy = (TP + TN) / Total_Pixels
         = 57.9% + (42.1% × 0%)  / 100%
         ≈ 57.9%
```

**Problem**: A model that never detects buildings, roads, or water achieves ~58% accuracy simply by exploiting class dominance.

#### Weighted Average Metrics are Insufficient

Even weighted F1-score gives disproportionate influence to dominant classes:

```
Weighted_F1 = Σ (class_frequency_i × F1_i)
            ≈ 0.579 × F1_field + 0.331 × F1_woodland + ...
```

Field and woodland dominate the score, masking poor performance on critical classes (buildings, roads).

---

## Chosen Evaluation Metrics

### Metric Hierarchy

| Priority | Metric | Purpose |
|----------|--------|---------|
| **Primary** | Macro-averaged IoU (mIoU) | Overall model quality, equal class weight |
| **Secondary** | Macro-averaged Dice | Complementary overlap measure |
| **Diagnostic** | Per-class Recall | Identify which classes are missed |
| **Optional** | Frequency-weighted mIoU | Hybrid metric for reporting |

---

## 1. Intersection over Union (IoU)

### Definition

For class `c`:

```
IoU_c = Intersection_c / Union_c
      = TP_c / (TP_c + FP_c + FN_c)
```

Where:
- `TP_c`: True positives (predicted=c, ground_truth=c)
- `FP_c`: False positives (predicted=c, ground_truth≠c)
- `FN_c`: False negatives (predicted≠c, ground_truth=c)

### Properties

- **Range**: [0, 1], higher is better
- **Symmetry**: Treats over-prediction and under-prediction equally
- **Strict**: Penalizes both false positives and false negatives
- **Interpretability**: Direct measure of spatial overlap

### Why IoU?

1. **Standard metric** in semantic segmentation (Pascal VOC, COCO, Cityscapes)
2. **Strict scoring**: Forces model to balance precision and recall
3. **No bias** toward dominant classes when macro-averaged

---

## 2. Dice Coefficient (F1-Score)

### Definition

For class `c`:

```
Dice_c = 2 × Intersection_c / (Prediction_c + GroundTruth_c)
       = 2 × TP_c / (2 × TP_c + FP_c + FN_c)
```

### Relationship to IoU

```
Dice = 2 × IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

Dice is always ≥ IoU, with larger difference for lower scores.

### Properties

- **Range**: [0, 1], higher is better
- **Less strict** than IoU (gives higher scores)
- **Emphasizes true positives** more than IoU
- **Popular in medical imaging** (class imbalance common)

### Why Dice?

1. **Complementary to IoU**: Provides different perspective on overlap
2. **Smoother gradients**: More forgiving of small errors
3. **Medical imaging heritage**: Proven effectiveness with rare classes

---

## 3. Macro-Averaged Aggregation

### Definition

```
mIoU = (1/N) × Σ IoU_c    for all classes c
mDice = (1/N) × Σ Dice_c  for all classes c
```

Where `N` is the number of classes.

### Critical Property: Equal Class Weight

**Each class contributes equally**, regardless of pixel frequency:

```
mIoU = (IoU_field + IoU_building + IoU_woodland + IoU_water + IoU_road) / 5
     = 0.20 × IoU_field + 0.20 × IoU_building + ... + 0.20 × IoU_road
```

### Mitigation of Class Dominance

**Example**: Trivial "Field-only" prediction

```
Per-class IoU:
  Field:     high    (0.75)  [dominates pixels]
  Building:  0.00           [completely missed]
  Woodland:  0.00           [completely missed]
  Water:     0.00           [completely missed]
  Road:      0.00           [completely missed]

Pixel-wise accuracy: ~58%
mIoU: (0.75 + 0 + 0 + 0 + 0) / 5 = 0.15 (15%)
```

**Result**: Trivial prediction scores poorly despite high pixel accuracy.

---

## 4. Per-Class Recall (Sensitivity)

### Definition

For class `c`:

```
Recall_c = TP_c / (TP_c + FN_c)
         = TP_c / Total_GroundTruth_c
```

### Purpose

**Diagnostic metric** to identify which classes are being missed:

- High Recall, Low IoU → Over-predicting (low precision)
- Low Recall, High IoU → Under-predicting (conservative)
- Low Recall, Low IoU → Poor detection overall

### Why Recall?

1. **Detects missing predictions**: Essential for rare classes
2. **Interpretable**: "What percentage of true buildings were found?"
3. **Actionable**: Guides model improvement

---

## 5. Frequency-Weighted Metrics (Optional)

### Definition

```
Frequency-weighted_mIoU = Σ (frequency_c × IoU_c)
```

Where `frequency_c` is the pixel-level or presence-level frequency of class `c`.

### When to Use

- **Reporting to stakeholders**: Reflects real-world importance
- **Secondary metric**: Not for primary model selection
- **Interpretability**: "What percentage of dataset pixels are correctly classified?"

### Why Optional?

- **Still biased** toward dominant classes (by design)
- **Useful context** but not primary objective
- **Complements macro metrics** without replacing them

---

## Handling Missing Classes

### Scenario: Class Absent in Image

**Question**: If class `c` is not present in ground truth for an image, how to handle?

**Strategy**: **Include with zero score**

```python
if class_c in ground_truth and class_c not in prediction:
    IoU_c = 0.0  # Penalize missing prediction
elif class_c in prediction and class_c not in ground_truth:
    IoU_c = 0.0  # Penalize false detection
elif class_c not in ground_truth and class_c not in prediction:
    IoU_c = N/A  # Exclude from average for this image
```

**Rationale**:
- Penalizes models that fail to detect rare classes
- Avoids inflating scores by ignoring difficult cases
- Aligns with real-world deployment (missing a building is bad)

---

## Mathematical Coherence Check

### Property 1: Non-Trivial Optimum

A model maximizing mIoU **cannot** succeed by predicting only dominant classes.

**Proof by example**:
```
Predict only Field:
  mIoU = (high + 0 + 0 + 0 + 0) / 5 = ~15%

Balanced prediction:
  mIoU = (high + medium + medium + low + low) / 5 = ~50%
```

### Property 2: Class Contribution Symmetry

```
∂(mIoU) / ∂(IoU_field) = ∂(mIoU) / ∂(IoU_building) = 1/N
```

Every class has **equal gradient influence** on the final score.

### Property 3: Strict Penalty for Absence

```
If IoU_building = 0, then mIoU ≤ 0.8  (4/5 theoretical maximum)
```

Missing even one class caps the achievable score.

---

## Evaluation Protocol

### Per-Image Evaluation

For each image:

1. Compute per-class IoU, Dice, Recall
2. Macro-average to get image-level mIoU, mDice
3. Store per-class scores for analysis

### Dataset-Level Aggregation

Across all images:

1. **Mean of per-image mIoU**: Primary metric
   ```
   Final_mIoU = mean(mIoU_image_1, mIoU_image_2, ...)
   ```

2. **Per-class average**: Secondary diagnostic
   ```
   Avg_IoU_building = mean(IoU_building across all images)
   ```

### Reporting Structure

```
Primary Metrics:
  - mIoU: 0.487
  - mDice: 0.612

Per-Class Performance:
  - Field:    IoU=0.78, Dice=0.87, Recall=0.82
  - Building: IoU=0.45, Dice=0.62, Recall=0.58
  - Woodland: IoU=0.52, Dice=0.68, Recall=0.65
  - Water:    IoU=0.38, Dice=0.55, Recall=0.48
  - Road:     IoU=0.31, Dice=0.47, Recall=0.42
```

---

## Comparison to Alternatives

| Metric | Class Weighting | Strengths | Weaknesses |
|--------|----------------|-----------|------------|
| **Pixel Accuracy** | By pixel count | Simple, fast | Dominated by frequent classes |
| **Weighted F1** | By pixel count | Standard in ML | Still biased toward large classes |
| **mIoU (ours)** | Equal (1/N) | No class bias, standard in segmentation | May underweight rare but large objects |
| **mDice (ours)** | Equal (1/N) | Less strict than IoU | Higher scores may be misleading |
| **Per-class Recall** | N/A | Diagnostic clarity | Doesn't measure precision |

---

## Implementation Requirements

### Robustness

- **Zero-division handling**: If Union=0, define IoU=1 (perfect match)
- **Missing classes**: Exclude from average (don't force 0)
- **Numerical stability**: Use float64 for accumulation

### Efficiency

- **Vectorization**: Use NumPy broadcasting
- **Avoid loops**: Compute all classes simultaneously where possible
- **Memory**: O(C) where C is number of classes (not O(H×W))

### Extensibility

- **Dynamic class detection**: Read classes from masks, not hardcoded
- **Modular functions**: Separate per-class, aggregation, logging
- **Configuration**: Easy to add new metrics or change weights

---

## Conclusion

### Primary Metric: **Macro-averaged IoU (mIoU)**

**Justification**:
- Equal weight to all classes
- Standard in semantic segmentation literature
- Strict penalization of errors
- Prevents trivial solutions

### Supporting Metrics

- **mDice**: Complementary overlap measure
- **Per-class Recall**: Diagnostic for rare class detection
- **Frequency-weighted mIoU**: Optional for stakeholder reporting

### Success Criteria

A good model must:
1. Achieve high mIoU (≥0.50 for baseline, ≥0.65 for production)
2. Have no class with IoU < 0.20 (avoid catastrophic failures)
3. Maintain balanced per-class Recall (no class <30% recall)

This evaluation strategy ensures that model improvements on rare but critical classes (buildings, roads) are properly rewarded, while trivial dominant-class predictions are penalized.