"""
Satellite Image Segmentation Pipeline
======================================

Project Structure:
-----------------
segmentation_pipeline/
│
├── data/
│   └── dataset.py              # Dataset utilities
│
├── models/
│   ├── base_model.py           # Abstract base class
│   ├── unet.py                 # U-Net implementation
│   ├── random_forest.py        # Random Forest classifier
│   └── kmeans.py               # K-Means clustering
│
├── training/
│   └── trainer.py              # Training pipeline
│
├── evaluation/
│   └── metrics.py              # Evaluation metrics
│
├── utils/
│   └── helpers.py              # Helper functions
│
└── main.py                     # Example usage

Design Choices:
--------------

1. **U-Net Architecture**:
   - Simple encoder-decoder with skip connections
   - Categorical cross-entropy with class weights for imbalance
   - Automatic GPU detection, CPU fallback

2. **Classical Models**:
   - **Random Forest**: Best supervised baseline, handles class imbalance well
   - **K-Means**: Fast unsupervised clustering for exploratory analysis
   - Both optimized with multiprocessing

3. **CPU Optimization**:
   - Batch processing for memory efficiency
   - Multiprocessing for classical models
   - Mixed precision training for U-Net (if GPU available)
   - Efficient data loading with generators

4. **Class Imbalance Handling**:
   - U-Net: class-weighted loss
   - Random Forest: class_weight='balanced'
   - K-Means: post-hoc mapping to majority class

5. **Memory Management**:
   - Generator-based data loading
   - Lazy loading of features
   - Batch size tuning for 16GB RAM
"""