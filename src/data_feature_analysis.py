"""
Global statistical analysis of pixel-level features for semantic segmentation.

Performs class-conditional feature analysis on a large-scale dataset using
representative subsampling and produces comprehensive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import gaussian_kde
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import warnings
from src.cste import ClassInfo, FeatureInfo, DataPath

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10





# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def load_feature_and_label(feature_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature tensor and label mask."""
    features = np.load(feature_path)  # (H, W, F)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    
    # Resize label to match downsampled features if needed
    if features.shape[:2] != label.shape:
        label = cv2.resize(label, (features.shape[1], features.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    return features, label


def sample_balanced_pixels(
    csv_path: str,
    samples_per_class: int = 50000,
    max_images: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample balanced pixels across classes from multiple images.
    
    Args:
        csv_path: Path to CSV mapping file
        samples_per_class: Target samples per class
        max_images: Maximum images to process
        
    Returns:
        features: (N, F) array
        labels: (N,) array
    """
    df = pd.read_csv(csv_path)
    
    # Limit number of images for efficiency
    if len(df) > max_images:
        df = df.sample(n=max_images, random_state=42)
    
    # Storage for sampled data per class
    class_features = {c: [] for c in ClassInfo.CLASS_NAMES.keys()}
    class_counts = {c: 0 for c in ClassInfo.CLASS_NAMES.keys()}
    
    print(f"Sampling pixels from {len(df)} images...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        # Check if all classes have enough samples
        if all(count >= samples_per_class for count in class_counts.values()):
            break
        
        try:
            features, label = load_feature_and_label(row['feature_path'], row['label_path'])
        except Exception as e:
            continue
        
        H, W, F = features.shape
        
        # Reshape for sampling
        features_flat = features.reshape(-1, F)
        label_flat = label.flatten()
        
        # Sample from each class present in this image
        for class_id in ClassInfo.CLASS_NAMES.keys():
            remaining = samples_per_class - class_counts[class_id]
            
            if remaining <= 0:
                continue
            
            # Get pixels of this class
            mask = label_flat == class_id
            class_pixels = features_flat[mask]
            
            if len(class_pixels) == 0:
                continue
            
            # Sample (with replacement if needed)
            n_sample = min(remaining, len(class_pixels), 10000)  # Max 10k per image
            indices = np.random.choice(len(class_pixels), size=n_sample, replace=False)
            
            class_features[class_id].append(class_pixels[indices])
            class_counts[class_id] += n_sample
    
    # Combine all classes
    all_features = []
    all_labels = []
    
    for class_id, feat_list in class_features.items():
        if len(feat_list) == 0:
            continue
        
        class_feat = np.vstack(feat_list)
        
        # Subsample to exact target if we have too many
        if len(class_feat) > samples_per_class:
            indices = np.random.choice(len(class_feat), size=samples_per_class, replace=False)
            class_feat = class_feat[indices]
        
        all_features.append(class_feat)
        all_labels.append(np.full(len(class_feat), class_id))
        
        print(f"Class {ClassInfo.CLASS_NAMES[class_id]}: {len(class_feat)} samples")
    
    features_array = np.vstack(all_features)
    labels_array = np.concatenate(all_labels)
    
    print(f"\nTotal sampled pixels: {len(features_array)}")
    
    return features_array, labels_array


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_global_distributions(features: np.ndarray, output_dir: Path) -> None:
    """
    Analyze and visualize global feature distributions.
    
    Creates histograms and KDE plots for each feature.
    """
    print("\n" + "="*60)
    print("GLOBAL FEATURE DISTRIBUTIONS")
    print("="*60)
    
    n_features = features.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        feat_data = features[:, i]
        
        # Statistics
        mean_val = np.mean(feat_data)
        std_val = np.std(feat_data)
        min_val = np.min(feat_data)
        max_val = np.max(feat_data)
        
        print(f"{FeatureInfo.FEATURE_NAMES[i]:20s}: "
              f"mean={mean_val:.4f}, std={std_val:.4f}, "
              f"range=[{min_val:.4f}, {max_val:.4f}]")
        
        # Plot
        ax = axes[i]
        ax.hist(feat_data, bins=50, alpha=0.6, density=True, edgecolor='black')
        
        # KDE overlay
        try:

            kde = gaussian_kde(feat_data[::10])  # Subsample for speed
            x_range = np.linspace(min_val, max_val, 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_title(f"{FeatureInfo.FEATURE_NAMES[i]}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_global_distributions.png', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: 01_global_distributions.png")


def analyze_class_conditional(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """
    Analyze class-conditional feature distributions.
    
    Creates boxplots and violin plots for each feature across classes.
    """
    print("\n" + "="*60)
    print("CLASS-CONDITIONAL FEATURE ANALYSIS")
    print("="*60)
    
    n_features = features.shape[1]
    
    # Prepare data for seaborn
    df_list = []
    for i in range(n_features):
        for class_id in ClassInfo.CLASS_NAMES.keys():
            mask = labels == class_id
            feat_vals = features[mask, i]
            
            if len(feat_vals) == 0:
                continue
            
            # Subsample for visualization efficiency
            if len(feat_vals) > 5000:
                feat_vals = np.random.choice(feat_vals, 5000, replace=False)
            
            for val in feat_vals:
                df_list.append({
                    'Feature': FeatureInfo.FEATURE_NAMES[i],
                    'Class': ClassInfo.CLASS_NAMES[class_id],
                    'Value': val
                })
    
    df = pd.DataFrame(df_list)
    
    # Create boxplots
    print("\nGenerating boxplots...")
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for i, feat_name in enumerate(list(FeatureInfo.FEATURE_NAMES.values())[:18]):
        ax = axes[i]
        feat_df = df[df['Feature'] == feat_name]
        
        # Boxplot
        colors = [np.array(ClassInfo.CLASS_COLORS[j])/255.0 for j in range(5)]
        bp = ax.boxplot(
            [feat_df[feat_df['Class'] == ClassInfo.CLASS_NAMES[c]]['Value'].values 
             for c in range(5)],
            labels=[ClassInfo.CLASS_NAMES[c] for c in range(5)],
            patch_artist=True,
            showfliers=False
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(feat_name, fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3)
    
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_class_conditional_boxplots.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 02_class_conditional_boxplots.png")
    
    # Create violin plots for top features
    print("\nGenerating violin plots...")
    fig, axes = plt.subplots(5, 4, figsize=(16, 15))
    axes = axes.flatten()
    
    for i, feat_name in enumerate(list(FeatureInfo.FEATURE_NAMES.values())[:18]):
        ax = axes[i]
        feat_df = df[df['Feature'] == feat_name]
        
        sns.violinplot(
            data=feat_df,
            x='Class',
            y='Value',
            ax=ax,
            palette=[np.array(ClassInfo.CLASS_COLORS[c])/255.0 for c in range(5)],
            cut=0
        )
        
        ax.set_title(feat_name, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_class_conditional_violins.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 03_class_conditional_violins.png")


def analyze_discriminativeness(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """
    Compute and visualize feature discriminativeness using ANOVA F-score.
    """
    print("\n" + "="*60)
    print("FEATURE DISCRIMINATIVENESS")
    print("="*60)
    
    f_scores = []
    p_values = []
    
    for i in range(features.shape[1]):
        # Group feature values by class
        groups = [features[labels == c, i] for c in ClassInfo.CLASS_NAMES.keys()]
        
        # ANOVA F-test
        f_stat, p_val = stats.f_oneway(*groups)
        
        f_scores.append(f_stat)
        p_values.append(p_val)
        
        print(f"{FeatureInfo.FEATURE_NAMES[i]:20s}: F={f_stat:8.2f}, p={p_val:.2e}")
    
    # Sort by F-score
    sorted_indices = np.argsort(f_scores)[::-1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(f_scores))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(f_scores)))
    
    bars = ax.barh(y_pos, [f_scores[i] for i in sorted_indices], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FeatureInfo.FEATURE_NAMES[i] for i in sorted_indices])
    ax.set_xlabel('ANOVA F-score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Discriminative Power Ranking', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{f_scores[idx]:.1f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_feature_discriminativeness.png', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: 04_feature_discriminativeness.png")
    
    return f_scores, p_values


def analyze_correlations(features: np.ndarray, output_dir: Path) -> None:
    """
    Compute and visualize feature correlation matrix.
    """
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(FeatureInfo.FEATURE_NAMES)))
    ax.set_yticks(np.arange(len(FeatureInfo.FEATURE_NAMES)))
    ax.set_xticklabels(FeatureInfo.FEATURE_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(FeatureInfo.FEATURE_NAMES)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontweight='bold')
    
    # Add correlation values
    for i in range(len(FeatureInfo.FEATURE_NAMES)-1):
        for j in range(len(FeatureInfo.FEATURE_NAMES)-1):
            if abs(corr_matrix[i, j]) > 0.5 and i != j:
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=7)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_feature_correlations.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 05_feature_correlations.png")
    
    # Print highly correlated pairs
    print("\nHighly correlated feature pairs (|r| > 0.7):")
    for i in range(len(FeatureInfo.FEATURE_NAMES)-1):
        for j in range(i+1, len(FeatureInfo.FEATURE_NAMES)-1):
            if abs(corr_matrix[i, j]) > 0.7:
                print(f"  {FeatureInfo.FEATURE_NAMES[i]:20s} <-> "
                      f"{FeatureInfo.FEATURE_NAMES[j]:20s}: {corr_matrix[i, j]:.3f}")


def analyze_pca(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """
    Perform PCA and visualize low-dimensional projections.
    """
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA()
    features_pca = pca.fit_transform(features_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"Variance explained by first 2 PCs: {cumulative_var[1]:.2%}")
    print(f"Variance explained by first 5 PCs: {cumulative_var[4]:.2%}")
    
    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(range(1, len(explained_var)+1), explained_var, alpha=0.6, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontweight='bold')
    ax1.set_title('Scree Plot', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components', fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_pca_scree.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 06_pca_scree.png")
    
    # 2D scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Subsample for visualization
    n_plot = min(20000, len(features_pca))
    indices = np.random.choice(len(features_pca), n_plot, replace=False)
    
    for class_id in ClassInfo.CLASS_NAMES.keys():
        mask = labels[indices] == class_id
        ax.scatter(
            features_pca[indices][mask, 0],
            features_pca[indices][mask, 1],
            c=[np.array(ClassInfo.CLASS_COLORS[class_id])/255.0],
            label=ClassInfo.CLASS_NAMES[class_id],
            alpha=0.5,
            s=10,
            edgecolors='none'
        )
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontweight='bold', fontsize=12)
    ax.set_title('PCA Projection (2D)', fontsize=14, fontweight='bold')
    ax.legend(markerscale=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_pca_2d_projection.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: 07_pca_2d_projection.png")

    # 3D scatter plot

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for class_id in ClassInfo.CLASS_NAMES.keys():
        mask = labels[indices] == class_id
        ax.scatter(
            features_pca[indices][mask, 0],
            features_pca[indices][mask, 1],
            features_pca[indices][mask, 2],
            c=[np.array(ClassInfo.CLASS_COLORS[class_id])/255.0],
            label=ClassInfo.CLASS_NAMES[class_id],
            alpha=0.5,
            s=10,
            edgecolors='none'
        )
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontweight='bold', fontsize=12)
    ax.set_zlabel(f'PC3 ({explained_var[2]:.1%} variance)', fontweight='bold', fontsize=12)
    ax.set_title('PCA Projection (3D)', fontsize=14, fontweight='bold')
    ax.legend(markerscale=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '08_pca_3d_projection.png', bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_global_analysis(
    csv_path: str,
    output_dir: str = "analysis_results",
    samples_per_class: int = 50000,
    max_images: int = 500
) -> None:
    """
    Run complete global statistical analysis pipeline.
    
    Args:
        csv_path: Path to CSV mapping file
        output_dir: Directory to save figures
        samples_per_class: Number of pixels to sample per class
        max_images: Maximum number of images to process
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GLOBAL PIXEL-LEVEL FEATURE ANALYSIS")
    print("="*60)
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Max images: {max_images}")
    
    # Sample balanced pixels
    features, labels = sample_balanced_pixels(csv_path, samples_per_class, max_images)
    
    # Run analyses
    analyze_global_distributions(features, output_path)
    analyze_class_conditional(features, labels, output_path)
    analyze_discriminativeness(features, labels, output_path)
    analyze_correlations(features, output_path)
    analyze_pca(features, labels, output_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    run_global_analysis(
        csv_path="data/metadata/train_mapping.csv",
        output_dir="analysis_results",
        samples_per_class=50000,
        max_images=500
    )