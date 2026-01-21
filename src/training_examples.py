"""
Training and evaluation examples for all four segmentation models.

Demonstrates:
1. Training each model independently
2. Evaluating trained models
3. Feature ablation studies
4. Model comparison
"""

from src.cste import DataPath, FeatureInfo, ClassInfo
from src.models import KMeansModel, RandomForestModel, GradientBoostingModel, UNetModel
from src.logger import get_logger
import pandas as pd
from typing import Tuple

log = get_logger("train_examples")


def compute_reduced_csv(
    ratio: float = 1.0,
    train_csv: str = DataPath.CSV_FEATURE_MASK_MAPPING_TRAIN,
    val_csv: str = DataPath.CSV_FEATURE_MASK_MAPPING_VAL,
) -> Tuple[str, str]:
    """
    Generate reduced training and validation CSVs by randomly sampling rows.

    Args:
        ratio: Fraction of samples to keep (0 < ratio <= 1)
        train_csv: Path to original training CSV
        val_csv: Path to original validation CSV

    Returns:
        Tuple of (reduced_train_csv, reduced_val_csv)
    """
    if not (0 < ratio <= 1.0):
        raise ValueError(f"ratio must be between 0 and 1. Got {ratio}")

    log.info(f"Computing reduced CSVs with ratio: {ratio:.2f}")

    #! Load original CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    #! Sample rows
    train_sampled = train_df.sample(frac=ratio, random_state=42).reset_index(drop=True)
    val_sampled = val_df.sample(frac=ratio, random_state=42).reset_index(drop=True)

    #! Generate output file names
    reduced_train_csv = train_csv.replace(".csv", f"_reduced_{ratio:.2f}.csv")
    reduced_val_csv = val_csv.replace(".csv", f"_reduced_{ratio:.2f}.csv")

    #! Save reduced CSVs
    train_sampled.to_csv(reduced_train_csv, index=False)
    val_sampled.to_csv(reduced_val_csv, index=False)

    log.info(f"Reduced training CSV saved to: {reduced_train_csv}")
    log.info(f"Reduced validation CSV saved to: {reduced_val_csv}")

    return reduced_train_csv, reduced_val_csv


reduced_train_csv, reduced_val_csv = compute_reduced_csv(ratio=0.2)


# ============================================================================
# K-MEANS (UNSUPERVISED)
# ============================================================================


def train_kmeans():
    """Train K-means clustering model."""
    log.info("Training K-means model...")

    trainer = KMeansModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_UNET_SELECTION,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}kmeans/",
        batch_size=20,
        max_iter=100,
        random_state=42,
    )

    trainer.train()


def evaluate_kmeans():
    """Evaluate trained K-means model."""
    log.info("Evaluating K-means model...")

    trainer = KMeansModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_UNET_SELECTION,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}kmeans/",
    )

    metrics = trainer.evaluate()
    log.info(f"K-means mIoU: {metrics['mean_iou']:.4f}")

    return metrics


# ============================================================================
# RANDOM FOREST (PIXEL-WISE SUPERVISED)
# ============================================================================


def train_random_forest():
    """Train Random Forest classifier."""
    log.info("Training Random Forest...")

    trainer = RandomForestModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_UNET_SELECTION,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}random_forest/",
        n_estimators=100,
        max_depth=20,
        max_samples_per_image=5000,
        random_state=42,
        n_jobs=-1,
    )

    trainer.train()


def evaluate_random_forest():
    """Evaluate trained Random Forest model."""
    log.info("Evaluating Random Forest...")

    trainer = RandomForestModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_UNET_SELECTION,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}random_forest/",
    )

    metrics = trainer.evaluate()
    log.info(f"Random Forest mIoU: {metrics['mean_iou']:.4f}")

    return metrics


# ============================================================================
# U-NET (DEEP LEARNING)
# ============================================================================


def train_unet():
    """Train U-Net model."""
    log.info("Training U-Net...")

    trainer = UNetModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_RBG_ONLY,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}unet/",
        one_hot=True,
        epochs=50,
        batch_size=2,
        learning_rate=1e-3,
        filters_base=16,
    )

    trainer.train()


def evaluate_unet():
    """Evaluate trained U-Net model."""
    log.info("Evaluating U-Net...")

    trainer = UNetModel(
        train_csv=reduced_train_csv,
        val_csv=reduced_val_csv,
        feature_list=FeatureInfo.FEATURE_RBG_ONLY,
        num_classes=ClassInfo.NUM_CLASSES,
        output_dir=f"{DataPath.MODEL_DIR}unet/",
        one_hot=True,
    )

    metrics = trainer.evaluate()
    log.info(f"U-Net mIoU: {metrics['mean_iou']:.4f}")

    return metrics


# ============================================================================
# FEATURE ABLATION STUDY
# ============================================================================


def feature_ablation_study():
    """
    Test different feature combinations with Random Forest.

    Demonstrates feature ablation for understanding feature importance.
    """
    log.info("=" * 60)
    log.info("FEATURE ABLATION STUDY")
    log.info("=" * 60)

    feature_sets = {
        "color_only": [FeatureInfo.RED, FeatureInfo.GREEN, FeatureInfo.BLUE],
        "color_texture": [
            FeatureInfo.RED,
            FeatureInfo.GREEN,
            FeatureInfo.BLUE,
            FeatureInfo.LOCAL_VARIANCE,
            FeatureInfo.LOCAL_ENTROPY,
            FeatureInfo.LBP,
        ],
        "color_texture_gradient": [
            FeatureInfo.RED,
            FeatureInfo.GREEN,
            FeatureInfo.BLUE,
            FeatureInfo.LOCAL_VARIANCE,
            FeatureInfo.LOCAL_ENTROPY,
            FeatureInfo.GRADIENT_MAG,
            FeatureInfo.GRADIENT_ORIENT,
        ],
        "full_selection": FeatureInfo.FEATURE_UNET_SELECTION,
    }

    results = {}

    for name, features in feature_sets.items():
        log.info(f"\nTesting: {name} ({len(features)} features)")

        output_dir = f"{DataPath.MODEL_DIR}ablation_{name}/"

        #! Train with reduced parameters for faster testing
        trainer = RandomForestModel(
            train_csv=DataPath.CSV_FEATURE_MASK_MAPPING_TRAIN,
            val_csv=DataPath.CSV_FEATURE_MASK_MAPPING_VAL,
            feature_list=features,
            num_classes=ClassInfo.NUM_CLASSES,
            output_dir=output_dir,
            n_estimators=50,
            max_depth=15,
            max_samples_per_image=3000,
            random_state=42,
        )

        trainer.train()
        metrics = trainer.evaluate()

        results[name] = {
            "num_features": len(features),
            "mean_iou": metrics["mean_iou"],
            "macro_f1": metrics["macro"]["f1_score"],
        }

        log.info(f"  mIoU: {metrics['mean_iou']:.4f}")
        log.info(f"  Macro F1: {metrics['macro']['f1_score']:.4f}")

    log.info("\n" + "=" * 60)
    log.info("ABLATION STUDY SUMMARY")
    log.info("=" * 60)

    for name, res in results.items():
        log.info(f"{name}:")
        log.info(f"  Features: {res['num_features']}")
        log.info(f"  mIoU: {res['mean_iou']:.4f}")
        log.info(f"  Macro F1: {res['macro_f1']:.4f}")

    return results


# ============================================================================
# COMPARE ALL MODELS
# ============================================================================


def compare_all_models():
    """Train and evaluate all four models for comparison."""
    log.info("=" * 60)
    log.info("TRAINING AND COMPARING ALL MODELS")
    log.info("=" * 60)

    results = {}

    #! 1. K-Means
    log.info("\n[1/4] K-Means...")
    train_kmeans()
    results["kmeans"] = evaluate_kmeans()

    #! 2. Random Forest
    log.info("\n[2/4] Random Forest...")
    train_random_forest()
    results["random_forest"] = evaluate_random_forest()

    #! 3. Gradient Boosting
    log.info("\n[3/4] Gradient Boosting...")
    train_gradient_boosting()
    results["gradient_boosting"] = evaluate_gradient_boosting()

    #! 4. U-Net
    log.info("\n[4/4] U-Net...")
    train_unet()
    results["unet"] = evaluate_unet()

    #! Print comparison
    log.info("\n" + "=" * 60)
    log.info("MODEL COMPARISON")
    log.info("=" * 60)

    for model_name, metrics in results.items():
        log.info(f"\n{model_name.upper()}:")
        log.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        log.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        log.info(f"  Macro F1: {metrics['macro']['f1_score']:.4f}")
        log.info(f"  Weighted F1: {metrics['weighted']['f1_score']:.4f}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    #! Example 1: Train and evaluate individual models
    # train_kmeans()
    # evaluate_kmeans()

    # train_random_forest()
    # evaluate_random_forest()

    # train_gradient_boosting()
    # evaluate_gradient_boosting()

    # train_unet()
    # evaluate_unet()

    #! Example 2: Feature ablation study
    # feature_ablation_study()

    #! Example 3: Compare all models
    compare_all_models()
