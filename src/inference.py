import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
from tensorflow import keras
from src.cste import DataPath, FeatureInfo
from src.post_treatement import posttreat_pipeline


def run_unet_inference(
    model_path: str,
    csv_path: str = DataPath.CSV_FEATURE_MASK_MAPPING_TEST,
    output_dir: str = DataPath.UNET_INFERENCE_DIR,
    selected_features: List[int] = FeatureInfo.FEATURE_RBG_ONLY,
    output_size: Tuple[int, int] | None = (512, 512),
    posttreatment: bool = False,
    **kwargs,
) -> None:
    """
    Run inference with a trained Keras segmentation model and save predicted masks as PNG.

    Args:
        model_path: Path to the .keras model
        csv_path: CSV containing img_id and feature_path
        output_dir: Directory where predicted masks will be saved
        selected_features: List of feature indices to use for inference
        output_size: Desired output mask size as (width, height)
        posttreatment: Whether to apply post-treatment to predicted masks
        **kwargs: Additional arguments for post-treatment (check posttreat_pipeline function)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = keras.models.load_model(model_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    required_columns = {"img_id", "feature_path"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    for idx, row in df.iterrows():
        img_id = row["img_id"]
        feature_path = row["feature_path"]

        # Load features using mmap to limit memory usage
        features = np.load(feature_path, mmap_mode="r")

        # Select features along channel axis
        features = features[:, :, selected_features]

        # Add batch dimension
        features_batch = np.expand_dims(features, axis=0)

        # Predict class probabilities
        pred_probs = model.predict(features_batch, verbose=0)[0]

        # Convert probabilities to class labels
        pred_mask = np.argmax(pred_probs, axis=-1).astype(np.uint8)

        # Resize mask if required (nearest neighbor only)
        if output_size is not None:
            pred_mask = cv2.resize(
                pred_mask, output_size, interpolation=cv2.INTER_NEAREST
            )

        if posttreatment:
            pred_mask = posttreat_pipeline(pred_mask, **kwargs)

        # Save mask as single-channel PNG
        output_path = os.path.join(output_dir, f"{img_id}_mask.png")
        cv2.imwrite(output_path, pred_mask)

        # Explicit cleanup
        del features, features_batch, pred_probs, pred_mask

        if (idx + 1) % 20 == 0:
            print(f"Inference progress: {idx + 1}/{len(df)}")

    print("Inference completed successfully.")
