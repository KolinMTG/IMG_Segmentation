"""Training pipeline for segmentation models."""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from data.dataset import SegmentationDataset
from models.base_model import BaseSegmentationModel
from evaluation.metrics import (
    evaluate_model,
    save_evaluation_report,
    print_evaluation_summary
)


class SegmentationTrainer:
    """
    Training pipeline for segmentation models.
    
    Handles data loading, training, evaluation, and saving.
    """
    
    def __init__(
        self,
        model: BaseSegmentationModel,
        train_dataset: SegmentationDataset,
        val_dataset: Optional[SegmentationDataset] = None,
        output_dir: str = './outputs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory to save outputs
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Initialized trainer for {model.model_name}")
        log.info(f"Output directory: {self.output_dir}")
    
    def train(self, **train_kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            **train_kwargs: Training parameters passed to model.train()
            
        Returns:
            Training history/metrics
        """
        log.info(f"\n{'='*50}")
        log.info(f"TRAINING {self.model.model_name}")
        log.info(f"{'='*50}\n")
        
        start_time = time.time()
        
        #! Load training data
        log.info("Loading training data...")
        X_train, y_train = self.train_dataset.get_all_data()
        log.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        #! Load validation data if available
        X_val, y_val = None, None
        if self.val_dataset is not None:
            log.info("Loading validation data...")
            X_val, y_val = self.val_dataset.get_all_data()
            log.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        #! Compute class weights for imbalanced data (supervised models only)
        if hasattr(self.model, 'model_name') and self.model.model_name in ['UNet', 'RandomForest']:
            class_weights = self.train_dataset.compute_class_weights()
            if 'class_weights' not in train_kwargs:
                train_kwargs['class_weights'] = class_weights
        
        #! Train model
        log.info("\nStarting training...")
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            **train_kwargs
        )
        
        elapsed_time = time.time() - start_time
        log.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        #! Save training history
        history['training_time_seconds'] = elapsed_time
        self._save_training_history(history)
        
        return history
    
    def evaluate(
        self,
        test_dataset: SegmentationDataset,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for prediction
            
        Returns:
            Evaluation metrics
        """
        log.info(f"\n{'='*50}")
        log.info(f"EVALUATING {self.model.model_name}")
        log.info(f"{'='*50}\n")
        
        #! Load test data
        log.info("Loading test data...")
        X_test, y_test = test_dataset.get_all_data()
        log.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        
        #! Evaluate
        results = evaluate_model(
            self.model,
            X_test,
            y_test,
            self.model.num_classes,
            batch_size=batch_size
        )
        
        #! Print summary
        print_evaluation_summary(results, self.model.num_classes)
        
        #! Save report
        save_evaluation_report(
            results,
            str(self.output_dir),
            self.model.model_name
        )
        
        return results
    
    def save_model(self) -> None:
        """Save trained model to output directory."""
        log.info(f"Saving model to {self.output_dir}...")
        self.model.save(str(self.output_dir))
        log.info("Model saved successfully")
    
    def _save_training_history(self, history: Dict[str, Any]) -> None:
        """
        Save training history to JSON.
        
        Args:
            history: Training history dictionary
        """
        import json
        
        history_path = self.output_dir / f'{self.model.model_name}_training_history.json'
        
        #! Convert numpy types to native Python types
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable_history[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_history[key] = int(value)
            elif isinstance(value, list):
                #! Handle lists that may contain numpy types
                serializable_history[key] = [
                    float(x) if isinstance(x, (np.float32, np.float64)) else x
                    for x in value
                ]
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        log.info(f"Saved training history to {history_path}")


def train_and_evaluate_model(
    model: BaseSegmentationModel,
    train_csv: str,
    test_csv: str,
    num_classes: int,
    feature_ids: Optional[List[int]] = None,
    val_split: float = 0.2,
    output_dir: str = './outputs',
    **train_kwargs
) -> Dict[str, Any]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        model: Model to train
        train_csv: Path to training CSV
        test_csv: Path to test CSV
        num_classes: Number of classes
        feature_ids: List of feature indices to use
        val_split: Validation split fraction
        output_dir: Output directory for model and results
        **train_kwargs: Additional training parameters
        
    Returns:
        Dictionary with training and evaluation results
    """
    log.info(f"\n{'='*70}")
    log.info(f"COMPLETE TRAINING PIPELINE: {model.model_name}")
    log.info(f"{'='*70}\n")
    
    #! Create datasets
    from data.dataset import create_train_val_split
    
    #! Split training data
    train_csv_split, val_csv_split = create_train_val_split(
        train_csv,
        val_split=val_split
    )
    
    #! Create dataset objects
    use_one_hot = hasattr(model, 'use_one_hot') and model.use_one_hot
    
    train_dataset = SegmentationDataset(
        train_csv_split,
        num_classes,
        feature_ids=feature_ids,
        one_hot=use_one_hot
    )
    
    val_dataset = SegmentationDataset(
        val_csv_split,
        num_classes,
        feature_ids=feature_ids,
        one_hot=use_one_hot
    )
    
    test_dataset = SegmentationDataset(
        test_csv,
        num_classes,
        feature_ids=feature_ids,
        one_hot=False  # Always use integer masks for evaluation
    )
    
    #! Create trainer
    trainer = SegmentationTrainer(
        model,
        train_dataset,
        val_dataset,
        output_dir=output_dir
    )
    
    #! Train
    training_history = trainer.train(**train_kwargs)
    
    #! Evaluate
    evaluation_results = trainer.evaluate(test_dataset)
    
    #! Save model
    trainer.save_model()
    
    #! Combine results
    results = {
        'training': training_history,
        'evaluation': evaluation_results
    }
    
    log.info(f"\n{'='*70}")
    log.info(f"PIPELINE COMPLETED: {model.model_name}")
    log.info(f"{'='*70}\n")
    
    return results
