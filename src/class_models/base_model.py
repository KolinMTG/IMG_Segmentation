"""Abstract base class for all segmentation models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import json
from pathlib import Path

from logger import get_logger
log = get_logger("model_base")

class BaseSegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    
    def __init__(self, num_classes: int, model_name: str):
        """
        Initialize base model.
        
        Args:
            num_classes: Number of segmentation classes
            model_name: Name identifier for the model
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.config = {}
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics/history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted segmentation masks
        """
        pass
    
    @abstractmethod
    def save(self, save_dir: str) -> None:
        """
        Save model and configuration.
        
        Args:
            save_dir: Directory to save model artifacts
        """
        pass
    
    @abstractmethod
    def load(self, save_dir: str) -> None:
        """
        Load model and configuration.
        
        Args:
            save_dir: Directory containing model artifacts
        """
        pass
    
    def _save_config(self, save_dir: str) -> None:
        """
        Save model configuration to JSON.
        
        Args:
            save_dir: Directory to save configuration
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        log.info(f"Saved config to {config_path}")
    
    def _load_config(self, save_dir: str) -> Dict[str, Any]:
        """
        Load model configuration from JSON.
        
        Args:
            save_dir: Directory containing configuration
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(save_dir) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        log.info(f"Loaded config from {config_path}")
        return config
