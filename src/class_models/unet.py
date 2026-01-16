"""U-Net model for semantic segmentation."""

import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.base_model import BaseSegmentationModel
from logger import get_logger
log = get_logger("model_unet")



class UNet(BaseSegmentationModel):
    """
    U-Net architecture for semantic segmentation.
    
    Simple encoder-decoder with skip connections.
    Handles class imbalance with weighted loss.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        filters: int = 32,
        use_one_hot: bool = False
    ):
        """
        Initialize U-Net model.
        
        Args:
            num_classes: Number of segmentation classes
            input_shape: Input shape (H, W, C)
            filters: Number of base filters (doubled at each level)
            use_one_hot: If True, expects one-hot encoded masks
        """
        super().__init__(num_classes, 'UNet')
        
        self.input_shape = input_shape
        self.filters = filters
        self.use_one_hot = use_one_hot
        
        #! Store configuration
        self.config = {
            'num_classes': num_classes,
            'input_shape': input_shape,
            'filters': filters,
            'use_one_hot': use_one_hot
        }
        
        #! Detect device (GPU if available, else CPU)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            log.info(f"GPU available: {len(gpus)} device(s)")
            #! Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            log.info("No GPU found, using CPU")
        
        #! Build model
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Build U-Net architecture.
        
        Returns:
            Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        #! Encoder (downsampling path)
        c1 = self._conv_block(inputs, self.filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self._conv_block(p1, self.filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self._conv_block(p2, self.filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self._conv_block(p3, self.filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        #! Bottleneck
        c5 = self._conv_block(p4, self.filters * 16)
        
        #! Decoder (upsampling path)
        u6 = layers.Conv2DTranspose(self.filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = self._conv_block(u6, self.filters * 8)
        
        u7 = layers.Conv2DTranspose(self.filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = self._conv_block(u7, self.filters * 4)
        
        u8 = layers.Conv2DTranspose(self.filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = self._conv_block(u8, self.filters * 2)
        
        u9 = layers.Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = self._conv_block(u9, self.filters)
        
        #! Output layer
        if self.use_one_hot:
            outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(c9)
        else:
            outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(c9)
        
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        
        log.info(f"Built U-Net with {model.count_params():,} parameters")
        
        return model
    
    def _conv_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """
        Convolutional block with two conv layers.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 8,
        class_weights: np.ndarray = None,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train U-Net model.
        
        Args:
            X_train: Training features, shape (N, H, W, C)
            y_train: Training masks, shape (N, H, W) or (N, H, W, num_classes)
            X_val: Validation features (optional)
            y_val: Validation masks (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        log.info(f"Training U-Net: epochs={epochs}, batch_size={batch_size}")
        
        #! Convert class weights to dictionary format
        if class_weights is not None:
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            log.info(f"Using class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        #! Prepare labels
        if not self.use_one_hot and len(y_train.shape) == 3:
            #! Convert to categorical format for loss function
            y_train_categorical = tf.keras.utils.to_categorical(y_train, self.num_classes)
            if y_val is not None:
                y_val_categorical = tf.keras.utils.to_categorical(y_val, self.num_classes)
        else:
            y_train_categorical = y_train
            y_val_categorical = y_val if y_val is not None else None
        
        #! Compile model with weighted loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', self._iou_metric, self._dice_metric]
        )
        
        #! Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if y_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if y_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        #! Train model
        validation_data = (X_val, y_val_categorical) if X_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train_categorical,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        log.info("Training completed")
        
        return history.history
    
    def predict(self, X: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features, shape (N, H, W, C)
            batch_size: Batch size for prediction
            
        Returns:
            Predicted masks, shape (N, H, W) with class IDs
        """
        #! Predict probabilities
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        
        #! Convert to class IDs
        preds = np.argmax(probs, axis=-1)
        
        return preds
    
    @staticmethod
    def _iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute mean IoU metric.
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Mean IoU score
        """
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        intersection = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
        union = tf.cast(tf.size(y_true), tf.float32)
        
        return intersection / union
    
    @staticmethod
    def _dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute Dice coefficient.
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Dice score
        """
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        intersection = 2.0 * tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
        union = tf.cast(tf.size(y_true) * 2, tf.float32)
        
        return intersection / union
    
    def save(self, save_dir: str) -> None:
        """
        Save model weights and configuration.
        
        Args:
            save_dir: Directory to save model artifacts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        #! Save model weights
        model_path = save_path / 'unet_model.h5'
        self.model.save_weights(str(model_path))
        log.info(f"Saved model weights to {model_path}")
        
        #! Save configuration
        self._save_config(save_dir)
    
    def load(self, save_dir: str) -> None:
        """
        Load model weights and configuration.
        
        Args:
            save_dir: Directory containing model artifacts
        """
        #! Load configuration
        self.config = self._load_config(save_dir)
        
        #! Rebuild model
        self.num_classes = self.config['num_classes']
        self.input_shape = tuple(self.config['input_shape'])
        self.filters = self.config['filters']
        self.use_one_hot = self.config['use_one_hot']
        
        self.model = self._build_model()
        
        #! Load weights
        model_path = Path(save_dir) / 'unet_model.h5'
        self.model.load_weights(str(model_path))
        log.info(f"Loaded model weights from {model_path}")
