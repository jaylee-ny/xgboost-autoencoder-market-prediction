from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from .base import BaseModel
from ..constants import AUTOENCODER_DEFAULTS, DECISION_THRESHOLD


class AutoencoderMLP(BaseModel):
    
    def __init__(
        self,
        encoding_dim: int = AUTOENCODER_DEFAULTS['encoding_dim'],
        epochs: int = AUTOENCODER_DEFAULTS['epochs'],
        batch_size: int = AUTOENCODER_DEFAULTS['batch_size'],
        learning_rate: float = AUTOENCODER_DEFAULTS['learning_rate'],
        random_state: int = 42
    ):
        super().__init__("AutoencoderMLP")
        
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.autoencoder: Optional[keras.Model] = None
        self.encoder: Optional[keras.Model] = None
        self.classifier: Optional[keras.Model] = None
        self.threshold = DECISION_THRESHOLD
        
    def _build_autoencoder(self, input_dim: int) -> tuple:
        """Build autoencoder for unsupervised feature learning."""
        input_layer = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return autoencoder, encoder
    
    def _build_classifier(self, encoding_dim: int) -> keras.Model:
        """Build MLP classifier on encoded features."""
        model = keras.Sequential([
            keras.layers.Input(shape=(encoding_dim,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoencoderMLP':
        """
        Two-stage training:
        1. Train autoencoder to learn compressed representations (unsupervised)
        2. Train classifier on encoded features (supervised)
        """
        self._validate_inputs(X, y)
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        input_dim = X_array.shape[1]
        
        print(f"  Stage 1: Training autoencoder (unsupervised)...")
        self.autoencoder, self.encoder = self._build_autoencoder(input_dim)
        
        self.autoencoder.fit(
            X_array, X_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        print(f"  Stage 2: Training classifier on encoded features...")
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size, verbose=0)
        
        self.classifier = self._build_classifier(self.encoding_dim)
        
        self.classifier.fit(
            X_encoded, y_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_array = X.values if hasattr(X, 'values') else X
        
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size, verbose=0)
        proba_pos = self.classifier.predict(X_encoded, batch_size=self.batch_size, verbose=0)
        proba_pos = proba_pos.flatten()
        
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is None:
            threshold = self.threshold
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        if len(X) != len(y):
            raise ValueError(f"Shape mismatch: {len(X)} vs {len(y)}")
        
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"Target must be binary, got: {y.unique()}")
