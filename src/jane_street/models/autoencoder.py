import numpy as np
import tensorflow as tf
from tensorflow import keras
from .base import BaseModel


class AutoencoderMLP(BaseModel):
    """
    Two-stage model: autoencoder for feature learning, then MLP for classification.
    """
    
    def __init__(
        self,
        encoding_dim=32,
        epochs=50,
        batch_size=1024,
        random_state=42
    ):
        super().__init__("AutoencoderMLP")
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.autoencoder = None
        self.encoder = None
        self.classifier = None
        
    def _build_autoencoder(self, input_dim):
        """Build autoencoder for unsupervised feature learning."""
        
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder
    
    def _build_classifier(self, encoding_dim):
        """Build MLP classifier on encoded features."""
        
        model = keras.Sequential([
            keras.layers.Input(shape=(encoding_dim,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y):
        """Train autoencoder then classifier."""
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        input_dim = X_array.shape[1]
        
        # Stage 1: Train autoencoder (unsupervised)
        self.autoencoder, self.encoder = self._build_autoencoder(input_dim)
        self.autoencoder.fit(
            X_array, X_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Stage 2: Encode features
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size, verbose=0)
        
        # Stage 3: Train classifier (supervised)
        self.classifier = self._build_classifier(self.encoding_dim)
        self.classifier.fit(
            X_encoded, y_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Return probabilities [prob_0, prob_1]."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_array = X.values if hasattr(X, 'values') else X
        
        X_encoded = self.encoder.predict(X_array, batch_size=self.batch_size, verbose=0)
        proba_pos = self.classifier.predict(X_encoded, batch_size=self.batch_size, verbose=0)
        proba_pos = proba_pos.flatten()
        
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
