from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Interface all models must implement."""
    
    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X, y):
        """Train model."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Return probabilities [prob_0, prob_1] for each sample."""
        pass
    
    def predict(self, X):
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
