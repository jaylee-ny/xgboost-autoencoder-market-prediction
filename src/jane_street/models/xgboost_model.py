import numpy as np
import xgboost as xgb
from .base import BaseModel


class XGBoostModel(BaseModel):

    def __init__(
        self,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ):
        super().__init__("XGBoost")
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
        }
        self.n_estimators = n_estimators
        self.model = None
        
    def fit(self, X, y):
        """Train XGBoost."""
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Return [prob_0, prob_1] for each sample."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        dtest = xgb.DMatrix(X)
        proba_pos = self.model.predict(dtest)
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
