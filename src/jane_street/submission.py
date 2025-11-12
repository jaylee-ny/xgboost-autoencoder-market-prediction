import janestreet

class CompetitionSubmitter:
    """
    Wraps trained model for Jane Street API submission.
    
    The competition uses a streaming API where test data is
    revealed incrementally to prevent look-ahead bias.
    """
    
    def __init__(self, model, pca=None):
        self.model = model
        self.pca = pca
        
    def submit(self):
        env = janestreet.make_env()
        iter_test = env.iter_test()
        
        for test_df, sample_prediction_df in iter_test:
            # Extract features
            feature_cols = [c for c in test_df.columns if c.startswith('feature_')]
            X_test = test_df[feature_cols]
            
            # Apply PCA if used during training
            if self.pca:
                X_test = self.pca.transform(X_test)
            
            # Predict (threshold at 0.5)
            predictions = self.model.predict_proba(X_test)[:, 1]
            actions = (predictions >= 0.5).astype(int)
            
            # Submit
            sample_prediction_df['action'] = actions
            env.predict(sample_prediction_df)
