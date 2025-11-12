"""
Integration test for full pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from jane_street import create_pipeline


@pytest.fixture
def sample_data_file():
    """Create sample CSV file for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50  # Increased from 10 to allow PCA reduction
    
    data = {
        'date': np.repeat(range(10), 20),
        'weight': np.random.uniform(0.5, 2.0, n_samples),
        'resp': np.random.normal(0, 0.01, n_samples),
    }
    
    for i in range(n_features):
        # Add some correlated features so PCA can reduce dimensions
        if i < 10:
            data[f'feature_{i}'] = np.random.randn(n_samples)
        else:
            # Make later features correlated with earlier ones
            base_idx = i % 10
            data[f'feature_{i}'] = data[f'feature_{base_idx}'] + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    Path(f.name).unlink()


def test_pipeline_end_to_end(sample_data_file):
    """Test complete pipeline runs without errors."""
    
    pipeline = create_pipeline(
        data_path=sample_data_file,
        apply_pca=True,
        random_state=42
    )
    
    X, y, weights, metadata = pipeline.load_data()
    
    assert X.shape[0] == 200
    assert X.shape[1] <= 50  # PCA should reduce or keep same
    assert X.shape[1] < metadata['n_features']  # Should have reduced from 50
    assert len(y) == 200
    assert metadata['n_features'] == 50
    
    model = pipeline.train()
    assert model.is_fitted
    
    results = pipeline.evaluate(n_splits=3)
    
    assert 'mean_utility' in results
    assert 'std_utility' in results
    assert len(results['utilities']) == 3


def test_pipeline_without_pca(sample_data_file):
    """Test pipeline works without PCA."""
    
    pipeline = create_pipeline(
        data_path=sample_data_file,
        apply_pca=False,
        random_state=42
    )
    
    X, y, weights, metadata = pipeline.load_data()
    
    assert X.shape[1] == 50  # No reduction
    
    model = pipeline.train()
    assert model.is_fitted


def test_pipeline_summary(sample_data_file):
    """Test pipeline summary generation."""
    
    pipeline = create_pipeline(
        data_path=sample_data_file,
        apply_pca=True
    )
    
    pipeline.load_data()
    summary = pipeline.get_summary()
    
    assert 'n_samples' in summary
    assert 'n_features_original' in summary
    assert 'n_features_final' in summary
    assert summary['pca_applied'] is True
    assert summary['n_features_final'] < summary['n_features_original']
