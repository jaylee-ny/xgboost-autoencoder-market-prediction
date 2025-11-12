import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from jane_street.data.loader import DataLoader


def test_data_loader_file_not_found():
    """Test error when file doesn't exist."""
    loader = DataLoader('nonexistent.csv')
    
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_data_loader_loads_features():
    """Test loading features correctly."""
    # Create temp CSV
    data = pd.DataFrame({
        'date': [1, 2, 3],
        'feature_0': [1.0, 2.0, 3.0],
        'feature_1': [0.5, 1.5, 2.5],
        'resp': [0.01, -0.01, 0.02],
        'weight': [1.0, 1.0, 1.0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        loader = DataLoader(temp_path)
        X, y, weights, metadata = loader.load()
        
        assert X.shape == (3, 2)
        assert len(y) == 3
        assert metadata['n_features'] == 2
        assert metadata['n_samples'] == 3
    finally:
        Path(temp_path).unlink()


def test_data_loader_binary_target():
    """Test target conversion to binary."""
    data = pd.DataFrame({
        'feature_0': [1.0, 2.0, 3.0],
        'resp': [0.01, -0.01, 0.02],
        'weight': [1.0, 1.0, 1.0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        loader = DataLoader(temp_path)
        X, y, weights, metadata = loader.load()
        
        assert y.dtype == int
        assert set(y.unique()).issubset({0, 1})
        assert y.tolist() == [1, 0, 1]
    finally:
        Path(temp_path).unlink()
