import time
import numpy as np
from jane_street import create_pipeline

def benchmark_inference():
    """Measure actual inference speed with/without PCA."""
    
    # Load data
    pipeline_full = create_pipeline('data/train.csv', apply_pca=False)
    X_full, y, weights, metadata = pipeline_full.load_data()
    
    pipeline_pca = create_pipeline('data/train.csv', apply_pca=True)
    X_pca, _, _, _ = pipeline_pca.load_data()
    
    # Train both models
    pipeline_full.train()
    pipeline_pca.train()
    
    # Benchmark
    n_runs = 100
    
    # Full features
    times_full = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pipeline_full.model.predict_proba(X_full)
        times_full.append(time.perf_counter() - start)
    
    # PCA features
    times_pca = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pipeline_pca.model.predict_proba(X_pca)
        times_pca.append(time.perf_counter() - start)
    
    mean_full = np.mean(times_full)
    mean_pca = np.mean(times_pca)
    speedup = mean_full / mean_pca
    
    print(f"Full features: {mean_full*1000:.2f}ms (±{np.std(times_full)*1000:.2f}ms)")
    print(f"PCA features:  {mean_pca*1000:.2f}ms (±{np.std(times_pca)*1000:.2f}ms)")
    print(f"Speedup:       {speedup:.2f}x")
    
    return {'full': mean_full, 'pca': mean_pca, 'speedup': speedup}
