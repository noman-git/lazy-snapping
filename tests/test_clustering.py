import sys
import numpy as np


sys.path.insert(0, '.')

from src.clustering import kmeans_exec, skkmeans_exec


def test_kmeans_exec() -> None:
    """Test with a simple 2D array and k=2, this tests the custom kmeans implementation as well"""
    centroids, indices = kmeans_exec(2, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert centroids.shape == (2, 2)
    assert indices.shape == (4,)
    
def test_skkmeans_exec() -> None:
    """Test with a simple 2D array and k=2"""
    centroids, indices = skkmeans_exec(2, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert centroids.shape == (2, 2)
    assert indices.shape == (4,)