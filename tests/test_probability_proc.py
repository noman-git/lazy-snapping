import sys
import numpy as np


sys.path.insert(0, '.')

from src.probability_proc import weight, calc_bin_mask


def test_weight() -> None:
    """Test with a simple 1D array"""
    weights = weight(np.array([0, 0, 1, 1, 1, 2, 2, 2, 2]))
    assert np.allclose(weights, np.array([0.22222222, 0.33333333, 0.44444444]))

def test_calc_prob() -> None:
    """Test with simple inputs close to the centroids"""
    image = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], 
                  [[1, 2, 3], [1, 2, 3], [1, 2, 3]], 
                  [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
    c1 = np.array([[1, 2, 3]])
    i1 = np.array([0])
    c2 = np.array([[7, 8, 9]])
    i2 = np.array([0])
    result = calc_bin_mask(image, c1, i1, c2, i2)
    assert result.shape == (3, 3)
    assert np.all(result == 255)