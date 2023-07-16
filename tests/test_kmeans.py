import numpy as np
import sys


sys.path.insert(0, '.')

from src.kmeans import MyKMeans

def test_my_kmeans():
    # Create a simple dataset
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Initialize a MyKMeans
    kmeans = MyKMeans(k=2, tol=0.001, max_iter=200, random_state=0)

    # Test the fit method
    kmeans.fit(X)
    assert np.array_equal(kmeans.centroids, np.array([[1., 2.], [10., 2.]]))

    # Test the predict method
    y_pred = kmeans.predict(X)
    assert np.array_equal(y_pred, np.array([0, 0, 0, 1, 1, 1]))

    # Test the _e_step method
    labels = kmeans._e_step(X)
    assert np.array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))

    # Test the _m_step method
    new_centroids = kmeans._m_step(X, labels)
    assert np.array_equal(new_centroids, np.array([[1., 2.], [10., 2.]]))