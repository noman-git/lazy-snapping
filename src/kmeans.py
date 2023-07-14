import numpy as np


class MyKMeans:
    """KMeans clustering algorithm class."""

    def __init__(self, k=3, tol=0.001, max_iter=200, random_state=None):
        """Initialize KMeans object."""
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, data):
        """Fit KMeans model to data."""
        # Generate initial centroids randomly within the range of the data
        self.centroids = np.random.uniform(
            np.amin(data, axis=0), np.amax(data, axis=0), size=(self.k, data.shape[1]))

        for _ in range(self.max_iter):
            labels = self._e_step(data)
            new_centroids = self._m_step(data, labels)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            else:
                self.centroids = new_centroids

    def _e_step(self, data):
        """Expectation step."""
        return np.argmin(np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2), axis=1)

    def _m_step(self, data, labels):
        """Maximization step."""
        new_centroids = []
        for k in range(self.k):
            points_in_cluster = data[labels == k]
            if len(points_in_cluster) > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                # Reinitialize centroid randomly if no points assigned to it
                new_centroids.append(data[np.random.randint(data.shape[0])])
        return np.array(new_centroids)

    def predict(self, data):
        """Predict the closest cluster each sample in data belongs to."""
        return self._e_step(data)