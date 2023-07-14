from .kmeans import MyKMeans
from sklearn.cluster import KMeans


def kmeans_exec(k, image_array):
    """Function to execute k-means clustering using custom implementation."""
    kmeanalgo = MyKMeans(k = k, random_state = 0)
    kmeanalgo.fit(image_array)
    centroids = kmeanalgo.centroids
    indices = kmeanalgo.predict(image_array)
    return centroids, indices

def skkmeans_exec(k, image_array):
    """Function to execute k-means clustering using built-in implementation."""
    kmeanalgo = KMeans(n_clusters=k, random_state=0)
    kmeanalgo.fit(image_array)
    centroids = kmeanalgo.cluster_centers_
    indices = kmeanalgo.labels_
    return centroids, indices