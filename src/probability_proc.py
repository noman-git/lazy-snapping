import numpy as np


def weight(index):
    """Calculate the weight of each pixel in the image."""
    _, counts = np.unique(index, return_counts=True)
    # return the relative frequency of each unique value in the index array
    return counts / len(index)  

def calc_prob(image, c1, i1, c2, i2):
    """Calculate the probability of each pixel in the image belonging to foreground or background."""
    # Get the clusters' weights
    cluster1_weight = weight(i1)
    cluster2_weight = weight(i2)

    # Reshape the image array to 2D for broadcasting (n, 3)
    reshaped_img = image.reshape(-1, 3)

    # Calculate foreground probabilities
    cluster1_distances = np.linalg.norm(reshaped_img[:, np.newaxis] - c1, axis=2)
    cluster1_probs = np.sum(cluster1_weight * np.exp(-cluster1_distances), axis=1)

    # Calculate background probabilities
    cluster2_distances = np.linalg.norm(reshaped_img[:, np.newaxis] - c2, axis=2)
    cluster2_probs = np.sum(cluster2_weight * np.exp(-cluster2_distances), axis=1)

    # Assign pixels to foreground or background based on higher probability
    binary_weights = np.where(cluster1_probs > cluster2_probs, 255, 0)

    # Reshape the result back to the original image shape before returning 
    return binary_weights.reshape(image.shape[:2])