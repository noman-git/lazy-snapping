import numpy as np
from time import time
from .clustering import kmeans_exec, skkmeans_exec
from .probability_proc import calc_prob
from .display import display_images

def lazySnapping(image_array, seed_image_array, kmeans_choice, seed_k, k):
    """The main function to perform all steps of lazy snapping.""" 
    if kmeans_choice == 'kmeans_exec':
        kmeans_type = kmeans_exec
    elif kmeans_choice == 'skkmeans_exec':
        kmeans_type = skkmeans_exec
    else:
        raise ValueError('Invalid k-means type. Please choose either kmeans_exec or skkmeans_exec.')
    
    start_time = time()
    
    # Step 1) Reshape seed image array and perform k-means clustering on it  
    _, seed_indices = kmeans_type(seed_k, seed_image_array.reshape(-1, 3))
    
    #reshape the image back to original shape without the third dimension
    seed_indices = seed_indices.reshape(image_array.shape[:2])
    
    # Filter out the most frequent value (black pixels) in the seed image array
    unique, counts = np.unique(seed_indices, return_counts=True)
    unique = np.delete(unique, np.argmax(counts))
    
    # Step 2) extract the foreground and background pixels from the image array
    cluster1_pixels = image_array[seed_indices == unique[0]]
    cluster2_pixels = image_array[seed_indices == unique[1]]
    
    # Step 3) Perform k-means clustering on both seed clusters separately
    cluster1_centroids, cluster1_indices = kmeans_type(k, cluster1_pixels) 
    cluster2_centroids, cluster2_indices = kmeans_type(k, cluster2_pixels)
    
    # Step 4) Calculate the probability of each pixel in the image belonging to either cluster (foreground or background)
    bin_weight_mask = calc_prob(image_array, cluster1_centroids, cluster1_indices, cluster2_centroids, cluster2_indices)
    
    # Step 4) Display the results
    display_images(bin_weight_mask, image_array, seed_image_array)
    
    end_time = time()
    
    print(f'{kmeans_choice}Process time:')
    print(f'{end_time - start_time:.4f}s ({(end_time - start_time) / 60:.3f} minutes)')