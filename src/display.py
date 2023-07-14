import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB
from matplotlib import pyplot as plt


def display_images(bw, image, seed_image_array):
    """Segment the image using the binary weights and display the results."""
    # Create a 3D mask from the 2D bw array
    mask = np.stack([bw]*3, axis=-1)

    # Create the segmented images by applying the mask
    cluster1_image_array = np.where(mask, image, 0)
    cluster2_image_array = np.where(mask, 0, image)

    # Convert the images to RGB
    dst0 = cvtColor(seed_image_array, COLOR_BGR2RGB)
    dst1 = cvtColor(image, COLOR_BGR2RGB)
    dst2 = cvtColor(cluster1_image_array, COLOR_BGR2RGB)
    dst3 = cvtColor(cluster2_image_array, COLOR_BGR2RGB)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(1, 4, figsize = (16, 4))

    # Display each image on a separate subplot
    for ax, dst in zip(axs, [dst0, dst1, dst2, dst3]):
        ax.axis('off')
        ax.imshow(dst)

    plt.show()