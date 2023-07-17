import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB
from matplotlib import pyplot as plt


def display_images(bin_mask, image, seed_image_array):
    """Segment the image using the binary weights and display the results."""
    # Create a 3D mask from the 2D bin_mask array
    mask = np.stack([bin_mask]*3, axis=-1)

    # Create the segmented images by applying the mask
    cluster1_image_array = np.where(mask, image, 0)
    cluster2_image_array = np.where(mask, 0, image)

    # Convert the images to RGB
    stroke_img = cvtColor(seed_image_array, COLOR_BGR2RGB)
    og_image = cvtColor(image, COLOR_BGR2RGB)
    cluster1_image = cvtColor(cluster1_image_array, COLOR_BGR2RGB)
    cluster2_image = cvtColor(cluster2_image_array, COLOR_BGR2RGB)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(1, 4, figsize = (16, 4))

    # Display each image on a separate subplot
    for ax, image in zip(axs, [stroke_img, og_image, cluster1_image, cluster2_image]):
        ax.axis('off')
        ax.imshow(image)

    plt.show()