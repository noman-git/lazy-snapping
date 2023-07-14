from cv2 import imread
from src.lazy_snapping import lazySnapping
from src.display import display_images


def main():
    # Define your images and seeds
    images = [("data/lady.PNG", "data/lady stroke 1.png"), 
              ("data/lady.PNG", "data/lady stroke 2.png"), 
              ("data/Mona-lisa.PNG", "data/Mona-lisa stroke 1.png"), 
              ("data/Mona-lisa.PNG", "data/Mona-lisa stroke 2.png"), 
              ("data/van Gogh.PNG", "data/van Gogh stroke.png")]
    N_values = [2, 32, 64, 98, 128]

    # Loop over your images and seeds
    for image_path, seed_path in images:
        # Read the images
        image = imread(image_path)
        seed = imread(seed_path)

        # Perform the lazy snapping
        for N in N_values:
            print(f'Result with {N} clusters: ')
            lazySnapping(image, seed, 'kmeans_exec', 3, N)
            lazySnapping(image, seed, 'skkmeans_exec', 3, N)

if __name__ == "__main__":
    main()