import argparse
import matplotlib.pyplot as plt
from skimage import io, transform, filters, exposure
import random
import numpy as np
from plantcv import plantcv as pcv
import cv2

def image_random_rotation(image):
    random_rotation_degree = random.randint(0, 360)
    return pcv.transform.rotate(image, random_rotation_degree, True)

def image_flip(image):
    return pcv.flip(image, direction='vertical')

def image_blured(image):
    return pcv.gaussian_blur(image, ksize=(3, 3), sigma_x=2, sigma_y=2)
    # return filters.gaussian(image, sigma=(2, 2), truncate=3.5, channel_axis=-1)

def image_contrasted(image):
    return exposure.equalize_hist(image)

def image_distortion(image):
    max_distortion = 0.2  # Maximum distortion factor
    distortion_factors = np.random.uniform(-max_distortion, max_distortion, size=4)
    transform_matrix = np.array([[1 + distortion_factors[0], distortion_factors[1], 0],
                                [distortion_factors[2], 1 + distortion_factors[3], 0],
                                [0, 0, 1]])
    return transform.warp(image, transform_matrix)

def image_zoom(image):
    center_x, center_y = np.array(image.shape[:2]) / 2
    zoom_size = 200
    zoom_region = image[int(center_x - zoom_size / 2):int(center_x + zoom_size / 2),
                    int(center_y - zoom_size / 2):int(center_y + zoom_size / 2)]
    return transform.resize(zoom_region, image.shape[:2], anti_aliasing=True)

def create_augmentation(image, image_ski):
    rotated = image_random_rotation(image)
    flipped = image_flip(image)
    blured = image_blured(image)
    contrasted = image_contrasted(image_ski)
    zoomed_image = image_zoom(image_ski)
    distorted_image = image_distortion(image_ski)
    return rotated, flipped, blured, contrasted, zoomed_image, distorted_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('path', type=str, help='Path to the image')
    parser.add_argument('--show', action='store_true', help='Show the augmented images')

    args = parser.parse_args()
    
    image_ski = io.imread(args.path)
    image, _ , _ = pcv.readimage(args.path, mode='rgb')
    rotated, flipped, blured, contrasted, zoomed_image, distorted_image = create_augmentation(image, image_ski)

    # save_images = {'rotated': rotated, 'flipped': flipped, 'blured': blured, 'contrasted': contrasted, 'zoomed': zoomed_image, 'distorted': distorted_image}
    # for name, image in save_images.items():
    #     io.imsave(f'{args.path}_{name}.png', image)
    # pcv.print_image(contrasted, 'contrast.png')
    # io.imsave("blabla.png", contrasted., )

    if args.show:
        fig, axes = plt.subplots(1, 7, figsize=(8, 3), gridspec_kw={'wspace': 0.1, 'width_ratios':[1, 1, 1, 1, 1, 1, 1]})
        fig.suptitle('Data Augmentation')
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Rotated')
        axes[1].axis('off')
        axes[2].imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Flipped')
        axes[2].axis('off')
        axes[3].imshow(cv2.cvtColor(blured, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Blured')
        axes[3].axis('off')
        axes[4].imshow(contrasted)
        axes[4].set_title('Contrasted')
        axes[4].axis('off')
        axes[5].imshow(zoomed_image)
        axes[5].set_title('Zoomed')
        axes[5].axis('off')
        axes[6].imshow(distorted_image)
        axes[6].set_title('Distorted')
        axes[6].axis('off')
        plt.show()