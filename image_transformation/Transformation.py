import argparse
import os
from plantcv import plantcv as pcv
from pathlib import Path
import matplotlib.pyplot as plt

def transform_gaussian_blur(image):
    #return pcv.gaussian_blur(image,ksize=(5, 5))
    s = pcv.rgb2gray_cmyk(rgb_img=image, channel="y")
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, object_type="light")
    s_gblur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)
    return s_gblur

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('path', type=file_path, help='Path to the image')
    parser.add_argument('--show', action='store_true', help='Show the augmented images')

    args = parser.parse_args()

    file_name = Path(args.path).stem
    image, _ , _ = pcv.readimage(args.path)
    grayscale_image, _ , _ = pcv.readimage(args.path, mode='gray')

    gaussian_blur_image = transform_gaussian_blur(image)
    masked = pcv.apply_mask(image, gaussian_blur_image, 'white')

    # Plot the image color values on a scatterplot of channel 1 vs channel 2 grayscale values
    fig, ax = pcv.visualize.pixel_scatter_plot(paths_to_imgs=[args.path], x_channel="a", y_channel="s")
    plt.show()
    # Use the visualization figure to identify a start and end point that are used to define
    # a line that segments the scatterplot into foreground and background
    mask = pcv.threshold.dual_channels(rgb_img=image, x_channel="a", y_channel="s", points=[(113, 255), (126, 0)], above=True)
    pcv.plot_image(mask)

    cs = pcv.visualize.colorspaces(rgb_img=image, original_img=False)
    pcv.plot_image(cs)
    # Convert the RGB image into a grayscale image by choosing one of the HSV or LAB channels
    grayscale_img = pcv.rgb2gray_cmyk(rgb_img=image, channel="y")
    # Set a binary threshold where the plant pixels will be labeled white and the background will be labeled black
    bin_mask = pcv.threshold.binary(gray_img=grayscale_img, threshold=60, object_type="light")
    pcv.plot_image(bin_mask)
    pcv.plot_image(gaussian_blur_image)
    pcv.plot_image(masked)


