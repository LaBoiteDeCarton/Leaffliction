import argparse
import os
from plantcv import plantcv as pcv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class Transformation:
    bin_mask = None
    image = None

    def __init__(self, image):
        self.image = image
        self.create_binary_mask()

    def transform_gaussian_blur(self, image=None):
        if image is None and self.bin_mask is None:
            self.create_binary_mask()
        if image is None:
            image = self.image
        return pcv.gaussian_blur(img=image, ksize=(5, 5), sigma_x=0, sigma_y=None)
    
    def create_binary_mask(self):
        # cs = pcv.visualize.colorspaces(rgb_img=self.image, original_img=False)
        # pcv.plot_image(cs)
        if self.image is None:
            raise ValueError("Image is not set")
        grayscale_m = pcv.rgb2gray_cmyk(rgb_img=self.image, channel="m")
        grayscale_y = pcv.rgb2gray_cmyk(rgb_img=self.image, channel="y")
        mask_m = pcv.threshold.binary(gray_img=grayscale_m, threshold=10, object_type="dark")
        mask_y = pcv.threshold.binary(gray_img=grayscale_y, threshold=20, object_type="light")
        mask = pcv.logical_and(bin_img1=mask_m, bin_img2=mask_y)

        # Fill in small objects (if an object is smaller than the `size` input variable it will get removed)
        # bin_mask = pcv.fill(bin_img=bin_mask_uncleaned, size=10)
        self.bin_mask = mask

        #self.bin_mask = pcv.threshold.gaussian(grayscale_img, ksize=3, offset=1, object_type="dark")
        return mask
    
    def roi_transform(self):
        if self.image is None:
            raise ValueError("Image is not set")
        if self.bin_mask is None:
            self.create_binary_mask()
        return pcv.roi.from_binary_image(self.image, self.bin_mask)
        h, w, _ = self.image.shape
        roi_rect = pcv.roi.rectangle(self.image, x=0, y=0, h=h, w=w)
        # Make a new filtered mask that only keeps the plants in your ROI and not objects outside of the ROI
        # We have set to partial here so that if a leaf partially overlaps but extends outside of your ROI, it 
        # will still be selected. 

        # Inputs for the filtering function:
        #    mask            = the clean mask you made above
        #    roi            = the region of interest you specified above
        #    roi_type       = 'partial' (default, for partially inside the ROI), 'cutto' (hard cut off), or 
        #                     'largest' (keep only largest object)
        filtered_mask  = pcv.roi.filter(mask=self.bin_mask, roi=roi_rect, roi_type='partial')

        from_bin_image = pcv.roi.from_binary_image(self.image, self.bin_mask)
        # print(from_bin_image)
        # pcv.plot_image(from_bin_image)

    def analyze_object_transform(self):
        if self.image is None:
            raise ValueError("Image is not set")
        if self.bin_mask is None:
            self.create_binary_mask()
        shape_img = pcv.analyze.size(img=self.image, labeled_mask=self.bin_mask)
        #pcv.plot_image(shape_img)

    def transform_color_spaces(self):
        return pcv.visualize.colorspaces(rgb_img=self.image, original_img=False)

    def transform_pixel_scatter_plot(self):
        return pcv.visualize.pixel_scatter_plot(paths_to_imgs=[self.image], x_channel="a", y_channel="s")

    def transform_dual_channels(self):
        return pcv.threshold.dual_channels(rgb_img=self.image, x_channel="a", y_channel="s", points=[(113, 255), (126, 0)], above=True)

    def transform_apply_mask(self):
        if self.image is None:
            raise ValueError("Image is not set")
        if self.bin_mask is None:
            self.create_binary_mask()
        return pcv.apply_mask(img=self.image, mask=self.bin_mask, mask_color='white')

    def subjectTransform(self):
        gaussian_blur_transformation = self.transform_gaussian_blur(self.bin_mask)
        mask_transformation = self.transform_apply_mask()
        return gaussian_blur_transformation, mask_transformation
    
    def pseudolandmark_transform(self):
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=self.image, mask=self.bin_mask, label=None)
        left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img=self.image, mask=self.bin_mask, label=None)
        
        # plt.imshow(self.image)
        # plt.scatter(center_v, color='red')
        # plt.show()       
        # print(top, bottom, center_v, left, right, center_h)

    def old_transform_histogram(self):
        if self.image is None:
            raise ValueError("Image is not set")
        _, hist_df = pcv.visualize.histogram(img=self.image, bins=255, hist_data=True)
        l = pcv.rgb2gray_lab(rgb_img=self.image, channel='l')
        a = pcv.rgb2gray_lab(rgb_img=self.image, channel='a')
        b = pcv.rgb2gray_lab(rgb_img=self.image, channel='b')
        h = pcv.rgb2gray_hsv(rgb_img=self.image, channel='h')
        s = pcv.rgb2gray_hsv(rgb_img=self.image, channel='s')
        v = pcv.rgb2gray_hsv(rgb_img=self.image, channel='v')
        _, l_hist_df = pcv.visualize.histogram(l, bins=255, hist_data=True)
        _, a_hist_df = pcv.visualize.histogram(a, bins=255, hist_data=True)
        _, b_hist_df = pcv.visualize.histogram(b, bins=255, hist_data=True)
        _, h_hist_df = pcv.visualize.histogram(h, bins=255, hist_data=True)
        _, s_hist_df = pcv.visualize.histogram(s, bins=255, hist_data=True)
        _, v_hist_df = pcv.visualize.histogram(v, bins=255, hist_data=True)
        hist_df = hist_df[hist_df['proportion of pixels (%)'] != 0]
        l_hist_df = l_hist_df[l_hist_df['proportion of pixels (%)'] != 0]
        a_hist_df = a_hist_df[a_hist_df['proportion of pixels (%)'] != 0]
        b_hist_df = b_hist_df[b_hist_df['proportion of pixels (%)'] != 0]
        h_hist_df = h_hist_df[h_hist_df['proportion of pixels (%)'] != 0]
        s_hist_df = s_hist_df[s_hist_df['proportion of pixels (%)'] != 0]
        v_hist_df = v_hist_df[v_hist_df['proportion of pixels (%)'] != 0]

        plt.plot(l_hist_df['pixel intensity'], l_hist_df['proportion of pixels (%)'], label="lightness", color="darkgreen")
        plt.plot(a_hist_df['pixel intensity'], a_hist_df['proportion of pixels (%)'], label="green-magenta", color="magenta")
        plt.plot(b_hist_df['pixel intensity'], b_hist_df['proportion of pixels (%)'], label="blue-yellow", color="yellow")
        plt.plot(h_hist_df['pixel intensity'], h_hist_df['proportion of pixels (%)'], label="hue", color="darkviolet")
        plt.plot(s_hist_df['pixel intensity'], s_hist_df['proportion of pixels (%)'], label="saturation", color="lightblue")
        plt.plot(v_hist_df['pixel intensity'], v_hist_df['proportion of pixels (%)'], label="value", color="orange")

        for color in hist_df['color channel'].unique():
            subset = hist_df[hist_df['color channel'] == color]
            plt.plot(subset['pixel intensity'], subset['proportion of pixels (%)'], label=color, color=color)
        
        plt.legend()
        plt.title('Histogram of Image')
        plt.show()

    def transform_histogram(self):
        if self.image is None:
            raise ValueError("Image is not set")

        channels = {'h': {'color': "darkviolet", 'label': "hue"}, 's': {'color': "lightblue", 'label': "saturation"}, 'v': {'color': "orange", 'label': "value"}, 'l': {'color': "darkgreen", 'label': "lightness"}, 'a': {'color': "magenta", 'label': "green-magenta"}, 'b': {'color': "yellow", 'label': "blue-yellow"}}
        hist_df = pd.DataFrame()

        for channel in channels:
            color = channels[channel]['color']
            label = channels[channel]['label']
            if channel in ['h', 's', 'v']:
                channel_image = pcv.rgb2gray_hsv(rgb_img=self.image, channel=channel)
            else:
                channel_image = pcv.rgb2gray_lab(rgb_img=self.image, channel=channel)
            _, hist_df_channel = pcv.visualize.histogram(channel_image, bins=255, hist_data=True)
            hist_df_channel.loc[:, 'color channel'] = color
            hist_df_channel.loc[:, 'label'] = label
            hist_df = pd.concat([hist_df, hist_df_channel])
        _, hist_df_rgb = pcv.visualize.histogram(img=self.image, bins=255, hist_data=True)
        hist_df_rgb.loc[:, 'label'] = hist_df_rgb['color channel']
        hist_df = pd.concat([hist_df, hist_df_rgb])
        hist_df = hist_df[hist_df['proportion of pixels (%)'] != 0]

        for color in hist_df['color channel'].unique():
            subset = hist_df[hist_df['color channel'] == color]
            label = subset['label'].iloc[0]
            plt.plot(subset['pixel intensity'], subset['proportion of pixels (%)'], label=label, color=color)
        plt.legend()
        plt.title('Histogram of Image')
        plt.show()


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
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"dst:{path} is not a valid directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('path', type=file_path, help='Path to the image')
    parser.add_argument('--show', action='store_true', help='Show the augmented images')
    parser.add_argument('--src', type=dir_path, help='Path to the source folder')
    parser.add_argument('--dst', type=dir_path, help='Path to the destination folder')

    args = parser.parse_args()

    file_name = Path(args.path).stem
    image, _ , _ = pcv.readimage(args.path)

    if args.show:
        plt.rcParams["figure.max_open_warning"] = False
        pcv.params.debug = "plot"
    
    transformation = Transformation(image)
    gaussian_blur_transformation, mask_transformation = transformation.subjectTransform()
    # pcv.plot_image(image)
    # pcv.plot_image(gaussian_blur_transformation)
    # pcv.plot_image(mask_transformation)
    # pcv.plot_image(transformation.bin_mask)
    transformation.roi_transform()
    transformation.analyze_object_transform()
    transformation.pseudolandmark_transform()
    #analyse color and plot it
    #analyze_color = pcv.analyze.color(image, labeled_masks)
    labeled_masks, num_labels = pcv.create_labels(transformation.bin_mask, rois=None, roi_type="partial")  
    plotting_img = pcv.visualize.colorspaces(image, original_img=True)
    transformation.transform_histogram()
    # histogram = pcv.analyze.color(rgb_img=plotting_img, labeled_mask=labeled_masks, colorspaces="all")
    



