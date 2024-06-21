import argparse
import os
from plantcv import plantcv as pcv
from pathlib import Path
import matplotlib.pyplot as plt

#plt.rcParams["figure.max_open_warning"] = False
pcv.params.debug = "plot"

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
        #height of the image
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
    histogram = pcv.analyze.color(transformation.image, labeled_masks[0], n_labels=1, colorspaces="all", label=None)


