import os
import tensorflow as tf
from keras.layers import Rescaling # This layer rescales pixel values

seed = 117

def get_preprocessing(directory):
    """
    This function augments the data in the given directory. It first creates the necessary directories for saving the 
    augmented images. Then, it continuously applies preprocessing and saves the images until it reaches the 'max_images' 
    limit for each class.

    Parameters:
    directory (str): The directory containing the original images.
    name_category (str): The category name to be used in the saved image file name.
    max_images (int, optional): The maximum number of images to augment for each class. Defaults to 512.
    """

    list = os.listdir(directory)

    datasets = tf.keras.utils.image_dataset_from_directory(
        directory,  # Directory where the images are located
        labels = 'inferred', # Classes will be inferred according to the structure of the directory
        label_mode = 'categorical', # Labels will be one-hot encoded
        class_names = list, # List of class names
        batch_size = 16,    # Number of processed samples before updating the model's weights
        image_size = (256, 256), # Defining a fixed dimension for all images
        shuffle = True,  # Shuffling data
        seed = seed,  # Random seed for shuffling and transformations
        validation_split = 0, # We don't need to create a validation set from the datasetsing set
        crop_to_aspect_ratio = True # Resize images without aspect ratio distortion
    )

    scaler = Rescaling(1./255) # Defining scaler values between 0 to 1

    datasets = datasets.map(lambda x, y: (scaler(x), y)) 

    return datasets