import os
import tensorflow as tf
import uuid

from data_modification import get_modification
from data_preprocessing import get_preprocessing

def save_images(dataset, name_category, classes, classe_count, max_images=512):
    """
    This function saves augmented images to the disk. It iterates over the dataset, applies augmentation to each image,
    and saves it to the 'augmented_images' directory. The function stops saving images for a class once it reaches the 
    'max_images' limit for that class.

    Parameters:
    dataset (tf.data.Dataset): The dataset containing the images and labels.
    name_category (str): The category name to be used in the saved image file name.
    classes (list): The list of class names.
    classe_count (dict): A dictionary keeping track of the number of images saved for each class.
    max_images (int, optional): The maximum number of images to save for each class. Defaults to 512.

    Returns:
    dict: The updated 'classe_count' dictionary.
    """

    augmentation = get_modification()

    for images, labels in dataset:

        for image, label in zip(images, labels):
            class_index = tf.argmax(label)
            class_name = classes[class_index]

            if classe_count[class_name] >= max_images:
                continue

            image = augmentation(image, training=True)
            image = tf.keras.preprocessing.image.array_to_img(image)
            image_id = uuid.uuid4()
            image.save(f'augmented_images/{name_category}/{class_name}/{name_category}_{image_id}.jpg')
            classe_count[class_name] += 1

            if all([count >= max_images for count in classe_count.values()]):
                break

        if all([count >= max_images for count in classe_count.values()]):
            break

    return classe_count


def augment_data(directory, name_category, max_images=512):
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
    classe_count = {class_name: 0 for class_name in list}

    os.makedirs(f'augmented_images/{name_category}', exist_ok=True)
    for class_name in list:
        os.makedirs(f'augmented_images/{name_category}/{class_name}/', exist_ok=True)
    
    while True:
        dataset = get_preprocessing(directory)
        save_images(dataset, name_category, list, classe_count, max_images)
        if all([classe_count[class_name] >= max_images for class_name in classe_count]):
            break


if __name__ == "__main__":
    os.makedirs('augmented_images', exist_ok=True)
    augment_data('../datasets/images/Apple', "train")