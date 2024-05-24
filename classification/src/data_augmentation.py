import uuid
import os
import tensorflow as tf
from src.data_modification import get_modification
from src.data_preprocessing import get_preprocessing


def save_images(dataset, name_category, classes, classe_count, main_classe,
                max_images=512):
    """
    This function saves augmented images to the disk. It iterates over the
    dataset, applies augmentation to each image, and saves it to the
    'augmented_images'directory. The function stops saving images for a
    class once it reaches the 'max_images' limit for that class.

    Parameters:
    dataset (tf.data.Dataset): The dataset containing the images and labels.
    name_category (str): The category name to be used in the saved image file
    name.
    classes (list): The list of class names.
    classe_count (dict): A dictionary keeping track of the number of images
    saved for each class.
    max_images (int, optional): The maximum number of images to save for each
    sclass. Defaults to 512.

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
            image_path = (
                f'learnings/augmented_images/{main_classe}/{name_category}/'
                f'{class_name}/{name_category}_{image_id}.jpg'
            )
            image.save(image_path)
            classe_count[class_name] += 1

            if all([count >= max_images for count in classe_count.values()]):
                break

        if all([count >= max_images for count in classe_count.values()]):
            break

    return classe_count


def check_data(directory, name_category, main_classe, max_images):
    """
    This function checks if the data in the given directory has already been
    augmented. It iterates over the directory and counts the number of images
    in each class. If the number of images is less than the 'max_images' limit
    for each class, the function returns False.

    Parameters:
    directory (str): The directory containing the original images.
    name_category (str): The category name to be used in the saved image file
    name.
    main_classe (str): The main class name to be used in the saved image file
    name.
    max_images (int): The maximum number of images to augment for each class.

    Returns:
    bool: True if the data has already been augmented, False otherwise.
    """

    list = os.listdir(directory)
    classe_count = {class_name: 0 for class_name in list}

    for class_name in list:
        path = (
            f'learnings/augmented_images/{main_classe}/{name_category}/'
            f'{class_name}/'
        )
        if not os.path.exists(path):
            return False
        count = len(os.listdir(path))
        classe_count[class_name] = count

    return all([count >= max_images for count in classe_count.values()])


def augment_data(directory, name_category, max_images=512):
    """
    This function augments the data in the given directory. It first creates
    the necessary directories for saving the augmented images. Then, it
    continuously applies preprocessing and saves the images until it reaches
    the 'max_images' limit for each class.

    Parameters:
    directory (str): The directory containing the original images.
    name_category (str): The category name to be used in the saved image file
    name.
    max_images (int, optional): The maximum number of images to augment for
    each class. Defaults to 512.
    """

    list = os.listdir(directory)
    classe_count = {class_name: 0 for class_name in list}

    main_classe = directory.rstrip('/')
    main_classe = main_classe.split("/")[-1]
    if check_data(directory, name_category, main_classe, max_images):
        msg = (
            f"{name_category} data for {main_classe} set"
            " has already been augmented."
        )
        print(msg)
        return
    os.makedirs(f'learnings/augmented_images/{main_classe}/{name_category}',
                exist_ok=True)
    for class_name in list:
        path = (
            f'learnings/augmented_images/{main_classe}/{name_category}/'
            f'{class_name}'
        )
        os.makedirs(path, exist_ok=True)

    while True:
        dataset = get_preprocessing(directory)
        save_images(dataset, name_category, list, classe_count, main_classe,
                    max_images)
        if all([classe_count[class_name] >= max_images for class_name in
                classe_count]):
            break


if __name__ == "__main__":
    augment_data('../datasets/images/Apple', "train")
