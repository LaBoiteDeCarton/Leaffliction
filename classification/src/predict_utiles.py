import os
from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image_to_white_bg(image):
    """
    Processes the given image to have a white background.

    Args:
        image: The input image.

    Returns:
        result: The processed image with a white background.
    """
    grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_not(binary)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask == np.array([255, 255, 255]),
                      np.array([255, 255, 255]), image)
    result = result.astype(np.uint8)

    return Image.fromarray(result)


def plot_classification(original_image_path, predicted_class):
    """
    Plots the original image and the processed image with the predicted class.

    Args:
        original_image_path: The path to the original image.
        predicted_class: The predicted class.

    Returns:
        None
    """
    original_image = Image.open(original_image_path)
    processed_image = process_image_to_white_bg(original_image)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('black')

    axes[0].imshow(original_image)
    axes[0].axis('off')

    axes[1].imshow(processed_image)
    axes[1].axis('off')

    title = 'DL classification'
    predicted_text = f'Class predicted: {predicted_class}'
    title = f'{"="*3}{" "*45}{title}{" "*45}{"="*3}'
    full_text = f'{title}\n\n{predicted_text}'
    plt.figtext(0.5, 0.01, full_text, ha="center", fontsize=16, color='white')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def get_model(args):
    """
    Retrieves the model to be used for prediction based on the provided
    arguments.

    Args:
        args: The command line arguments.

    Returns:
        model: The loaded model to be used for prediction.
        main_classe: The main class of the model.

    Raises:
        ValueError: If the specified model does not exist or if the path does
        not contain any valid models.
    """
    model_arg = args.model
    main_classe = None
    models = [model.split('_')[0] for model in os.listdir('learnings/models')]
    if model_arg:
        if model_arg in models:
            main_classe = model_arg
        else:
            err = f"Model {model_arg} does not exist"
            raise ValueError(err)
    else:
        path_strip = args.image.split('/')
        for model in models:
            if model in path_strip:
                main_classe = model.split('_')[0]
                break

    if not main_classe:
        err = f"Please choose a path that contains any {models} " + \
         "in the path, or precise the model to use."
        raise ValueError(err)
    print(f'\nModel used: {main_classe}')
    if not os.path.exists(f'learnings/augmented_images/{main_classe}'):
        err = (f"Directory 'learnings/augmented_images/{main_classe}' ",
               "does not exist")
        raise ValueError(err)
    model = load_model(f'learnings/models/{main_classe}_model.keras')
    return model, main_classe
