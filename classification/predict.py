import argparse
import os
import glob
from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.data_preprocessing import get_preprocessing


def check_image_arg(image):
    if not os.path.exists(image):
        raise ValueError("Image does not exist")
    if not image.endswith(('.jpeg', '.jpg', '.png')):
        raise ValueError("Image must be a valid image file")


def check_augmented_images_directory():
    try:
        image_files = glob.glob('learnings/augmented_images/**/*.jpeg',
                                recursive=True)
        image_files += glob.glob('learnings/augmented_images/**/*.jpg',
                                 recursive=True)
        image_files += glob.glob('learnings/augmented_images/**/*.png',
                                 recursive=True)
        if not image_files:
            raise ValueError("Images must be valid image files")
    except Exception as e:
        err = ("something is wrong with the augmented ",
               f"images directory: {str(e)}")
        raise ValueError(err)


def check_models_directory():
    if not os.path.exists('learnings/models'):
        raise ValueError("models directory does not exist")
    if not os.listdir('learnings/models'):
        raise ValueError("models directory must not be empty")
    for model in os.listdir('learnings/models'):
        if not model.endswith('.keras'):
            raise ValueError("Models must be keras files")


def check_args(args):
    try:
        check_image_arg(args.image)
        check_augmented_images_directory()
        check_models_directory()
    except ValueError as e:
        print(e)
        exit(1)


def process_image_to_white_bg(image):
    grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_not(binary)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask == np.array([255, 255, 255]),
                      np.array([255, 255, 255]), image)
    result = result.astype(np.uint8)

    return Image.fromarray(result)


def plot_classification(original_image_path, predicted_class):
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


def predict(args):
    main_classe = args.image.rstrip('/')
    if 'Apple' in main_classe:
        main_classe = 'Apple'
    elif 'Grappe' in main_classe:
        main_classe = 'Grappe'
    else:
        err = ("please choose a path that contains 'Apple' or 'Grappe' ",
               "in the path.")
        raise ValueError(err)
    if not os.path.exists(f'learnings/augmented_images/{main_classe}'):
        err = (f"Directory 'learnings/augmented_images/{main_classe}' ",
               "does not exist")
        raise ValueError(err)
    model = load_model(f'learnings/models/{main_classe}_model.keras')

    path = f'learnings/augmented_images/{main_classe}/validation'
    validation = get_preprocessing(path)
    preds = model.predict(validation)
    val_loss, val_acc = model.evaluate(validation)

    print('\nValidation Loss: ', val_loss)
    print('\nValidation Accuracy: ', np.round(val_acc * 100), '%')

    original_image = Image.open(args.image)
    preprocessed_image = np.array(original_image) / 255.0
    preds = model.predict(np.expand_dims(preprocessed_image, axis=0))
    labels = os.listdir(f'learnings/augmented_images/{main_classe}/validation')

    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]

    print(f'\nPredicted Class: {preds_label}')
    print(f'\nConfidence Score: {preds[0][preds_class]}')

    plot_classification(args.image, preds_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a model.')
    parser.add_argument("image", type=str, help='Path to the image to predict')
    args = parser.parse_args()
    check_args(args)
    predict(args)
