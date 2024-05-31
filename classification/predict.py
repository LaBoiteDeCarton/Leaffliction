import argparse
import os
import glob
from PIL import Image
import numpy as np
from src.predict_utiles import get_model, plot_classification
from src.data_preprocessing import get_preprocessing
from termcolor import colored
from pyfiglet import Figlet
f = Figlet(font='slant')


def check_image_arg(image):
    if not os.path.exists(image):
        raise ValueError("Image does not exist")
    if not image.lower().endswith(('.jpeg', '.jpg', '.png')):
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
        err = ("something is wrong with the" +
               f" augmented images directory: {str(e)}")
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


def predict(model, main_classe, args):
    if args.accuracy:
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
    parser.add_argument("-m", "--model", type=str,
                        help='Model touse for prediction')
    parser.add_argument("-a", "--accuracy", action='store_true',
                        help='calculate accuracy of the model')
    args = parser.parse_args()
    check_args(args)
    print(colored(f.renderText('Leaffliction :'), 'green') +
          colored(f.renderText('Predict'), 'magenta'))
    try:
        model, main_classe = get_model(args)
        predict(model, main_classe, args)
    except Exception as e:
        print(e)
        exit(1)
