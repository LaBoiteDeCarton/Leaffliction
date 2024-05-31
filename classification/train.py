import argparse
import os
from pyfiglet import Figlet
from termcolor import colored
from src.data_preprocessing import get_preprocessing
from src.data_augmentation import augment_data
from src.train_model import train_model
f = Figlet(font='slant')


def check_args(args):
    try:
        if not os.path.isdir(args.directory):
            raise ValueError("directory must be a valid path")
        if not os.listdir(args.directory):
            raise ValueError("directory must not be empty")
        if args.max_images < 0:
            raise ValueError("max_images must be a positive integer")
        if args.max_images_validation < 0:
            raise ValueError("max_images_validation must be positive integer")
        if args.max_images_test < 0:
            raise ValueError("max_images_test must be a positive integer")
    except ValueError as e:
        print(colored(e, 'red'))
        exit(1)


def data_augmentation(directory, max_images, max_images_validation,
                      max_images_test):
    augment_data(directory, "train", max_images)
    augment_data(directory, "validation", max_images_validation)
    augment_data(directory, "test", max_images_test)


def check_learnings():
    if not os.path.exists("learnings"):
        if os.path.exists("learnings.zip"):
            os.system("unzip learnings.zip")


def train(main_classe, num_classes):
    path = f"learnings/models/{main_classe}_model.keras"
    if os.path.exists(path):
        print("Model already exists")
        return
    train_path = f"learnings/augmented_images/{main_classe}/train"
    train = get_preprocessing(train_path)
    test = get_preprocessing(f"learnings/augmented_images/{main_classe}/test")
    train_model(train, test, path, num_classes)
    os.system("zip -r learnings.zip learnings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('directory', type=str, help='Directory of the data')
    parser.add_argument('--max_images', type=int, default=512,
                        help='Maximum number of images')
    parser.add_argument("--max_images_validation", type=int, default=64,
                        help="Maximum number of images for validation")
    parser.add_argument("--max_images_test", type=int, default=64,
                        help="Maximum number of images for test")
    args = parser.parse_args()

    check_args(args)
    check_learnings()
    print(colored(f.renderText('Leaffliction :'), 'green') +
          colored(f.renderText('Train'), 'magenta'))
    data_augmentation(args.directory, args.max_images,
                      args.max_images_validation, args.max_images_test)
    num_classes = len(os.listdir(args.directory))
    main_classe = args.directory.rstrip('/')
    main_classe = main_classe.split("/")[-1]
    train(main_classe, num_classes)
