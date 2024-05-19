import argparse
import os


def check_args(args):
    if not os.path.exists(args.image):
        raise ValueError("Image does not exist")
    if not args.image.endswith(('.jpeg', '.jpg', '.png')):
        raise ValueError("Image must be a valid image file")
    if not os.path.exists('learnings'):
        raise ValueError("learnings directory does not exist")
    try:
        for subdirectory in os.listdir('learnings/augmented_images'):
            for subsubdirectory in os.listdir(f'learnings/augmented_images/{subdirectory}'):
                for subsubsubdirectory in os.listdir(f'learnings/augmented_images/{subdirectory}/{subsubdirectory}'):
                    for image in os.listdir(f'learnings/augmented_images/{subdirectory}/{subsubdirectory}/{subsubsubdirectory}'):
                        if not image.endswith(('.jpeg', '.jpg', '.png')):
                            raise ValueError("Images must be valid image files")
    except OSError:
        raise ValueError("something is wrong with the augmented images directory", OSError)
    if not os.path.exists('learnings/models'):
        raise ValueError("models directory does not exist")
    if not os.listdir('learnings/models'):
        raise ValueError("models directory must not be empty")
    for model in os.listdir('learnings/models'):
        if not model.endswith('.keras'):
            raise ValueError("Models must be keras files")


def predict():
    print("Predicting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a model.')
    parser.add_argument("image", type=str, help='Image to predict')
    args = parser.parse_args()
    check_args(args)
    predict()
