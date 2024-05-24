import argparse
import os
import glob
from PIL import Image
from keras.models import load_model
import cv2
import numpy as np

from matplotlib import pyplot as plt
import numpy as np

from src.data_preprocessing import get_preprocessing

def check_image_arg(image):
    if not os.path.exists(image):
        raise ValueError("Image does not exist")
    if not image.endswith(('.jpeg', '.jpg', '.png')):
        raise ValueError("Image must be a valid image file")

def check_augmented_images_directory():
    try:
        image_files = glob.glob('learnings/augmented_images/**/*.jpeg', recursive=True)
        image_files += glob.glob('learnings/augmented_images/**/*.jpg', recursive=True)
        image_files += glob.glob('learnings/augmented_images/**/*.png', recursive=True)
        if not image_files:
            raise ValueError("Images must be valid image files")
    except Exception as e:
        raise ValueError(f"something is wrong with the augmented images directory: {str(e)}")

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
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the image
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Invert the binary image to get the mask
    mask = cv2.bitwise_not(binary)
    
    # Convert the mask to three channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Use the mask to set the background of the original image to white
    result = np.where(mask==np.array([255, 255, 255]), np.array([255, 255, 255]), image)
    
    # Convert the result to an 8-bit unsigned integer array
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)


def plot_classification(original_image_path, predicted_class):
    # Load the original image
    original_image = Image.open(original_image_path)
    
    # Process the image to have a white background
    processed_image = process_image_to_white_bg(original_image)

    # Create a plot with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set the background color to black
    fig.patch.set_facecolor('black')

    # Display the original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    
    # Display the processed image
    axes[1].imshow(processed_image)
    axes[1].axis('off')

    # Add the predicted class annotation
    # Move the title to the bottom
    title = 'DL classification'
    predicted_text = f'Class predicted: {predicted_class}'
    title = f'{"="*3}{" "*49}{title}{" "*49}{"="*3}'
    full_text = f'{title}\n\n{predicted_text}'
    plt.figtext(0.5, 0.01, full_text, ha="center", fontsize=16, color='white')
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust space for the title
    plt.show()


def predict(args):
    model = load_model('learnings/models/Apple_model.keras')
    original_image = Image.open(args.image)
    preds = model.predict(np.expand_dims(original_image, axis = 0))
    labels = os.listdir('learnings/augmented_images/Apple/validation')

    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]

    print(f'\nPredicted Class: {preds_label}')
    print(f'\nConfidence Score: {preds[0][preds_class]}')

    # plt.imshow(original_image)
    # plt.title(f'Predicted Class: {preds_label}')
    # plt.show()    
    plot_classification(args.image, preds_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a model.')
    parser.add_argument("image", type=str, help='Image to predict')
    args = parser.parse_args()
    check_args(args)
    predict(args)
