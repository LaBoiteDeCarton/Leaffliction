---

# Leaffliction

## Overview

This project aims to develop a convolutional neural network (CNN) for image classification using TensorFlow and Keras. The CNN architecture is designed to classify images into different categories, such as apples and grapes.

## Requirements

Ensure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Overview

The project consists of several components:

- **Data Preprocessing**: The `data_preprocessing.py` script prepares the image data for training by scaling the pixel values and creating TensorFlow datasets.
- **Data Augmentation**: The `data_augmentation.py` script augments the training data to improve model generalization by applying random rotations, brightness adjustments, and contrast changes to the images.
- **Convolutional Neural Network (CNN)**: The `src/cnn.py` module defines the architecture of the convolutional neural network used for classification. It includes convolutional layers followed by max-pooling, batch normalization, dropout, and fully connected layers.
- **Model Training**: The `train_model.py` script trains the CNN model using the augmented training data and evaluates it on the validation set. It utilizes early stopping and model checkpointing to prevent overfitting and save the best-performing model.
- **Prediction**: The `predict.py` script takes an input image and predicts the class of leaf affliction using the trained model. It preprocesses the input image and visualizes the prediction with confidence scores.

## Usage

**Data Preparation**: Ensure your dataset is organized in the following structure:
   ```
   dataset
   ├── class1
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── class2
   │   ├── image1.jpg
   │   └── image2.jpg
   └── ...
   ```

To train the model:

```
python train.py <directory> [--max_images MAX_IMAGES] [--max_images_validation MAX_IMAGES_VALIDATION] [--max_images_test MAX_IMAGES_TEST]
```

- `<directory>`: Path to the directory containing the image data.
- `--max_images`: Maximum number of images for training (default: 512).
- `--max_images_validation`: Maximum number of images for validation (default: 100).
- `--max_images_test`: Maximum number of images for testing (default: 100).

To predict leaf affliction from an image:

```
python predict.py <image_path>
```

- `<image_path>`: Path to the image to be predicted.

## Convolutional Neural Network (CNN) Architecture
The CNN architecture used for image classification consists of multiple convolutional layers followed by fully connected layers. The architecture is designed to automatically learn features from input images and classify them into different categories. The key components of the CNN architecture include:

- **Convolutional Layers**: These layers apply convolutional filters to the input images to extract features such as edges, textures, and patterns.
- **Activation Functions**: Activation functions such as ReLU (Rectified Linear Unit) are used to introduce non-linearity into the model, allowing it to learn complex relationships between features.
- **Pooling Layers**: Pooling layers downsample the feature maps obtained from convolutional layers, reducing the spatial dimensions and the number of parameters in the model.
- **Dropout**: Dropout layers randomly drop a fraction of neurons during training to prevent overfitting and improve generalization.
- **Fully Connected Layers**: These layers receive flattened feature maps from the convolutional layers and perform classification based on the learned features.

---
