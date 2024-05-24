import os
from keras.models import load_model
import numpy as np
from PIL import Image

from src.data_preprocessing import get_preprocessing



def predict():
    model = load_model('best_model.keras')

    # validation = get_preprocessing("learnings/augmented_images/validation")
    # preds = model.predict(validation)  # Running model on the validation dataset
    # val_loss, val_acc = model.evaluate(validation) # Obtaining Loss and Accuracy on the val dataset

    # print('\nValidation Loss: ', val_loss)
    # print('\nValidation Accuracy: ', np.round(val_acc * 100), '%')

    image_path = '../datasets/images/Apple/Apple_scab/image (342).JPG'
    original_image = Image.open(image_path)
    og_width, og_height = original_image.size

    # Resizing image for optimal performance
    new_width = int(og_width * .20) # 20% of the original size
    new_height = int(og_height * .20) # 20% of the original size

    resized_img = original_image.resize((new_width, new_height))
    print('Picture of a Powdery Plant: \n')
    resized_img

    # Manually preprocessing image
    preprocessed_image = original_image.resize((256, 256))
    preprocessed_image = np.array(preprocessed_image) / 255.0

    preds = model.predict(np.expand_dims(preprocessed_image, axis = 0))
    labels = os.listdir('learnings/augmented_images/validation')

    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]

    print(f'\nPredicted Class: {preds_label}')
    print(f'\nConfidence Score: {preds[0][preds_class]}')


if __name__ == "__main__":
    predict()