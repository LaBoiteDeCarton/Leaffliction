
import numpy as np
from utils.preprocessing import get_preprocessing
from utils.cnn import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint # Classes used to save weights and stop training when improvements reach a limit
from keras.models import load_model
from PIL import Image
import os
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Model


def train():
    # Check if the model already exists
    train, test, validation = get_preprocessing()
    if os.path.exists('best_model.keras'):
        # Load the model from the file
        model = load_model('best_model.keras')
    else:
        model = get_model()
        # Defining an Early Stopping and Model Checkpoints
        early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                    patience = 5, mode = 'max',
                                    restore_best_weights = True)

        checkpoint = ModelCheckpoint('best_model.keras',
                                    monitor = 'val_accuracy',
                                    save_best_only = True)
        # Training and Testing Model
        try:
            history = model.fit(
                train, epochs = 50,
                validation_data = test,
                callbacks = [early_stopping, checkpoint])
            
            # Save the model after training
            model.save('best_model.keras')
        except Exception as e:
            print("An error occurred:", e)

    # model.summary()

    # preds = model.predict(validation)  # Running model on the validation dataset
    # val_loss, val_acc = model.evaluate(test) # Obtaining Loss and Accuracy on the val dataset

    # print('\nValidation Loss: ', val_loss)
    # print('\nValidation Accuracy: ', np.round(val_acc * 100), '%')

    # for layer in model.layers:
    #     print(layer.name)

    # layer_name = 'conv2d_4'
    # intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # intermediate_output = intermediate_layer_model.predict(validation)


    # Loading an image from the Validation/ Powdery directory
    image_path = '../datasets/Validation/Powdery/9b6a318cc5721d73.jpg'
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
    labels = ['Healthy', 'Powdery', 'Rust']

    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]

    print(f'\nPredicted Class: {preds_label}')
    print(f'\nConfidence Score: {preds[0][preds_class]}')

if __name__ == "__main__":
    train()