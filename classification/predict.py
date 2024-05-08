from keras.models import load_model
import numpy as np

from utils.preprocessing import get_preprocessing

def predict():
    model = load_model('final_model.keras')
    train, test, validation = get_preprocessing()
    preds = model.predict(validation)  # Running model on the validation dataset
    val_loss, val_acc = model.evaluate(validation) # Obtaining Loss and Accuracy on the val dataset

    print('\nValidation Loss: ', val_loss)
    print('\nValidation Accuracy: ', np.round(val_acc * 100), '%')


if __name__ == "__main__":
    predict()