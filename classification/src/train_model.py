from src.data_preprocessing import get_preprocessing
from src.cnn import get_model
from keras.callbacks import EarlyStopping, ModelCheckpoint # Classes used to save weights and stop training when improvements reach a limit
from keras.models import load_model
import os

def train_model(train, test, path, num_classes):
    # Check if the model already exists
    model = get_model(num_classes)
    # Defining an Early Stopping and Model Checkpoints
    early_stopping = EarlyStopping(monitor = 'val_accuracy',
                                patience = 5, mode = 'max',
                                restore_best_weights = True)

    checkpoint = ModelCheckpoint(path,
                                monitor = 'val_accuracy',
                                save_best_only = True)
    # Training and Testing Model
    try:
        model.fit(
            train, epochs = 50,
            validation_data = test,
            callbacks = [early_stopping, checkpoint])
        
        # Save the model after training
        model.save(path)
    except Exception as e:
        print("An error occurred:", e)