from src.cnn import get_cnn
from keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(train, test, path, num_classes):
    """
    Trains a model using the given training and testing data.

    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        path (str): The path to save the trained model.
        num_classes (int): The number of classes in the data.

    Returns:
        None
    """

    model = get_cnn(num_classes)

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=5, mode='max',
                                   restore_best_weights=True)

    checkpoint = ModelCheckpoint(path,
                                 monitor='val_accuracy',
                                 save_best_only=True)

    try:
        model.fit(train, epochs=50,
                  validation_data=test,
                  callbacks=[early_stopping, checkpoint])

        model.save(path)
    except Exception as e:
        print("An error occurred:", e)
