import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input


def configure_device():
    """
    Configures the device for TensorFlow execution.

    This function checks for available GPUs and CPUs and sets the memory growth
    and visible devices accordingly. If no GPUs or CPUs are found, it uses the
    default strategy provided by TensorFlow.

    Returns:
        strategy (tf.distribute.Strategy): The TensorFlow distribution strategy
            to be used for execution.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print('\nGPU Found! Using GPU...')
        except RuntimeError as e:
            print(e)
    elif cpus:
        try:
            tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            print('\nCPU Found! Using CPU...')
        except RuntimeError as e:
            print(e)
    else:
        strategy = tf.distribute.get_strategy()
        print('Number of replicas:', strategy.num_replicas_in_sync)

    return strategy


def get_cnn(num_classes):
    """
    Creates and compiles a convolutional neural network model for image
    classification.

    Args:
        num_classes (int): The number of classes for classification.

    Returns:
        model (tf.keras.Model): The compiled CNN model.
    """
    strategy = configure_device()

    with strategy.scope():
        model = Sequential()

        model.add(Input(shape=(256, 256, 3)))  # Input image shape

        # Feature Learning Layers
        model.add(Conv2D(
            32,
            (3, 3),
            strides=1,
            padding='same'
        ))
        model.add(Activation('relu'))  # Activation function
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        # Flattening tensors
        model.add(Flatten())

        # Fully-Connected Layers
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(num_classes, activation='softmax'))

        # Compiling model
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.0001),  # 1e-4
            loss='categorical_crossentropy',  # Ideal for multiclass tasks
            metrics=['accuracy']  # Evaluation metric
        )

        return model
