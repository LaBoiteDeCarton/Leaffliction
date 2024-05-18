import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D # Convolutional layer
from keras.layers import Activation # Layer for activation functions
from keras.layers import BatchNormalization # This is used to normalize the activations of the neurons.
from keras.layers import MaxPooling2D # Max pooling layer
from keras.layers import Dropout # This serves to prevent overfitting by dropping out a random set of activations.
from keras.layers import Flatten # Layer used to flatten 2D arrays for fully-connected layers.
from keras.layers import Dense # This layer adds fully-connected layers to the neural network.
from keras.layers import Input

from src.data_modification import get_modification




# # Configuring GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  
#         print('\nGPU Found! Using GPU...')
#     except RuntimeError as e:
#         print(e)
# else:
#     strategy = tf.distribute.get_strategy()
#     print('Number of replicas:', strategy.num_replicas_in_sync)

# Configuring CPU for apple silicon

def get_model(num_classes):

    cpus = tf.config.experimental.list_physical_devices('CPU')
    if cpus:
        try:
            tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")  
            print('\nCPU Found! Using CPU...')
        except RuntimeError as e:
            print(e)
    else:
        strategy = tf.distribute.get_strategy()
        print('Number of replicas:', strategy.num_replicas_in_sync)

    with strategy.scope():
        model = Sequential()

        # augmentation = get_modification() # Getting data augmentation pipeline
        model.add(Input(shape=(256,256,3))) # Input image shape
        # model.add(augmentation) # Adding data augmentation pipeline to the model

        # Feature Learning Layers
        model.add(Conv2D(32,                  # Number of filters/Kernels
                        (3,3),               # Size of kernels (3x3 matrix)
                        strides = 1,         # Step size for sliding the kernel across the input (1 pixel at a time).
                        padding = 'same'     # 'Same' ensures that the output feature map has the same dimensions as the input by padding zeros around the input. 
                        ))
        model.add(Activation('relu'))# Activation function
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (5,5), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (5,5), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(512, (3,3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
        model.add(Dropout(0.3))

        # Flattening tensors
        model.add(Flatten())

        # Fully-Connected Layers
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(num_classes, activation = 'softmax')) # Classification layer

        # Compiling model
        model.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001), # 1e-4
                loss = 'categorical_crossentropy', # Ideal for multiclass tasks
                metrics = ['accuracy']) # Evaluation metric
        
        return model